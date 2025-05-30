import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import abc

class RadianceField(nn.Module):
    def __init__(self, output_sdf: bool = False):
        """
        Args:
            output_sdf: indicate that the returned extra part of forward() contains sdf
        """
        super().__init__()
        self.output_sdf = output_sdf

    @abc.abstractmethod
    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, dirs: torch.Tensor | None, cond: torch.Tensor = None) \
            -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Args:
            x: input points [shape, 3 or 4]
            dirs: input directions [shape, 3]
            cond: extra conditions
        Returns:
            rgb [shape, 3], sigma [shape, 1] if applicable, extra outputs as dict
        """
        pass

    # pinf fading support, optional

    def update_fading_step(self, fading_step: int):
        pass

    def print_fading(self):
        pass


class HybridRadianceField(RadianceField):
    def __init__(self, static_model: RadianceField, dynamic_model: RadianceField):
        super().__init__(static_model.output_sdf)
        self.static_model = static_model
        self.dynamic_model = dynamic_model

    def update_fading_step(self, fading_step: int):
        self.static_model.update_fading_step(fading_step)
        self.dynamic_model.update_fading_step(fading_step)

    def print_fading(self):
        print('static: ', end='')
        self.static_model.print_fading()
        print('dynamic: ', end='')
        self.dynamic_model.print_fading()

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        s_static = self.static_model.query_density(x[..., :3], cond, **kwargs)
        s_dynamic = self.dynamic_model.query_density(x, cond)
        return s_static + s_dynamic

    def forward(self, x: torch.Tensor, dirs: torch.Tensor | None, cond: torch.Tensor = None):
        rgb_s, sigma_s, extra_s = self.static_model.forward(x[..., :3], dirs, cond)
        rgb_d, sigma_d, extra_d = self.dynamic_model.forward(x, dirs, cond)
        return self.merge_result(self.output_sdf, rgb_s, sigma_s, extra_s, rgb_d, sigma_d, extra_d)

    @staticmethod
    def merge_result(output_sdf: bool, rgb_s, sigma_s, extra_s, rgb_d, sigma_d, extra_d):
        if output_sdf:
            sigma = sigma_d
            rgb = rgb_d
        else:   # does alpha blend, when delta -> 0
            sigma = sigma_s + sigma_d
            rgb = (rgb_s * sigma_s + rgb_d * sigma_d) / (sigma + 1e-6)

        extra_s |= {
            'rgb_s': rgb_s,
            'rgb_d': rgb_d,
            'sigma_s': sigma_s,
            'sigma_d': sigma_d,
        }
        if len(extra_d) > 0:
            extra_s['dynamic'] = extra_d
        return rgb, sigma, extra_s

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                width = 1 / in_features
            else:
                width = np.sqrt(6 / in_features) / self.omega_0
            self.linear.weight.uniform_(-width, width)

    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


class SIREN_NeRFt(RadianceField):
    def __init__(self, D=8, W=256, input_ch=4, skips=(4,), use_viewdirs=False, first_omega_0=30.0, unique_first=False,
                 fading_fin_step=0, **kwargs):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_NeRFt, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = 3 if use_viewdirs else 0
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0

        hidden_omega_0 = 1.0

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0, is_first=unique_first)] +
            [SineLayer(W, W, omega_0=hidden_omega_0)
             if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D - 1)]
        )

        self.sigma_linear = nn.Linear(W, 1)

        if use_viewdirs:
            self.views_linear = SineLayer(3, W // 2, omega_0=first_omega_0)
            self.feature_linear = SineLayer(W, W // 2, omega_0=hidden_omega_0)
            self.feature_view_linears = nn.ModuleList([SineLayer(W, W, omega_0=hidden_omega_0)])

        self.rgb_linear = nn.Linear(W, 3)

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - radiance_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def query_density_and_feature(self, input_pts: torch.Tensor, cond: torch.Tensor):
        h = input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        sigma = self.sigma_linear(h)
        return F.relu(sigma), h

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.query_density_and_feature(x, cond)[0]

    def forward(self, x, dirs, cond: torch.Tensor = None):
        sigma, h = self.query_density_and_feature(x, cond)

        if self.use_viewdirs:
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(dirs)

            h = torch.cat([input_pts_feature, input_views_feature], -1)

            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        # outputs = torch.cat([rgb, sigma], -1)

        return torch.sigmoid(rgb), sigma, {}

def create_model(name: str, args, input_ch: int = 3) -> RadianceField:
    D = args.netdepth
    W = args.netwidth
    if name == 'nerf':
        raise NotImplementedError("NeRF model is not implemented yet.")
    elif name == 'siren':
        return SIREN_NeRFt(
            D=D, W=W, input_ch=input_ch, use_viewdirs=args.use_viewdirs,
            first_omega_0=args.omega, unique_first=args.use_first_omega,
            fading_fin_step=args.fading_layers
        )
    elif name == 'neus':
        raise NotImplementedError("NeuS model is not implemented yet.")
    elif name == 'nsr':
        raise NotImplementedError("NSR model is not implemented yet.")
    elif name == 'hybrid':  # legacy PINF
        raise NotImplementedError("Hybrid model is not implemented yet.")
    raise NotImplementedError(f"Unknown model name {name}")
