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
        else:  # does alpha blend, when delta -> 0
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


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    device = weights.get_device()
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)

    below = torch.max(torch.zeros_like(inds - 1, device=device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def weighted_sum_of_samples(wei_list: list[torch.Tensor], content: list[torch.Tensor] | torch.Tensor | None):
    if isinstance(content, list):  # list of [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * ct, dim=-2) for weights, ct in zip(wei_list, content)]

    elif content is not None:  # [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * content, dim=-2) for weights in wei_list]

    return [torch.sum(weights, dim=-1) for weights in wei_list]


class NeRFOutputs:
    def __init__(self, rgb_map: torch.Tensor, depth_map: torch.Tensor | None, acc_map: torch.Tensor, **kwargs):
        """
        Args:
            rgb_map: [n_rays, 3]. Estimated RGB color of a ray.
            depth_map: [n_rays]. Depth map. Optional.
            acc_map: [n_rays]. Sum of weights along each ray.
        """
        self.rgb = rgb_map
        self.depth = depth_map
        self.acc = acc_map
        self.extras = kwargs

    def as_tuple(self):
        return self.rgb, self.depth, self.acc, self.extras

    @staticmethod
    def merge(outputs: list["NeRFOutputs"], shape=None, skip_extras=False) -> "NeRFOutputs":
        """Merge list of outputs into one
        Args:
            outputs: Outputs from different batches.
            shape: If not none, reshape merged outputs' first dimension
            skip_extras: Ignore extras when merging, used for merging coarse outputs
        """
        if len(outputs) == 1:  # when training
            return outputs[0]
        extras = {}
        if not skip_extras:
            keys = outputs[0].extras.keys()  # all extras must have same keys
            extras = {k: [] for k in keys}
            for output in outputs:
                for k in keys:
                    extras[k].append(output.extras[k])
            for k in extras:
                assert isinstance(extras[k][0], (torch.Tensor, NeRFOutputs)), \
                    "All extras must be either torch.Tensor or NeRFOutputs when merging"
                if isinstance(extras[k][0], NeRFOutputs):
                    extras[k] = NeRFOutputs.merge(extras[k], shape)  # recursive merging
                elif extras[k][0].dim() == 0:
                    extras[k] = torch.tensor(extras[k]).mean()  # scalar value, reduce to avg
                else:
                    extras[k] = torch.cat(extras[k])

        ret = NeRFOutputs(
            torch.cat([out.rgb for out in outputs]),
            torch.cat([out.depth for out in outputs]) if outputs[0].depth is not None else None,
            torch.cat([out.acc for out in outputs]),
            **extras
        )
        if shape is not None:
            ret.rgb = ret.rgb.reshape(*shape, 3)
            ret.depth = ret.depth.reshape(shape) if ret.depth is not None else None
            ret.acc = ret.acc.reshape(shape)
            for k in ret.extras:
                if isinstance(ret.extras[k], torch.Tensor) and ret.extras[k].dim() > 0:
                    ret.extras[k] = torch.reshape(ret.extras[k], [*shape, *ret.extras[k].shape[1:]])
        return ret

    def add_background(self, background: torch.Tensor):
        """Add background to rgb output
        Args:
            background: scalar or image
        """
        self.rgb = self.rgb + background * (1.0 - self.acc[..., None])
        for v in self.extras.values():
            if isinstance(v, NeRFOutputs):
                v.add_background(background)


class PINFRenderer:
    def __init__(self, model, prop_model, aabb):
        self.model = model
        self.prop_model = prop_model
        self.aabb = aabb

    def render(self,
               rays_o: torch.Tensor,
               rays_d: torch.Tensor,
               rays_t: torch.Tensor,
               near: torch.Tensor,
               far: torch.Tensor,
               background: torch.Tensor,
               chunk: int,
               n_depth_samples: int,
               n_importance: int,
               use_perturb: int,
               ):
        rays_o, rays_d, rays_t, near, far, original_rays_shape = self._validate_inputs(rays_o, rays_d, rays_t, near, far)

        shape = original_rays_shape[:-1]

        ret_list = []
        for batch_start in range(0, rays_d.shape[0], chunk):
            ret = self._render(
                rays_o=rays_o[batch_start:batch_start + chunk],
                rays_d=rays_d[batch_start:batch_start + chunk],
                rays_t=rays_t,
                near=near,
                far=far,
                n_depth_samples=n_depth_samples,
                n_importance=n_importance,
                use_perturb=use_perturb,
            )
            ret_list.append(ret)

        output = NeRFOutputs.merge(ret_list, shape)
        if background is not None:
            output.add_background(background)

        return output

    def _render(self, rays_o: torch.Tensor, rays_d: torch.Tensor, rays_t: torch.Tensor, near: torch.Tensor, far: torch.Tensor, n_depth_samples: int, n_importance: int, use_perturb: int):
        """
        :param rays_o: [n_rays, 3] tensor representing the origin of rays.
        :param rays_d: [n_rays, 3] tensor representing the direction of rays.
        :param rays_t: [1,] tensor representing the time of rays.
        :param near: [1,] tensor representing the near distance for each ray.
        :param far: [1,] tensor representing the far distance for each ray.
        :param n_depth_samples: int, number of depth samples to take along each ray.
        :param n_importance: int, number of importance samples to take. -1 means no importance sampling.
        :param use_perturb: int, if 1, use stratified sampling with perturbation.
        :return:
        """
        n_rays = rays_d.shape[0]
        xyz, z_vals = self._sample_points(rays_o, rays_d, near, far, n_depth_samples, use_perturb)  # [n_rays, n_depth_samples, 3], [n_rays, n_depth_samples]
        view_dirs = self._view_dirs(rays_d, n_depth_samples, use_normalize=True)  # [n_rays, n_depth_samples, 3]

        xyz_flat = xyz.view(-1, 3)  # [n_rays * n_depth_samples, 3]
        view_dirs_flat = view_dirs.view(-1, 3)  # [n_rays * n_depth_samples, 3]

        mask = self.aabb_mask(xyz_flat)  # [n_rays * n_depth_samples,]
        view_dirs_flat_masked = view_dirs_flat[mask]  # [n_valid_points, 3]
        xyz_flat_masked = self._attach_time(xyz_flat[mask], rays_t)  # [n_valid_points, 4]

        rgb_prop_flat, sigma_prop_flat, extra_prop_flat = self._get_raw_with_mask(self.prop_model, xyz_flat_masked, view_dirs_flat_masked, mask)  # [n_rays * n_depth_samples, 3], [n_rays * n_depth_samples, 1]
        rgb_prop = rgb_prop_flat.view(n_rays, n_depth_samples, 3)  # [n_rays, n_depth_samples, 3]
        sigma_prop = sigma_prop_flat.view(n_rays, n_depth_samples, 1)  # [n_rays, n_depth_samples, 1]
        alpha_prop = self.sigma_to_alpha(sigma=sigma_prop, z_vals=z_vals, rays_d_exp=rays_d.unsqueeze(1).expand(n_rays, n_depth_samples, 3))  # [n_rays, n_depth_samples]
        weights_prop = self.alpha_to_weights(alpha=alpha_prop)  # [n_rays, n_depth_samples]
        acc_map_prop = self.weights_to_acc_map(weights=weights_prop)  # [n_rays]
        rgb_map_prop = self.weights_to_rgb_map(weights=weights_prop, rgb=rgb_prop)  # [n_rays, 3]
        prop_output = NeRFOutputs(rgb_map_prop, None, acc_map_prop)

        if n_importance > 0:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [n_rays, n_depth_samples-1]
            weights = weights_prop[..., 1:-1].detach()  # [n_rays, n_depth_samples-2]
            z_samples = sample_pdf(z_vals_mid, weights, n_importance, det=not use_perturb)  # [n_rays, n_importance]
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [n_rays, n_depth_samples + n_importance]

            xyz = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_depth_samples + n_importance, 3]
            view_dirs = self._view_dirs(rays_d, n_depth_samples + n_importance, use_normalize=True)  # [n_rays, n_depth_samples + n_importance, 3]

            xyz_flat = xyz.view(-1, 3)  # [n_rays * (n_depth_samples + n_importance), 3]
            view_dirs_flat = view_dirs.view(-1, 3)

            mask = self.aabb_mask(xyz_flat)  # [n_rays * (n_depth_samples + n_importance),]
            view_dirs_flat_masked = view_dirs_flat[mask]  # [n_valid_points, 3]
            xyz_flat_masked = self._attach_time(xyz_flat[mask], rays_t)  # [n_valid_points, 4]

            # hybrid model outputs
            rgb_flat, sigma_flat, extra_flat = self._get_raw_with_mask(self.model, xyz_flat_masked, view_dirs_flat_masked, mask)  # [n_rays * (n_depth_samples + n_importance), 3], [n_rays * (n_depth_samples + n_importance), 1]
            # rgb = rgb_flat.view(n_rays, n_depth_samples + n_importance, 3)  # [n_rays, n_depth_samples + n_importance, 3]
            # sigma = sigma_flat.view(n_rays, n_depth_samples + n_importance, 1)  # [n_rays, n_depth_samples + n_importance, 1]

            # static outputs
            rgb_static = extra_flat['rgb_s'].view(n_rays, n_depth_samples + n_importance, 3)  # [n_rays, n_depth_samples + n_importance, 3]
            sigma_static = extra_flat['sigma_s'].view(n_rays, n_depth_samples + n_importance, 1)  # [n_rays, n_depth_samples + n_importance, 1]
            alpha_static = self.sigma_to_alpha(sigma=sigma_static, z_vals=z_vals, rays_d_exp=rays_d.unsqueeze(1).expand(n_rays, n_depth_samples + n_importance, 3))  # [n_rays, n_depth_samples + n_importance]
            # weights_static = self.alpha_to_weights(alpha=alpha_static)  # [n_rays, n_depth_samples + n_importance]
            # acc_map_static = self.weights_to_acc_map(weights=weights_static)  # [n_rays]
            # rgb_map_static = self.weights_to_rgb_map(weights=weights_static, rgb=rgb_static)  # [n_rays, 3]

            # dynamic outputs
            rgb_dynamic = extra_flat['rgb_d'].view(n_rays, n_depth_samples + n_importance, 3)  # [n_rays, n_depth_samples + n_importance, 3]
            sigma_dynamic = extra_flat['sigma_d'].view(n_rays, n_depth_samples + n_importance, 1)  # [n_rays, n_depth_samples + n_importance, 1]
            alpha_dynamic = self.sigma_to_alpha(sigma=sigma_dynamic, z_vals=z_vals, rays_d_exp=rays_d.unsqueeze(1).expand(n_rays, n_depth_samples + n_importance, 3))  # [n_rays, n_depth_samples + n_importance]
            # weights_dynamic = self.alpha_to_weights(alpha=alpha_dynamic)  # [n_rays, n_depth_samples + n_importance]
            # acc_map_dynamic = self.weights_to_acc_map(weights=weights_dynamic)  # [n_rays]
            # rgb_map_dynamic = self.weights_to_rgb_map(weights=weights_dynamic, rgb=rgb_dynamic)  # [n_rays, 3]

            # hybrid model algorithm
            alpha_list = [alpha_dynamic, alpha_static]
            color_list = [rgb_dynamic, rgb_static]
            dens = 1.0 - torch.stack(alpha_list, dim=-1)  # [n_rays, n_depth_samples + n_importance, 2]
            dens = torch.cat([dens, torch.prod(dens, dim=-1, keepdim=True)], dim=-1) + 1e-9  # [n_rays, n_depth_samples + n_importance, 3]
            Ti_all = torch.cumprod(dens, dim=-2) / dens  # [n_rays, n_depth_samples + n_importance, 3]
            weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # [n_rays, n_depth_samples + n_importance, 2]
            rgb_map = sum(weighted_sum_of_samples(weights_list, color_list))  # [n_rays, 3]
            acc_map = sum(weighted_sum_of_samples(weights_list, None))  # [n_rays]

            self_weights_list = [alpha_list[alpha_i] * Ti_all[..., alpha_i] for alpha_i in range(len(alpha_list))]
            acc_map_stack = weighted_sum_of_samples(self_weights_list, None)
            rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)

            extra_final = {
                'dynamic': NeRFOutputs(rgb_map_stack[0], None, acc_map_stack[0]),
                'static': NeRFOutputs(rgb_map_stack[1], None, acc_map_stack[1]),
                'sigma_s': sigma_static,
                'sigma_d': sigma_dynamic,
                'coarse': prop_output,
            }
            final_output = NeRFOutputs(rgb_map, None, acc_map, **extra_final)
        else:
            raise NotImplementedError("Importance sampling is not implemented yet.")

        return final_output

    @staticmethod
    def _get_raw_with_mask(model, xyz_flat_masked: torch.Tensor, view_dirs_flat_masked: torch.Tensor, mask: torch.Tensor):
        dtype = xyz_flat_masked.dtype
        device = xyz_flat_masked.device
        n_mask = mask.shape[0]
        rgb_masked, sigma_masked, extra_masked = model(xyz_flat_masked, view_dirs_flat_masked)  # [n_valid_points, 3], [n_valid_points, 1], extra outputs

        # rgb_full = torch.zeros((n_mask, 3), dtype=dtype, device=device).masked_scatter(mask.unsqueeze(-1), rgb_masked)  # [n_rays, n_depth_samples, 3]
        # sigma_full = torch.zeros((n_mask, 1), dtype=dtype, device=device).masked_scatter(mask, sigma_masked)  # [n_rays, n_depth_samples, 1]

        rgb_full = torch.zeros((n_mask, 3), dtype=dtype, device=device)  # [n_rays, n_depth_samples, 3]
        rgb_full[mask] = rgb_masked  # [n_rays, n_depth_samples, 3]
        sigma_full = torch.zeros((n_mask, 1), dtype=dtype, device=device)  # [n_rays, n_depth_samples, 1]
        sigma_full[mask] = sigma_masked  # [n_rays, n_depth_samples, 1]

        extra_full = {}
        for key, value_masked in extra_masked.items():
            assert isinstance(key, str)
            if key.startswith('rgb_'):
                # extra_full[key] = torch.zeros((n_mask, 3), dtype=dtype, device=device).masked_scatter(mask.unsqueeze(-1), value_masked)
                extra_full[key] = torch.zeros((n_mask, 3), dtype=dtype, device=device)  # [n_rays, n_depth_samples, 3]
                extra_full[key][mask] = value_masked  # [n_rays, n_depth_samples, 3]
            elif key.startswith('sigma_'):
                # extra_full[key] = torch.zeros((n_mask, 1), dtype=dtype, device=device).masked_scatter(mask, value_masked)
                extra_full[key] = torch.zeros((n_mask, 1), dtype=dtype, device=device)  # [n_rays, n_depth_samples, 1]
                extra_full[key][mask] = value_masked  # [n_rays, n_depth_samples, 1]
            else:
                raise ValueError(f"Unknown key {key} in extra outputs.")

        return rgb_full, sigma_full, extra_full

    def aabb_mask(self, xyz: torch.Tensor):
        pts = xyz[..., :3]
        aabb = self.aabb
        inside = torch.logical_and(torch.less_equal(aabb[:3], pts), torch.less_equal(pts, aabb[3:]))
        return torch.logical_and(torch.logical_and(inside[..., 0], inside[..., 1]), inside[..., 2])

    @staticmethod
    def sigma_to_alpha(sigma: torch.Tensor, z_vals: torch.Tensor, rays_d_exp: torch.Tensor):
        """
        :param sigma: [n_rays, n_depth_samples, 1] tensor representing the density.
        :param z_vals: [n_rays, n_depth_samples] tensor representing the z-values (depth samples).
        :param rays_d_exp: [n_rays, n_depth_samples, 3] tensor representing the expanded ray directions.
        :return: alpha: [n_rays, n_depth_samples] tensor representing the alpha values.
        """
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [n_rays, n_depth_samples-1]
        dists = torch.cat([dists, dists[..., -1:]], -1)  # [n_rays, n_depth_samples]
        dists = dists * torch.norm(rays_d_exp, dim=-1)  # [n_rays, n_depth_samples]
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
        return alpha

    @staticmethod
    def alpha_to_weights(alpha: torch.Tensor):
        """
        :param alpha: [n_rays, n_depth_samples] tensor representing the alpha values.
        :return: weights: [n_rays, n_depth_samples] tensor representing the weights.
        """
        device = alpha.device
        dtype = alpha.dtype
        n_rays = alpha.shape[0]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((n_rays, 1), dtype=dtype, device=device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        return weights

    @staticmethod
    def weights_to_acc_map(weights: torch.Tensor):
        """
        :param weights: [n_rays, n_depth_samples] tensor representing the weights.
        :return: acc_map: [n_rays] tensor representing the accumulated weights.
        """
        acc_map = torch.sum(weights, dim=-1)
        return acc_map

    @staticmethod
    def weights_to_rgb_map(weights: torch.Tensor, rgb: torch.Tensor):
        """
        :param weights: [n_rays, n_depth_samples] tensor representing the weights.
        :param rgb: [n_rays, n_depth_samples, 3] tensor representing the RGB values.
        :return: rgb_map: [n_rays, 3] tensor representing the RGB map.
        """
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        return rgb_map

    @staticmethod
    def _validate_inputs(rays_o, rays_d, rays_t, near, far):
        """
        :param rays_o: [n_rays, 3] tensor representing the origin of rays.
        :param rays_d: [n_rays, 3] tensor representing the direction of rays.
        :param rays_t: [1,] tensor representing the time of rays.
        :param near: [1,] tensor representing the near distance for each ray.
        :param far: [1,] tensor representing the far distance for each ray.
        :return: rays_o, rays_d, rays_t, near, far, original_rays_shape
        """
        assert isinstance(rays_o, torch.Tensor) and isinstance(rays_d, torch.Tensor) and isinstance(rays_t, torch.Tensor) and isinstance(near, torch.Tensor) and isinstance(far, torch.Tensor), "All inputs must be torch tensors."
        assert rays_o.dtype == rays_d.dtype == rays_t.dtype == near.dtype == far.dtype, "All inputs must have the same dtype."
        assert rays_o.device == rays_d.device == rays_t.device == near.device == far.device, "All inputs must be on the same device."
        assert rays_o.shape[-1] == 3 and rays_d.shape[-1] == 3, "Rays origin and direction must have shape [..., 3]."
        assert rays_t.ndim == near.ndim == far.ndim == 1, "Rays_t, near, far must be 1D tensors."
        original_rays_shape = rays_d.shape

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        return rays_o, rays_d, rays_t, near, far, original_rays_shape

    @staticmethod
    def _sample_points(rays_o: torch.Tensor, rays_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor, n_depth_samples: int, use_perturb: int):
        """
        :param rays_o: [n_rays, 3] tensor representing the origin of rays.
        :param rays_d: [n_rays, 3] tensor representing the direction of rays.
        :param near: [1,] tensor representing the near distance for each ray.
        :param far: [1,] tensor representing the far distance for each ray.
        :param n_depth_samples: int, number of depth samples to take along each ray.
        :param use_perturb: int, if 1, use stratified sampling with perturbation.
        :return: pts: [n_rays, n_depth_samples, 3] tensor representing sampled points along the rays.
        """
        device = rays_d.device
        dtype = rays_d.dtype
        n_rays = rays_d.shape[0]

        t_vals = torch.linspace(0., 1., steps=n_depth_samples, dtype=dtype, device=device).view(1, n_depth_samples)  # [1, n_depth_samples]
        t_vals_exp = t_vals.expand(n_rays, n_depth_samples)  # [n_rays, n_depth_samples], view
        near_exp = near.expand(n_rays, n_depth_samples)  # [n_rays, n_depth_samples], view
        far_exp = far.expand(n_rays, n_depth_samples)  # [n_rays, n_depth_samples], view

        z_vals = near_exp * (1. - t_vals_exp) + far_exp * t_vals_exp  # [n_rays, n_depth_samples]

        if use_perturb:
            mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])  # [N, n_depth_samples-1]
            upper = torch.cat([mids, z_vals[:, -1:]], dim=1)  # [n_rays, n_depth_samples]
            lower = torch.cat([z_vals[:, :1], mids], dim=1)  # [n_rays, n_depth_samples]
            t_rand = torch.rand((n_rays, n_depth_samples), dtype=dtype, device=device)  # [n_rays, n_depth_samples]
            z_vals = lower + (upper - lower) * t_rand  # [n_rays, n_depth_samples]

        z_vals_exp = z_vals.unsqueeze(-1)  # [N, n_depth_samples, 1]
        rays_d_exp = rays_d.unsqueeze(1).expand(n_rays, n_depth_samples, 3)  # [n_rays, n_depth_samples, 3]
        rays_o_exp = rays_o.unsqueeze(1).expand(n_rays, n_depth_samples, 3)  # [n_rays, n_depth_samples, 3]

        pts = rays_o_exp + rays_d_exp * z_vals_exp  # [n_rays, n_depth_samples, 3]

        return pts, z_vals

    @staticmethod
    def _view_dirs(rays_d: torch.Tensor, n_depth_samples: int, use_normalize: bool):
        """
        :param rays_d: [n_rays, 3] tensor representing the direction of rays.
        :param n_depth_samples: int, number of depth samples to take along each ray.
        :param use_normalize: bool, whether to normalize the ray directions.
        :return:
        """
        if use_normalize:
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)  # [n_rays, 3]
        viewdirs = rays_d.unsqueeze(1).expand(-1, n_depth_samples, -1).contiguous()  # [n_rays, n_depth_samples, 3]
        return viewdirs

    @staticmethod
    def _attach_time(xyz: torch.Tensor, t: torch.Tensor):
        """
        :param xyz: [n_batch, 3] tensor representing the points in space.
        :param t: [1,] tensor representing the time.
        :return: xyzt: [n_batch, 4] tensor representing the points in space with time attached.
        """
        assert xyz.ndim == 2 and xyz.shape[-1] == 3, "xyz must be of shape [n_batch, 3]"
        assert t.ndim == 1 and t.shape[0] == 1, "t must be of shape [1]"
        n_batch = xyz.shape[0]
        t_exp = t.view(1, 1).expand(n_batch, 1)
        xyzt = torch.cat([xyz, t_exp], dim=-1)  # [n_batch, 4]
        return xyzt
