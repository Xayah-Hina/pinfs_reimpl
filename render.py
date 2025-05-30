import torch
from model import *
from typing import NamedTuple, Callable


def attach_time(pts: torch.Tensor, t: float):
    return torch.cat([pts, torch.tensor(t, dtype=pts.dtype, device=pts.device).expand(*pts.shape[:-1], 1)], dim=-1)


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


class Rays(NamedTuple):
    origins: torch.Tensor
    viewdirs: torch.Tensor

    def foreach(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        return Rays(fn(self.origins), fn(self.viewdirs))

    def to(self, device):
        return Rays(self.origins.to(device), self.viewdirs.to(device))


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


def weighted_sum_of_samples(wei_list: list[torch.Tensor], content: list[torch.Tensor] | torch.Tensor | None):
    if isinstance(content, list):  # list of [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * ct, dim=-2) for weights, ct in zip(wei_list, content)]

    elif content is not None:  # [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * content, dim=-2) for weights in wei_list]

    return [torch.sum(weights, dim=-1) for weights in wei_list]


def raw2outputs(raw, z_vals, rays_d, mask=None, cos_anneal_ratio=1.0) -> tuple[NeRFOutputs, torch.Tensor]:
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: returned result of RadianceField: rgb, sigma, extra. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        mask: [num_rays, num_samples]. aabb masking
        cos_anneal_ratio: float.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [n_rays, n_samples]
    dists = torch.cat([dists, dists[..., -1:]], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    def sigma2alpha(sigma: torch.Tensor):  # [n_rays, n_samples, 1] -> [n_rays, n_samples, 1]
        if mask is not None:
            sigma = sigma * mask
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # [n_rays, n_samples]
        return alpha

    extra: dict = raw[2]
    gradients = None
    if 'sdf' in extra:
        raise NotImplementedError("SDF rendering is not implemented in this function")
    elif 'sigma_s' in extra:
        alpha_list = [sigma2alpha(extra['sigma_d']), sigma2alpha(extra['sigma_s'])]
        color_list = [extra['rgb_d'], extra['rgb_s']]

    else:
        # shortcut for single model
        alpha = sigma2alpha(raw[1])
        rgb = raw[0]
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]
        # depth_map = torch.sum(weights * z_vals, -1)
        depth_map = None  # unused
        acc_map = torch.sum(weights, -1)
        return NeRFOutputs(rgb_map, depth_map, acc_map), weights

    for key in 'rgb_s', 'rgb_d', 'dynamic':
        extra.pop(key, None)

    dens = 1.0 - torch.stack(alpha_list, dim=-1)  # [n_rays, n_samples, 2]
    dens = torch.cat([dens, torch.prod(dens, dim=-1, keepdim=True)], dim=-1) + 1e-9  # [n_rays, n_samples, 3]
    Ti_all = torch.cumprod(dens, dim=-2) / dens  # [n_rays, n_samples, 3], accu along samples, exclusive
    weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # a list of [n_rays, n_samples]

    rgb_map = sum(weighted_sum_of_samples(weights_list, color_list))  # [n_rays, 3]
    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = sum(weighted_sum_of_samples(weights_list, None))  # [n_rays]

    # Estimated depth map is expected distance.
    # Disparity map is inverse depth.
    # depth_map = sum(weighted_sum_of_samples(weights_list, z_vals[..., None]))  # [n_rays]
    depth_map = None
    # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    # alpha * Ti
    weights = weights_list[0]  # [n_rays, n_samples]

    if len(alpha_list) > 1:  # hybrid model
        self_weights_list = [alpha_list[alpha_i] * Ti_all[..., alpha_i] for alpha_i in
                             range(len(alpha_list))]  # a list of [n_rays, n_samples]
        rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
        acc_map_stack = weighted_sum_of_samples(self_weights_list, None)

        if gradients is not None:
            extra['grad_map'] = weighted_sum_of_samples(self_weights_list[1:], gradients)[0]

        # assume len(alpha_list) == 2 for hybrid model
        extra['dynamic'] = NeRFOutputs(rgb_map_stack[0], None, acc_map_stack[0])
        extra['static'] = NeRFOutputs(rgb_map_stack[1], None, acc_map_stack[1])

    return NeRFOutputs(rgb_map, depth_map, acc_map, **extra), weights


def get_warped_pts(vel_model, orig_pts: torch.Tensor, fading: float, mod: str = "rand") -> torch.Tensor:
    # mod, "rand", "forw", "back", "none"
    if (mod == "none") or (vel_model is None):
        return orig_pts

    orig_pos, orig_t = torch.split(orig_pts, [3, 1], -1)

    with torch.no_grad():
        _vel = vel_model(orig_pts)
    # _vel.shape, [n_rays, n_samples(+n_importance), 3]
    if mod == "rand":
        # random_warpT = np.random.normal(0.0, 0.6, orig_t.get_shape().as_list())
        # random_warpT = np.random.uniform(-3.0, 3.0, orig_t.shape)
        random_warpT = torch.rand_like(orig_t) * 6.0 - 3.0  # [-3,3]
    else:
        random_warpT = 1.0 if mod == "back" else (-1.0)  # back
    # mean and standard deviation: 0.0, 0.6, so that 3sigma < 2, train +/- 2*delta_T
    random_warpT = random_warpT * fading
    # random_warpT = torch.Tensor(random_warpT)

    warp_t = orig_t + random_warpT
    warp_pos = orig_pos + _vel * random_warpT
    warp_pts = torch.cat([warp_pos, warp_t], dim=-1)
    warp_pts = warp_pts.detach()  # stop gradiant

    return warp_pts


def get_warped_raw(model: RadianceField, vel_model, warp_mod, warp_fading_dt, pts, dirs):
    if warp_mod == "none" or None in [vel_model, warp_fading_dt]:
        # no warping
        return model.forward(pts, dirs, pts[..., -1:])

    warp_pts = get_warped_pts(vel_model, pts, warp_fading_dt, warp_mod)
    if not isinstance(model, HybridRadianceField):
        return model.forward(warp_pts, dirs)

    raw_s = model.static_model.forward(pts[..., :3], dirs, pts[..., -1:])
    raw_d = model.dynamic_model.forward(warp_pts, dirs)
    raw = HybridRadianceField.merge_result(model.output_sdf, *raw_s, *raw_d)
    return raw


def mask_from_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    pts = pts[..., :3]
    inside = torch.logical_and(torch.less_equal(aabb[:3], pts), torch.less_equal(pts, aabb[3:]))
    return torch.logical_and(torch.logical_and(inside[..., 0], inside[..., 1]), inside[..., 2]).unsqueeze(-1)


class PINFRenderer:
    def __init__(self,
                 model: RadianceField,
                 prop_model: RadianceField | None,
                 n_samples: int,
                 n_importance: int = 0,
                 near: torch.Tensor | float = 0.0,
                 far: torch.Tensor | float = 1.0,
                 perturb: bool = False,
                 vel_model=None,
                 warp_fading_dt=None,
                 warp_mod="rand",
                 aabb: torch.Tensor = None,
                 ):
        """Volumetric rendering.
        Args:
          model: Model for predicting RGB and density at each pointin space.
          n_samples: int. Number of different times to sample along each ray.
          n_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          warp_fading_dt, to train nearby frames with flow-based warping, fading*delt_t
        """
        self.model = model
        self.prop_model = prop_model
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.near = near
        self.far = far
        self.perturb = perturb
        self.vel_model = vel_model
        self.warp_fading_dt = warp_fading_dt
        self.warp_mod = warp_mod
        self.aabb = aabb
        self.cos_anneal_ratio = 1.0

    def run(self, rays: Rays, rays_t: float | None, near: torch.Tensor, far: torch.Tensor,
            ret_raw: bool = False, perturb: bool = None, ignore_vel: bool = False
            ) -> NeRFOutputs:

        n_samples = self.n_samples
        n_importance = self.n_importance
        model = self.model
        prop_model = self.prop_model if self.prop_model is not None else model
        vel_model = None if ignore_vel else self.vel_model
        warp_mod = self.warp_mod
        warp_fading_dt = self.warp_fading_dt
        if perturb is None:
            perturb = self.perturb

        rays_o, rays_d = rays  # [n_rays, 3] each
        n_rays = rays_o.shape[0]

        t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.expand([n_rays, n_samples])

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand_like(z_vals)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
        if rays_t is not None:
            pts = attach_time(pts, rays_t)
        viewdirs = rays_d[..., None, :].expand(*pts.shape[:-1], -1)

        # raw = run_network(pts)
        raw = get_warped_raw(prop_model, vel_model, warp_mod, warp_fading_dt, pts, viewdirs)
        mask = mask_from_aabb(pts, self.aabb)
        outputs, weights = raw2outputs(raw, z_vals, rays_d, mask, cos_anneal_ratio=self.cos_anneal_ratio)

        if n_importance > 0:
            outputs_0 = outputs

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            weights = weights[..., 1:-1].detach()

            if self.model.output_sdf:
                raise NotImplementedError("SDF rendering is not implemented in this function")
            else:
                z_samples = sample_pdf(z_vals_mid, weights, self.n_importance, det=not perturb)

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            # [n_rays, n_samples + n_importance, 3]
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            if rays_t is not None:
                pts = attach_time(pts, rays_t)
            viewdirs = rays_d[..., None, :].expand(*pts.shape[:-1], -1)

            raw = get_warped_raw(model, vel_model, warp_mod, warp_fading_dt, pts, viewdirs)
            mask = mask_from_aabb(pts, self.aabb)
            outputs, _ = raw2outputs(raw, z_vals, rays_d, mask, cos_anneal_ratio=self.cos_anneal_ratio)
            outputs.extras['coarse'] = outputs_0

        if not ret_raw:
            outputs.extras = {k: outputs.extras[k] for k in outputs.extras
                              if k.endswith('map') or isinstance(outputs.extras[k], NeRFOutputs)}

        return outputs

    def render(self, rays_o, rays_d, chunk=1024 * 32,
               timestep: float = None, background=None,
               **kwargs) -> NeRFOutputs:
        """Render rays
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          focal: float. Focal length of pinhole camera.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """
        shape = rays_d.shape[:-1]  # [..., 3]

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        rays = Rays(rays_o, rays_d)

        near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])

        # Render and reshape
        ret_list = []
        for i in range(0, rays_o.shape[0], chunk):
            rays_chunk = rays.foreach(lambda t: t[i: i + chunk])
            ret = self.run(rays_chunk, timestep, near=near[i: i + chunk], far=far[i: i + chunk], **kwargs)
            ret_list.append(ret)

        output = NeRFOutputs.merge(ret_list, shape)
        if background is not None:
            output.add_background(background)

        return output
