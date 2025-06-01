from .dataset import PINeuFlowDataset, PINeuFlowDatasetValidation
from .network_pinf import RadianceField, HybridRadianceField, SIREN_NeRFt, PINFRenderer


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
