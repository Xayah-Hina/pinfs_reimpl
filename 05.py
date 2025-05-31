# remove precrop part
import tensorboardX
import os
import sys
import shutil
from datetime import datetime

from tqdm import tqdm
from render import *
from native import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img2mse(x: torch.Tensor, y: torch.Tensor):
    return torch.mean((x - y) ** 2)


def mse2psnr(x: torch.Tensor):
    return -10. * torch.log10(x)


def fade_in_weight(step, start, duration):
    return min(max((float(step) - start) / duration, 0.0), 1.0)


def model_fading_update(model: RadianceField, prop_model: RadianceField | None, vel_model: None,
                        global_step, vel_delay):
    model.update_fading_step(global_step)
    if prop_model is not None:
        prop_model.update_fading_step(global_step)
    if vel_model is not None:
        raise NotImplementedError("Velocity model fading is not implemented yet.")


def pos_smoke2world(Psmoke, s2w):
    pos_scale = Psmoke  # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape)  # 3.target to 4.world
    return pos_rot + pos_off


def convert_aabb(in_min, in_max, voxel_tran):
    in_min = torch.tensor(in_min, device=voxel_tran.device).expand(3)
    in_max = torch.tensor(in_max, device=voxel_tran.device).expand(3)
    in_min = pos_smoke2world(in_min, voxel_tran)
    in_max = pos_smoke2world(in_max, voxel_tran)
    cmp = torch.less(in_min, in_max)
    in_min, in_max = torch.where(cmp, in_min, in_max), torch.where(cmp, in_max, in_min)
    return torch.cat((in_min, in_max))


class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a")
        cmdline = " ".join(sys.argv) + "\n"
        self.log.write(cmdline)

    def write(self, message):
        if not self.silent:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def printENV():
    check_list = ['CUDA_VISIBLE_DEVICES']
    for name in check_list:
        if name in os.environ:
            print(name, os.environ[name])
        else:
            print(name, "Not find")

    sys.stdout.flush()


def set_rand_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')

    # data info
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # blender flags
    parser.add_argument("--half_res", type=str, default='normal',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # network arch
    parser.add_argument("--net_model", type=str, default='nerf',
                        help='which model to use, nerf, siren...')
    parser.add_argument("--s_model", type=str, default='',
                        help='which model to use for static part, nerf, siren, neus...')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=0,  # 10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=0,  # 4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--omega", type=float, default=30.0,
                        help="first_omega_0 in SIREN")
    parser.add_argument("--use_first_omega", action="store_true",
                        help="enable is_first in SIREN")
    parser.add_argument("--vel_no_slip", action='store_true',
                        help="use no-slip boundray in velocity training")
    parser.add_argument("--use_color_t", action='store_true',
                        help="use time input in static part's color net")

    # network save load
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--fix_seed", type=int, default=42,
                        help='the random seed.')

    # train params - sampling
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=4096,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--train_warp", default=False, action='store_true',
                        help='train radiance model with velocity warpping')
    parser.add_argument("--vol_output_W", type=int, default=256,
                        help='In output mode: the output resolution along x; In training mode: the sampling resolution for training')

    # train params - iterations
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--fading_layers", type=int, default=-1,
                        help='for siren and hybrid models, the step to finish fading model layers one by one during training.')
    parser.add_argument("--tempo_fading", type=int, default=2000,
                        help='for hybrid model, how many steps try to use static model to represent whole scene')
    parser.add_argument("--vel_delay", type=int, default=10000,
                        help='for siren and hybrid models, the step to start learning the velocity.')
    parser.add_argument("--N_iter", type=int, default=20000,
                        help='for siren and hybrid models, the step to start learning the velocity.')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=400,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=2000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=25000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # train params - loss weights
    parser.add_argument("--vgg_strides", type=int, default=4,
                        help='vgg stride, should >= 2')
    parser.add_argument("--ghostW", type=float,
                        default=-0.0, help='weight for the ghost density regularization')
    parser.add_argument("--ghost_scale", type=float,
                        default=4.0, help='tolerance for the ghost density regularization')
    parser.add_argument("--vggW", type=float,
                        default=-0.0, help='weight for the VGG loss')
    parser.add_argument("--overlayW", type=float,
                        default=-0.0, help='weight for the overlay regularization')
    parser.add_argument("--nseW", type=float,
                        default=0.001, help='velocity model, training weight for the physical equations')
    parser.add_argument("--eikonal", type=float,
                        default=0.01, help='weight for eikonal loss')
    parser.add_argument("--devW", type=float,
                        default=0.0, help='weight for deviation loss')
    parser.add_argument("--neumann", type=float,
                        default=0.0, help='weight for neumann loss')

    # scene params
    parser.add_argument("--bbox_min", type=str,
                        default='', help='use a boundingbox, the minXYZ')
    parser.add_argument("--bbox_max", type=str,
                        default='1.0,1.0,1.0', help='use a boundingbox, the maxXYZ')
    parser.add_argument("--near", type=float,
                        default=-1.0, help='near plane in rendering, <0 use scene default')
    parser.add_argument("--far", type=float,
                        default=-1.0, help='far plane in rendering, <0 use scene default')

    # task params
    parser.add_argument("--vol_output_only", action='store_true',
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    return parser


def prepare_logging(expdir, args):
    # logs
    os.makedirs(expdir, exist_ok=True)
    date_str = datetime.now().strftime("%m%d-%H%M%S")
    filedir = 'train' if not (args.vol_output_only or args.render_only) else 'test'
    filedir += date_str
    logdir = os.path.join(expdir, filedir)
    os.makedirs(logdir, exist_ok=True)

    sys.stdout = Logger(logdir, False, fname="log.out")
    # sys.stderr = Logger(log_dir, False, fname="log.err")  # for tqdm

    print(" ".join(sys.argv), flush=True)
    printENV()

    # files backup
    shutil.copyfile(args.config, os.path.join(expdir, filedir, 'config.txt'))
    f = os.path.join(logdir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    # filelist = ['run_pinf.py', 'run_pinf_helpers.py', 'pinf_rendering.py',
    #             # 'nerf/utils.py',
    #             # 'radiance_fields/nerf.py',
    #             'radiance_fields/siren.py',
    #             'radiance_fields/neus.py',
    #             ]
    # if args.dataset_type == 'nv_data':
    #     filelist.append('datasets/neural_volumes.py')
    # for filename in filelist:
    #     shutil.copyfile('./' + filename, os.path.join(logdir, filename.replace("/", "-")))

    return logdir


def save_model(path: str, global_step: int,
               model: RadianceField, prop_model: RadianceField | None, optimizer: torch.optim.Optimizer,
               vel_model=None, vel_optimizer=None):
    save_dic = {
        'global_step': global_step,
        'network_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if prop_model is not None:
        save_dic['network_prop_state_dict'] = prop_model.state_dict()

    if vel_model is not None:
        save_dic['network_vel_state_dict'] = vel_model.state_dict()
        save_dic['vel_optimizer_state_dict'] = vel_optimizer.state_dict()

    torch.save(save_dic, path)


def get_rays(K: np.ndarray, c2w: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> Rays:
    dirs = torch.stack([(xs - K[0, 2]) / K[0, 0], -(ys - K[1, 2]) / K[1, 1], -torch.ones_like(xs)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize directions
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return Rays(rays_o, rays_d)


def vgg_sample(vgg_strides: int, num_rays: int, frame: torch.Tensor, bg_color: torch.Tensor, dw: int = None,
               steps: int = None):
    if steps is None:
        strides = vgg_strides + np.random.randint(-1, 2)  # args.vgg_strides(+/-)1 or args.vgg_strides
    else:
        strides = vgg_strides + steps % 3 - 1
    H, W = frame.shape[:2]
    if dw is None:
        dw = max(20, min(40, int(np.sqrt(num_rays))))
    vgg_min_border = 10
    strides = min(strides, min(H - vgg_min_border, W - vgg_min_border) / dw)
    strides = int(strides)

    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij'),
                         dim=-1).to(frame.device)  # (H, W, 2)
    target_grey = torch.mean(torch.abs(frame - bg_color), dim=-1, keepdim=True)  # (H, W, 1)
    img_wei = coords.to(torch.float32) * target_grey
    center_coord = torch.sum(img_wei, dim=(0, 1)) / torch.sum(target_grey)
    center_coord = center_coord.cpu().numpy()
    # add random jitter
    random_R = dw * strides / 2.0
    # mean and standard deviation: center_coord, random_R/3.0, so that 3sigma < random_R
    random_x = np.random.normal(center_coord[1], random_R / 3.0) - 0.5 * dw * strides
    random_y = np.random.normal(center_coord[0], random_R / 3.0) - 0.5 * dw * strides

    offset_w = int(min(max(vgg_min_border, random_x), W - dw * strides - vgg_min_border))
    offset_h = int(min(max(vgg_min_border, random_y), H - dw * strides - vgg_min_border))

    coords_crop = coords[offset_h:offset_h + dw * strides:strides, offset_w:offset_w + dw * strides:strides, :]
    return coords_crop, dw


def create_models_optis(args, input_ch):
    model = create_model(args.net_model, args, input_ch).to(device)
    grad_vars = list(model.parameters())
    if args.s_model != '':
        model_s = create_model(args.s_model, args, input_ch - 1).to(device)
        grad_vars += model_s.parameters()
        model = HybridRadianceField(model_s, model)

    prop_model = None
    if args.N_importance > 0:
        prop_model = create_model(args.net_model, args, input_ch).to(device)
        grad_vars += list(prop_model.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    return model, prop_model, optimizer


def pinf_train():
    args = config_parser().parse_args()
    set_rand_seed(args.fix_seed)
    #########################################################
    dataset_native = PINeuFlowDataset(dataset_path='data/sphere', dataset_type='train', downscale=1, use_fp16=False, device=device)

    bkg_color = torch.tensor([1.0, 1.0, 1.0], device=device)  # default background color  aabb = tensor([-2.4200, -2.3426, -2.7100,  2.3826,  2.4600,  2.0926]) 0.00632911

    voxel_tran = dataset_native.extra_params.voxel_transform
    voxel_scale = dataset_native.extra_params.voxel_scale
    near = dataset_native.extra_params.near  # NOTICE ! this could be overridden by the config files
    far = dataset_native.extra_params.far  # NOTICE ! this could be overridden by the config files
    H = dataset_native.extra_params.height
    W = dataset_native.extra_params.width
    voxel_tran[:3, :3] *= voxel_scale
    aabb = convert_aabb(
        in_min=[0.0],
        in_max=[1.0],
        voxel_tran=voxel_tran,
    ).to(device)
    #########################################################

    # Create log dir and copy the config file
    expdir: str = os.path.join(args.basedir, args.expname)
    logdir: str = prepare_logging(expdir, args)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    # aabb, train_data, bkg_color, t_info = create_aabb(args)
    model, prop_model, optimizer = create_models_optis(args, input_ch=4)

    renderer = PINFRenderer(
        model=model,
        prop_model=prop_model,
        n_samples=args.N_samples,
        n_importance=args.N_importance,
        near=near,
        far=far,
        perturb=args.perturb > 0,
        vel_model=None,
        aabb=aabb,
    )

    start = 0
    start = start + 1
    global_step = start
    model_fading_update(model, prop_model, None, start, None)

    dataloader_native = dataloader_simple(dataset=dataset_native)
    for epoch in range(0, 20):
        torch.set_default_tensor_type('torch.FloatTensor')
        for i_loader, data in tqdm(enumerate(dataloader_native), total=len(dataloader_native)):
            i = global_step

            model_fading_update(model, prop_model, None, global_step, None)
            tempo_fading = fade_in_weight(global_step, 0, args.tempo_fading)

            focal = data['focals'][0].item()
            K =np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ], dtype=np.float32)
            pose = data['poses'][0]
            time_locate = data['times'][0].item()
            target = data['images'][0]

            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                xs, ys = torch.meshgrid(
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    indexing='xy'
                )
                selected = np.random.choice(4 * dH * dW, size=[args.N_rand], replace=False)
                if i == start:
                    print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
            else:
                xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
                selected = np.random.choice(H * W, size=[args.N_rand], replace=False)
            xs = torch.flatten(xs)[selected].to(device)
            ys = torch.flatten(ys)[selected].to(device)

            rays = get_rays(K, pose, xs, ys)
            rays = rays.foreach(lambda t: t.to(device))
            target_s = target[ys.long(), xs.long()]
            output = renderer.render(
                rays.origins, rays.viewdirs, chunk=args.chunk,
                ret_raw=True,
                timestep=time_locate,
                background=bkg_color)
            rgb, _, acc, extras = output.as_tuple()

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            if 'static' in extras and tempo_fading < 1.0 - 1e-8:
                img_loss = img_loss * tempo_fading + img2mse(extras['static'].rgb, target_s) * (1.0 - tempo_fading)
            loss = img_loss

            writer.add_scalar(f"Loss/img_loss_d", img2mse(rgb, target_s), i)
            writer.add_scalar(f"Loss/img_loss_s", img2mse(extras['static'].rgb, target_s), i)
            writer.add_scalar(f"Loss/img_loss", img_loss, i)

            loss.backward()
            optimizer.step()

            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if (i in (1000, 3000, 5000, 7000, 9000, 10000, 20000, 40000) or i % args.i_weights == 0) and i > start + 1:
                path = os.path.join(expdir, '{:06d}.tar'.format(i))
                save_model(path, global_step, model, prop_model, optimizer, None, None)
                print('Saved checkpoints at', path)
            global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    pinf_train()
