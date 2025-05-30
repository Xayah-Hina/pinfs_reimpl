import tensorboardX
import os
import time
import sys
import shutil
import imageio
import cv2
import json
from datetime import datetime
from tqdm import tqdm, trange
from render import *

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


def intrinsics_from_hwf(H: int, W: int, focal: float):
    return np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)


def trans_t(t: float):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_phi(phi: float):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_theta(th: float):
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def pose_spherical(theta: float, phi: float, radius: float, rotZ=True, center: np.ndarray = None):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # center: additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @ c2w

    if center is not None:
        c2w[:3, 3] += center
    return c2w


class VideoData:
    def __init__(self, args: dict | None, basedir: str = '', half_res: str = None):
        if args is None:
            self.delta_t = 1.0
            self.transform_matrix = np.empty(0)
            self.frames = np.empty(0)
            self.focal = 0.0
            return

        filename = os.path.join(basedir, args['file_name'])
        meta = imageio.v3.immeta(filename)
        reader = imageio.imiter(filename)

        frame_rate = args.get('frame_rate', meta['fps'])
        frame_num = args.get('frame_num')
        if not np.isfinite(frame_num):
            frame_num = meta['nframes']
            if not np.isfinite(frame_num):
                frame_num = meta['duration'] * meta['fps']
            frame_num = round(frame_num)

        self.delta_t = 1.0 / frame_rate
        if 'transform_matrix' in args:
            self.transform_matrix = np.array(args['transform_matrix'], dtype=np.float32)
        else:
            self.transform_matrix = np.array(args['transform_matrix_list'], dtype=np.float32)

        frames = tuple(reader)[:frame_num]
        H, W = frames[0].shape[:2]
        if half_res == 'half':
            H //= 2
            W //= 2
        elif half_res == 'quarter':
            H //= 4
            W //= 4
        elif half_res is not None:
            if half_res != 'normal':
                print("Unsupported half_res value", half_res)
            half_res = None

        if half_res is not None:
            frames = [cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) for frame in frames]
        self.frames: np.ndarray = np.float32(frames) / 255.0
        self.focal = float(0.5 * self.frames.shape[2] / np.tan(0.5 * args['camera_angle_x']))

    def c2w(self, frame: int = None) -> np.ndarray:
        if self.transform_matrix.ndim == 2 or frame is None:
            return self.transform_matrix
        return self.transform_matrix[frame]

    def intrinsics(self):
        return intrinsics_from_hwf(self.frames.shape[1], self.frames.shape[2], self.focal)

    def __len__(self) -> int:
        return self.frames.shape[0]


class PINFFrameDataBase:
    def __init__(self):
        # placeholders
        self.voxel_tran: np.ndarray | None = None
        self.voxel_scale: np.ndarray | None = None
        self.videos: dict[str, list[VideoData]] = {}
        self.t_info: np.ndarray | None = None
        self.render_poses: np.ndarray | None = None
        self.render_timesteps: np.ndarray | None = None
        self.bkg_color: np.ndarray | None = None
        self.near, self.far = 0.0, 1.0


class PINFFrameData(PINFFrameDataBase):
    def __init__(self, basedir: str, half_res: str | bool = None, normalize_time: bool = False,
                 apply_tran: bool = False, **kwargs):
        super().__init__()
        with open(os.path.join(basedir, 'info.json'), 'r') as fp:
            # read render settings
            meta = json.load(fp)
        near = float(meta['near'])
        far = float(meta['far'])
        radius = (near + far) * 0.5
        phi = float(meta['phi'])
        rotZ = (meta['rot'] == 'Z')
        r_center = np.float32(meta['render_center'])
        bkg_color = np.float32(meta['frame_bkg_color'])
        if isinstance(half_res, bool):  # compatible with nerf
            half_res = 'half' if half_res else None

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]],
                              axis=1)  # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'], [3]).astype(np.float32)

        if apply_tran:
            voxel_tran[:3, :3] *= voxel_scale[0]
            scene_tran = np.linalg.inv(voxel_tran)
            voxel_tran = np.eye(4, dtype=np.float32)
            voxel_scale /= voxel_scale[0]
            near, far = 0.1, 2.0  # TODO apply conversion

        else:
            scene_tran = None

        self.voxel_tran: np.ndarray = voxel_tran
        self.voxel_scale: np.ndarray = voxel_scale

        self.videos: dict[str, list[VideoData]] = {
            'train': [],
            'test': [],
            'val': [],
        }

        # read video frames
        # all videos should be synchronized, having the same frame_rate and frame_num
        for s in ('train', 'val', 'test'):
            video_list = meta[s + '_videos'] if (s + '_videos') in meta else []

            for train_video in video_list:
                video = VideoData(train_video, basedir, half_res=half_res)
                self.videos[s].append(video)

            if len(video_list) == 0:
                self.videos[s] = self.videos['train'][:1]

        self.videos['test'] += self.videos['val']  # val vid not used for now
        self.videos['test'] += self.videos['train']  # for test
        video = self.videos['train'][0]
        # assume identical frame rate and length
        if normalize_time:
            self.t_info = np.float32([0.0, 1.0, 1.0 / len(video)])
        else:
            self.t_info = np.float32([0.0, video.delta_t * len(video), video.delta_t])  # min t, max t, delta_t

        # set render settings:
        sp_n = 40  # an even number!
        sp_poses = [
            pose_spherical(angle, phi, radius, rotZ, r_center)
            for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
        ]

        if scene_tran is not None:
            for vk in self.videos:
                for video in self.videos[vk]:
                    video.transform_matrix = scene_tran @ video.transform_matrix
            sp_poses = [scene_tran @ pose for pose in sp_poses]

        self.render_poses = np.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
        self.render_timesteps = np.linspace(self.t_info[0], self.t_info[1], num=sp_n).astype(np.float32)
        self.bkg_color = bkg_color
        self.near, self.far = near, far


class PINFDataset:
    def __init__(self, base: PINFFrameDataBase, split: str = 'train'):
        super().__init__()
        self.base = base
        self.videos = self.base.videos[split]

    def __len__(self):
        return len(self.videos) * len(self.videos[0])

    def get_video_and_frame(self, item: int) -> tuple[VideoData, int]:
        vi, fi = divmod(item, len(self.videos[0]))
        video = self.videos[vi]
        return video, fi


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


def pinf_train():
    args = config_parser().parse_args()
    set_rand_seed(args.fix_seed)

    # Create log dir and copy the config file
    expdir: str = os.path.join(args.basedir, args.expname)
    logdir: str = prepare_logging(expdir, args)
    writer = tensorboardX.SummaryWriter(logdir=logdir)

    time0 = time.time()
    if args.dataset_type == 'pinf_data':
        pinf_data = PINFFrameData(args.datadir, half_res=args.half_res, normalize_time=True)

    elif args.dataset_type == 'nv_data':
        raise NotImplementedError("Neural Volumes dataset is not implemented yet.")

    else:
        raise NotImplementedError(f"Unsupported dataset type {args.dataset_type}")

    train_data = PINFDataset(pinf_data)

    print(f'Loading takes {time.time() - time0:.4f} s')
    time0 = time.time()

    voxel_tran = pinf_data.voxel_tran
    voxel_scale = pinf_data.voxel_scale

    bkg_color = torch.Tensor(pinf_data.bkg_color).to(device)
    near, far = pinf_data.near, pinf_data.far
    if args.near > 0:
        near = args.near
    if args.far > 0:
        far = args.far
    t_info = pinf_data.t_info

    print(f'Conversion takes {time.time() - time0:.4f} s')

    # Load data
    # images, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far = load_pinf_frame_data(args.datadir, args.half_res, args.testskip)
    print('Loaded pinf frame data', args.datadir)
    print('Loaded voxel matrix', voxel_tran, 'voxel scale', voxel_scale)

    voxel_tran[:3, :3] *= voxel_scale
    voxel_tran = torch.Tensor(voxel_tran).to(device)
    del voxel_scale

    print('Scene has background color', bkg_color)

    # Create Bbox model
    if args.bbox_min != "":
        in_min = [float(x) for x in args.bbox_min.split(",")]
        in_max = [float(x) for x in args.bbox_max.split(",")]
        aabb = convert_aabb(in_min, in_max, voxel_tran)
        print(f"aabb = {aabb}")
    else:
        aabb = None

    # Create nerf model
    input_ch = 4

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

    start = 0

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

    global_step = start

    model_fading_update(model, prop_model, None, start, None)

    n_rand = args.N_rand

    # Move to GPU, except images
    # poses = torch.Tensor(poses).to(device)
    # timesteps = torch.Tensor(timesteps).to(device)

    n_iters = args.N_iter + 1

    print('Begin')

    start = start + 1

    testimgdir = logdir + "_imgs"
    os.makedirs(testimgdir, exist_ok=True)
    psnr1k = np.zeros(1000)

    for i in trange(start, n_iters):
        model_fading_update(model, prop_model, None, global_step, None)

        # train radiance all the time, train vel less, train with d2v even less.
        trainVGG = (args.vggW > 0.0) and (i % 4 == 0)  # less vgg training

        # fading in for networks
        tempo_fading = fade_in_weight(global_step, 0, args.tempo_fading)
        ###########################################################

        # Random from one frame
        video, frame_i = train_data.get_video_and_frame(np.random.randint(len(train_data)))
        target = torch.Tensor(video.frames[frame_i]).to(device)
        K = video.intrinsics()
        H, W = target.shape[:2]
        pose = torch.Tensor(video.c2w(frame_i)).to(device)
        time_locate = t_info[-1] * frame_i
        if hasattr(video, 'background'):
            background = torch.tensor(video.background, device=device)
        else:
            background = bkg_color

        if trainVGG:  # get a cropped img (dw,dw) to train vgg
            coords_crop, dw = vgg_sample(args.vgg_strides, n_rand, target, bkg_color, steps=i)
            coords_crop = torch.reshape(coords_crop, [-1, 2])
            ys, xs = coords_crop[:, 0], coords_crop[:, 1]  # vgg_sample using ij, convert to xy
        else:
            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                xs, ys = torch.meshgrid(
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    indexing='xy'
                )
                selected = np.random.choice(4 * dH * dW, size=[n_rand], replace=False)
                if i == start:
                    print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
            else:
                xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
                selected = np.random.choice(H * W, size=[n_rand], replace=False)
            xs = torch.flatten(xs)[selected].to(device)
            ys = torch.flatten(ys)[selected].to(device)

        rays = get_rays(K, pose, xs, ys)  # (n_rand, 3), (n_rand, 3)
        rays = rays.foreach(lambda t: t.to(device))
        target_s = target[ys.long(), xs.long()]  # (n_rand, 3)

        if background is not None and background.dim() > 2:
            background = background[ys.long(), xs.long()]

        #####  core radiance optimization loop  #####
        output = renderer.render(
            rays.origins, rays.viewdirs, chunk=args.chunk,
            ret_raw=True,
            timestep=time_locate,
            background=background)
        rgb, _, acc, extras = output.as_tuple()

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        if 'static' in extras and tempo_fading < 1.0 - 1e-8:
            img_loss = img_loss * tempo_fading + img2mse(extras['static'].rgb, target_s) * (1.0 - tempo_fading)
            # rgb = rgb * tempo_fading + extras['rgbh1'] * (1.0-tempo_fading)

        # trans = extras['raw'][...,-1]
        loss = img_loss

        writer.add_scalar(f"Loss/img_loss_d", img2mse(rgb, target_s), i)
        writer.add_scalar(f"Loss/img_loss_s", img2mse(extras['static'].rgb, target_s), i)
        writer.add_scalar(f"Loss/img_loss", img_loss, i)

        psnr = mse2psnr(img_loss.detach())
        psnr1k[i % 1000] = psnr

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

    if sys.gettrace() is not None:  # in debug mode
        pinf_train()
    else:
        try:
            pinf_train()
        except Exception as e:
            import traceback

            traceback.print_exception(e, file=sys.stdout)  # write to log
