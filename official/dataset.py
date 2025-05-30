import os
import sys
import imageio
import cv2
import json
from render import *


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
