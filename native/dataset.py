import torch
import torchvision.io as io
import numpy as np
import tqdm
import typing
import types
import os
import yaml
import trimesh
import random


class PINeuFlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 dataset_type: typing.Literal['train', 'val', 'test'],
                 downscale: int,
                 use_fp16: bool,
                 device: torch.device,
                 ):
        # self.images
        self.images = PINeuFlowDataset._load_images(dataset_path, dataset_type, downscale)  # [T, V, H, W, C]

        # self.poses and self.focals
        self.poses, self.focals, self.extra_params = PINeuFlowDataset._load_camera_calibrations(dataset_path, dataset_type, downscale)

        # self.times
        self.times = torch.linspace(0, 1, steps=self.images.shape[0], dtype=torch.float16 if use_fp16 else torch.float32).view(-1, 1)

        # self.states
        self.states = types.SimpleNamespace()
        self.states.dataset_type = dataset_type

        # set dtype and move to device
        self.to(device=device, dtype=torch.float16 if use_fp16 else torch.float32)

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=1,
            shuffle=self.states.dataset_type == 'train',
            num_workers=0,
            collate_fn=self.collate
        )

    def collate(self, batch: list):
        images = torch.stack([single['image'] for single in batch])  # [B, H, W, C]
        poses = torch.stack([single['pose'] for single in batch])  # [B, 4, 4]
        focals = torch.stack([single['focal'] for single in batch])  # [B]
        times = torch.stack([single['time'] for single in batch])  # [B, 1]
        video_indices = torch.tensor([single['video_index'] for single in batch])  # [B]
        frame_indices = torch.tensor([single['frame_index'] for single in batch])  # [B]

        return {
            'pixels': images,  # [B, N, C]
            'poses': poses,  # [B, 4, 4]
            'focals': focals,  # [B]
            'times': times,  # [B, 1]
            'idx_v': video_indices,  # [B]
            'idx_f': frame_indices,  # [B]
        }

    def __getitem__(self, index):
        """
        :param index: frame index
        :return:
        """
        if self.states.dataset_type == 'train':
            time_shift = random.uniform(-0.5, 0.5)
        else:
            time_shift = 0
        video_index = random.randint(0, self.poses.shape[0] - 1)

        if index == 0 and time_shift <= 0:
            target_image = self.images[index, video_index]
            target_time = self.times[index]
        elif index == self.images.shape[0] - 1 and time_shift >= 0:
            target_image = self.images[index, video_index]
            target_time = self.times[index]
        else:
            if time_shift >= 0:
                target_image = (1 - time_shift) * self.images[index, video_index] + time_shift * self.images[index + 1, video_index]
                target_time = (1 - time_shift) * self.times[index] + time_shift * self.times[index + 1]
            else:
                target_image = (1 + time_shift) * self.images[index, video_index] + (-time_shift) * self.images[index - 1, video_index]
                target_time = (1 + time_shift) * self.times[index] + (-time_shift) * self.times[index - 1]

        return {
            'image': target_image,
            'pose': self.poses[video_index],
            'time': target_time,
            'video_index': video_index,
            'frame_index': index,
            'focal': self.focals[video_index],
        }

    def to(self, device: torch.device, dtype: torch.dtype):
        self.images = self.images.to(dtype=dtype).to(device=device)
        self.poses = self.poses.to(dtype=dtype).to(device=device)
        self.focals = self.focals.to(dtype=dtype).to(device=device)
        self.times = self.times.to(dtype=dtype).to(device=device)

    @property
    def device(self):
        return self.images.device

    @property
    def dtype(self):
        return self.images.dtype

    @property
    def num_frames(self):
        return self.images.shape[0]

    @property
    def num_videos(self):
        return self.images.shape[1]

    @property
    def num_colors(self):
        return self.images.shape[4]

    @property
    def height(self):
        assert self.images.shape[2] == self.extra_params.height
        return self.images.shape[2]

    @property
    def width(self):
        assert self.images.shape[3] == self.extra_params.width
        return self.images.shape[3]

    @property
    def dt(self):
        return 1.0 / self.num_frames

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_images(dataset_path, dataset_type, downscale):
        """
        Load images from the dataset path and return them as a tensor.
        Args:
            dataset_path (str): Path to the dataset.
            dataset_type (str): Type of the dataset ('train', 'val', 'test').
            downscale (int): Downscale factor for the images.
        Returns:
            torch.Tensor: Tensor containing the loaded images. [T, V, H, W, C]
        """
        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)
            videos_info = scene_info['training_videos'] if dataset_type == 'train' else scene_info['validation_videos']
            frames = []
            for path in tqdm.tqdm([os.path.normpath(os.path.join(dataset_path, video_path)) for video_path in videos_info], desc=f'[Loading Images ({dataset_type})...]'):
                try:
                    v_frames, a_frames, _info = io.read_video(path, pts_unit='sec')
                    frames.append(v_frames / 255.0)
                except Exception as e:
                    raise FileNotFoundError(f'Could not load video {path}: {e}')
            frames = torch.stack(frames)
            V, T, H, W, C = frames.shape
            H_downscale, W_downscale = int(H // downscale), int(W // downscale)
            frames_downscale = torch.nn.functional.interpolate(frames.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W), size=(H_downscale, W_downscale), mode='bilinear', align_corners=False).reshape(V, T, C, H_downscale, W_downscale).permute(1, 0, 3, 4, 2)
            return frames_downscale

    @staticmethod
    def _load_camera_calibrations(dataset_path, dataset_type, downscale):
        """
        Load camera calibrations from the dataset path and return them as tensors.
        Args:
            dataset_path (str): Path to the dataset.
            dataset_type (str): Type of the dataset ('train', 'val', 'test').
            downscale (int): Downscale factor for the camera parameters.
        Returns:
            tuple: A tuple containing:
                - poses (torch.Tensor): Camera poses. [V, 4, 4]
                - focals (torch.Tensor): Focal lengths. [V]
                - widths (int): Width of the images.
                - heights (int): Height of the images.
                - extra_params (types.SimpleNamespace): Additional parameters including near and far planes, voxel transform, scale, etc.
        """
        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)
            cameras_info = scene_info['training_camera_calibrations'] if dataset_type == 'train' else scene_info['validation_camera_calibrations']
            poses, focals, widths, heights, nears, fars = [], [], [], [], [], []
            for path in tqdm.tqdm([os.path.normpath(os.path.join(dataset_path, camera_path)) for camera_path in cameras_info], desc=f'[Loading Camera ({dataset_type})...]'):
                try:
                    camera_info = np.load(path)
                    poses.append(torch.tensor(camera_info["cam_transform"]))
                    focals.append(float(camera_info["focal"]) * float(camera_info["width"]) / float(camera_info["aperture"]))
                    widths.append(int(camera_info["width"]))
                    heights.append(int(camera_info["height"]))
                    nears.append(float(camera_info["near"]))
                    fars.append(float(camera_info["far"]))
                except Exception as e:
                    raise FileNotFoundError(f'Could not load camera {path}: {e}')
            assert len(set(widths)) == 1, f"Error: Inconsistent widths found: {widths}. All cameras must have the same resolution."
            assert len(set(heights)) == 1, f"Error: Inconsistent heights found: {heights}. All cameras must have the same resolution."
            assert len(set(nears)) == 1, f"Error: Inconsistent near planes found: {nears}. All cameras must have the same near plane."
            assert len(set(fars)) == 1, f"Error: Inconsistent far planes found: {fars}. All cameras must have the same far plane."

            poses = torch.stack(poses)
            focals = torch.tensor(focals)

            focals = focals / downscale

            extra_params = types.SimpleNamespace()
            extra_params.width = set(widths).pop() // downscale
            extra_params.height = set(heights).pop() // downscale
            extra_params.near = set(nears).pop()
            extra_params.far = set(fars).pop()
            extra_params.voxel_transform = torch.tensor(scene_info['voxel_transform'])
            extra_params.voxel_scale = torch.tensor(scene_info['voxel_scale'])
            extra_params.s_min = torch.tensor(scene_info['s_min'])
            extra_params.s_max = torch.tensor(scene_info['s_max'])
            extra_params.s_w2s = torch.inverse(extra_params.voxel_transform).expand([4, 4])
            extra_params.s2w = torch.inverse(extra_params.s_w2s)
            extra_params.s_scale = extra_params.voxel_scale.expand([3])

            return poses, focals, extra_params


class PINeuFlowDatasetValidation:
    def __init__(self, dataset: PINeuFlowDataset):
        self.dataset = dataset

    def t_dataloader(self):
        dataloader = self.dataset.dataloader()
        for i, data in enumerate(dataloader):
            data: dict
            print(f"[{i + 1}/{len(dataloader)}] Video Index: {data['idx_v'][0]}, Frame Index: {data['idx_f'][0]}")

    def show_poses(self, size=0.1):
        poses = self.dataset.poses.detach().cpu().numpy()
        assert poses.shape == (self.dataset.num_videos, 4, 4), "Poses should be of shape (V, 4, 4)"

        axes = trimesh.creation.axis(axis_length=4)
        box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
        box.colors = np.array([[128, 128, 128]] * len(box.entities))
        objects = [axes, box]

        for pose in poses:
            # a camera is visualized with 8 line segments.
            pos = pose[:3, 3]
            a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
            b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
            c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
            d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

            dir = (a + b + c + d) / 4 - pos
            dir = dir / (np.linalg.norm(dir) + 1e-8)
            o = pos + dir * 3

            segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
            segs = trimesh.load_path(segs)
            objects.append(segs)

        trimesh.Scene(objects).show()
