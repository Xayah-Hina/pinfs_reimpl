from native import PINeuFlowDataset, PINeuFlowDatasetValidation
from official import PINFDataset, PINFFrameData
import torch

from matplotlib import pyplot as plt

if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = PINeuFlowDataset(dataset_path='data/sphere', dataset_type='train', downscale=1, use_fp16=False, device=device)
    for pose in dataset.poses:
        print(f"Pose: {pose}")

    pinf_data = PINFFrameData('./data/pinf/Sphere', half_res='normal', normalize_time=True)

    v1 = pinf_data.videos['train'][0]
    print(v1.frames.shape)

    plt.imshow(v1.frames[0])
    # viewer = PINeuFlowDatasetValidation(dataset=dataset)
    # viewer.t_dataloader()
