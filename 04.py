from native import PINeuFlowDataset, PINeuFlowDatasetValidation
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    dataset = PINeuFlowDataset(dataset_path='data/sphere', dataset_type='train', downscale=1, use_fp16=False, device=device)
    viewer = PINeuFlowDatasetValidation(dataset=dataset)
    viewer.t_dataloader()
