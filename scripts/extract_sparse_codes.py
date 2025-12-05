import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.encoder import Encoder
from src.models.bottleneck import DictionaryLearning
from torchvision import datasets, transforms

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return {'image': original_tuple[0], 'path': path}

def extract_and_save_sparse_codes(encoder_ckpt, bottleneck_ckpt, data_dir, output_dir,
                                  batch_size=32, num_workers=4, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    encoder = Encoder(in_channels=3, num_hiddens=128, num_residual_blocks=2, num_residual_hiddens=32)
    bottleneck = DictionaryLearning()  # Fill in with your config if needed
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    bottleneck.load_state_dict(torch.load(bottleneck_ckpt, map_location=device))
    encoder.eval()
    bottleneck.eval()
    encoder.to(device)
    bottleneck.to(device)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            images = batch['image'].to(device)
            z_e = encoder(images)
            sparse_codes = bottleneck._sparse_encode(z_e, bottleneck.dictionary)  # [num_atoms, num_patches]
            for j in range(images.size(0)):
                codes = sparse_codes[:, j].cpu()  # Adjust indexing if needed
                img_name = os.path.splitext(os.path.basename(batch['path'][j]))[0]
                torch.save(codes, os.path.join(output_dir, f"{img_name}.pt"))
    print(f"Saved sparse codes to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and save sparse codes from LASER model")
    parser.add_argument('--encoder_ckpt', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--bottleneck_ckpt', type=str, required=True, help='Path to bottleneck checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to image dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save sparse code .pt files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    extract_and_save_sparse_codes(args.encoder_ckpt, args.bottleneck_ckpt, args.data_dir, args.output_dir,
                                 args.batch_size, args.num_workers, args.device)
