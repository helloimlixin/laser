import os
import torch
from tqdm import tqdm

def kmeans_torch(data, num_clusters, num_iters=100):
    N, D = data.shape
    device = data.device
    indices = torch.randperm(N, device=device)[:num_clusters]
    centers = data[indices].clone()
    for _ in range(num_iters):
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)
        for k in range(num_clusters):
            assigned = data[labels == k]
            if assigned.shape[0] > 0:
                centers[k] = assigned.mean(dim=0)
    return labels, centers

def main(sparse_codes_dir, output_dir, num_clusters=2048, num_iters=100):
    os.makedirs(output_dir, exist_ok=True)
    all_codes = []
    file_list = sorted([f for f in os.listdir(sparse_codes_dir) if f.endswith('.pt')])
    print(f"Loading {len(file_list)} sparse code files...")
    for fname in tqdm(file_list):
        codes = torch.load(os.path.join(sparse_codes_dir, fname))  # shape: [num_patches, sparsity_level]
        all_codes.append(codes)
    all_codes = torch.cat(all_codes, dim=0)  # [total_patches, sparsity_level]
    print(f"Running k-means on {all_codes.shape[0]} codes...")
    labels, centers = kmeans_torch(all_codes, num_clusters, num_iters)
    # Save cluster centers
    torch.save(centers, os.path.join(output_dir, 'kmeans_centers.pt'))
    # Assign tokens for each file
    idx = 0
    for fname in tqdm(file_list):
        codes = torch.load(os.path.join(sparse_codes_dir, fname))
        n = codes.shape[0]
        file_labels = labels[idx:idx+n]
        idx += n
        out_path = os.path.join(output_dir, fname.replace('.pt', '_tokens.pt'))
        torch.save(file_labels, out_path)
    print(f"Saved token files to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="K-means quantization for sparse codes")
    parser.add_argument('--sparse_codes_dir', type=str, required=True, help='Directory with .pt files of sparse codes')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save token .pt files')
    parser.add_argument('--num_clusters', type=int, default=2048, help='Number of clusters (vocab size)')
    parser.add_argument('--num_iters', type=int, default=100, help='K-means iterations')
    args = parser.parse_args()
    main(args.sparse_codes_dir, args.output_dir, args.num_clusters, args.num_iters)
