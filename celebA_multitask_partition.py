"""
Partition CelebA - Chọn random 30k ảnh
"""

import argparse
import importlib
import json
import os
import sys
import numpy as np


def load_dataloader(datasetfile_path):
    with open(datasetfile_path, "r") as f:
        datasetfile = json.load(f)

    dataset_dir = os.path.dirname(os.path.abspath(datasetfile_path))
    
    # Hardcode tên dataloader của bạn
    dataloader_name = "dataloader_multitask"
    
    if dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)

    try:
        dataloader = importlib.import_module(dataloader_name)
        return datasetfile, dataloader
    except:
        py_path = os.path.join(dataset_dir, f"{dataloader_name}.py")
        if not os.path.exists(py_path):
            raise FileNotFoundError(f"Không tìm thấy {py_path}")
        
        spec = importlib.util.spec_from_file_location(dataloader_name, py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return datasetfile, module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="celeba")
    parser.add_argument("--dataset", default="datasets/celebA/datasetfile_celeba")
    parser.add_argument("--slices_per_shard", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=30000)
    args = parser.parse_args()

    datasetfile, dataloader = load_dataloader(args.dataset)
    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    nb_train = datasetfile.get("nb_train", 0)
    print(f"Total train samples: {nb_train}")

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(nb_train, dtype=np.int64)

    selected = rng.choice(all_indices, size=min(args.samples, nb_train), replace=False)
    selected = np.sort(selected)

    print(f"Selected {len(selected)} samples")

    perm = rng.permutation(selected)
    slices = np.array_split(perm, args.slices_per_shard)

    splitfile = np.array([np.array(s, dtype=np.int64) for s in slices], dtype=object)
    np.save(f"{container_dir}/splitfile.npy", splitfile)

    print(f"✅ Created splitfile with {args.slices_per_shard} slices")
    print("Ready for training!")


if __name__ == "__main__":
    main()