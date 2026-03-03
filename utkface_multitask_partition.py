import argparse
import importlib
import json
import os
import numpy as np


def build_age_slices(ages):
    # 5 bins đều trên [0..116] giống dataloader_ver2.AGE_BINS
    edges = np.array([0, 24, 48, 72, 96, 117], dtype=np.int64)
    ages = np.asarray(ages, dtype=np.int64)
    ages = np.clip(ages, edges[0], edges[-1] - 1)
    bins = np.digitize(ages, edges[1:-1], right=False).astype(np.int64)
    return [np.where(bins == idx)[0] for idx in range(5)]


def build_race5_slices(races):
    races = np.asarray(races, dtype=np.int64)
    return [np.where(races == idx)[0] for idx in range(5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True, help="Container name")
    parser.add_argument(
        "--dataset",
        default="datasets/UTKFace/datasetfile_ver2",
        help="Path to UTKFace datasetfile",
    )
    parser.add_argument("--label", default="0", help="Request label to initialize")
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        datasetfile = json.loads(f.read())

    module_name = ".".join(
        args.dataset.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)

    all_indices = np.arange(datasetfile["nb_train"])
    attrs = dataloader.load_attributes(all_indices, category="train")

    gender_slices = [
        np.where(attrs["gender"] == 0)[0],
        np.where(attrs["gender"] == 1)[0],
    ]
    age_slices = build_age_slices(attrs["age"])
    race_slices = build_race5_slices(attrs["race"])

    splitfile = np.array([all_indices, all_indices, all_indices], dtype=object)

    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    np.save(f"{container_dir}/splitfile.npy", splitfile)
    requests = np.array([np.array([], dtype=np.int64) for _ in range(3)], dtype=object)
    np.save(f"{container_dir}/requestfile:{args.label}.npy", requests)

    np.savez(
        f"{container_dir}/multitask_slices.npz",
        gender=np.array(gender_slices, dtype=object),
        age=np.array(age_slices, dtype=object),
        race=np.array(race_slices, dtype=object),
    )

    meta = {
        "tasks": ["gender", "age", "race"],
        "task_to_shard": {"gender": 0, "age": 1, "race": 2},
        "slices_per_shard": {"0": 2, "1": 5, "2": 5},
        "age_edges": [0, 24, 48, 72, 96, 116],
        "race5_mapping": {
            "0": "White",
            "1": "Black",
            "2": "Asian",
            "3": "Indian",
            "4": "Others",
        },
    }

    with open(f"{container_dir}/multitask_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Created UTKFace multitask partitions:")
    print(f"  shard 0 (gender): {[len(x) for x in gender_slices]}")
    print(f"  shard 1 (age):    {[len(x) for x in age_slices]}")
    print(f"  shard 2 (race):   {[len(x) for x in race_slices]}")


if __name__ == "__main__":
    main()
