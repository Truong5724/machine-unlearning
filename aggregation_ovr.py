import argparse
import importlib
import json
import os

import numpy as np

OVR_TASKS = [
    "gender_female",
    "gender_male",
    "age_bin0",
    "age_bin1",
    "age_bin2",
    "race_white",
    "race_black",
    "race_asian",
    "race_indian",
    "race_others",
]


def load_dataloader(dataset_path):
    with open(dataset_path, "r") as f:
        datasetfile = json.loads(f.read())
    module_name = ".".join(
        dataset_path.replace("\\", "/").split("/")[:-1]
        + [datasetfile["dataloader"]]
    )
    dl = importlib.import_module(module_name)
    return datasetfile, dl


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá UTKFace OVR (10 head binary) trên test set."
    )
    parser.add_argument("--container", default="utkface_ovr")
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ovr")
    args = parser.parse_args()

    ds, dl = load_dataloader(args.dataset)
    test_idx = np.arange(ds["nb_test"])
    _, y_dict = dl.load_ovr(test_idx, category="test")

    print("=" * 70)
    print("UTKFACE OVR EVALUATION")
    print("=" * 70)
    print(f"Container: {args.container}")
    print(f"Label    : {args.label}")
    print(f"Dataset  : {args.dataset}")
    print()

    # Load outputs (N,1) cho từng task
    preds = {}
    for shard, task in enumerate(OVR_TASKS):
        path = f"containers/{args.container}/outputs/shard-{shard}:{args.label}.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing output: {path}")
        out = np.load(path, allow_pickle=True)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        preds[task] = out[:, 0]  # (N,)

    # Tính accuracy từng head
    accs = {}
    for task in OVR_TASKS:
        y_true = np.asarray(y_dict[task], dtype=np.int64)
        y_hat = (preds[task] > 0.5).astype(np.int64)
        acc = float((y_true == y_hat).mean())
        accs[task] = acc
        print(f"{task:<15}: {acc*100:6.2f}%")

    # Group-level accuracy (gender / age / race)
    print("\nGroup-level accuracy (argmax trong mỗi group):")

    # Gender (2 head)
    g0 = preds["gender_female"]
    g1 = preds["gender_male"]
    gender_hat = (g1 > g0).astype(np.int64)  # 0=female,1=male
    y_gender = np.where(y_dict["gender_male"] == 1, 1, 0)
    gender_acc = float((gender_hat == y_gender).mean())
    print(f"Gender: {gender_acc*100:6.2f}%")

    # Age (3 bins)
    a0 = preds["age_bin0"]
    a1 = preds["age_bin1"]
    a2 = preds["age_bin2"]
    age_stack = np.stack([a0, a1, a2], axis=1)  # (N,3)
    age_hat = np.argmax(age_stack, axis=1)
    # ground truth: 0 if age_bin0==1, etc.
    y_age = np.zeros_like(age_hat)
    y_age[y_dict["age_bin1"] == 1] = 1
    y_age[y_dict["age_bin2"] == 1] = 2
    age_acc = float((age_hat == y_age).mean())
    print(f"Age bins: {age_acc*100:6.2f}%")

    # Race (5 heads)
    r_stack = np.stack(
        [
            preds["race_white"],
            preds["race_black"],
            preds["race_asian"],
            preds["race_indian"],
            preds["race_others"],
        ],
        axis=1,
    )  # (N,5)
    race_hat = np.argmax(r_stack, axis=1)  # 0..4
    # ground truth từ y_dict
    y_race = np.zeros_like(race_hat)
    for i, name in enumerate(
        ["race_white", "race_black", "race_asian", "race_indian", "race_others"]
    ):
        y_race[y_dict[name] == 1] = i
    race_acc = float((race_hat == y_race).mean())
    print(f"Race   : {race_acc*100:6.2f}%")

    print("\nMean head accuracy:", float(np.mean(list(accs.values()))) * 100, "%")
    print("=" * 70)


if __name__ == "__main__":
    main()

