import argparse
import importlib
import json
import os
from glob import glob
from hashlib import sha256
from time import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot, softmax
from tqdm import tqdm

from architectures.utkface_multitask import MultiTaskModel


TASK_BY_SHARD = {0: "gender", 1: "age", 2: "race"}
NUM_CLASSES = {"gender": 2, "age": 3, "race": 5}


def get_task(shard):
    if shard not in TASK_BY_SHARD:
        raise ValueError("Only 3 shards are supported for multitask mode: 0, 1, 2")
    return TASK_BY_SHARD[shard]


def get_hash(indices):
    if len(indices) == 0:
        return sha256("empty".encode()).hexdigest()
    text = ":".join(np.asarray(indices, dtype=np.int64).astype(str))
    return sha256(text.encode()).hexdigest()


def load_dataset_config(datasetfile_path):
    with open(datasetfile_path, "r") as f:
        datasetfile = json.loads(f.read())
    module_name = ".".join(
        datasetfile_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader


def load_slice_plan(container, task):
    split_path = f"containers/{container}/multitask_slices.npz"
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing {split_path}. Run utkface_multitask_partition.py first."
        )
    data = np.load(split_path, allow_pickle=True)
    return [np.asarray(x, dtype=np.int64) for x in data[task]]


def load_requests(container, label, shard):
    request_path = f"containers/{container}/requestfile:{label}.npy"
    if not os.path.exists(request_path):
        return np.array([], dtype=np.int64)
    requests = np.load(request_path, allow_pickle=True)
    if shard >= len(requests):
        return np.array([], dtype=np.int64)
    return np.asarray(requests[shard], dtype=np.int64)


def make_optimizer(model, name, learning_rate):
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    raise ValueError("Unsupported optimizer")


def iter_batches(indices, batch_size, shuffle=True):
    indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        indices = np.random.permutation(indices)
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0

    precisions = []
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precision_macro = float(np.mean(precisions)) if precisions else 0.0
    recall_macro = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0

    return {
        "acc": acc,
        "precision": precision_macro,
        "bacc": recall_macro,
        "f1": f1_macro,
    }


def make_class_weight(labels, num_classes, device):
    """
    Inverse-frequency weights, normalized so mean=1.
    Chỉ dùng cho shard race (shard 2) để bù lệch class.
    """
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0  # tránh div-by-zero
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    
    print(f"[info] class_weight = {w.round(4).tolist()}")
    return torch.tensor(w, dtype=torch.float32, device=device)


def train(args):
    task = get_task(args.shard)
    nb_classes = NUM_CLASSES[task]

    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    slice_plan = load_slice_plan(args.container, task)
    requested_indices = load_requests(args.container, args.label, args.shard)

    if requested_indices.size > 0:
        filtered_plan = []
        for slice_indices in slice_plan:
            filtered_plan.append(np.setdiff1d(slice_indices, requested_indices))
        slice_plan = filtered_plan
        slice_plan = [s for s in slice_plan if len(s) > 0]

    if np.sum([len(x) for x in slice_plan]) == 0:
        print(f"All data removed by unlearning for shard {args.shard} ({task}).")
        return

    # ── Weighted loss chỉ cho shard 2 (race) ──────────────────────────────
    if args.shard == 2 and args.use_class_weight:
        all_indices = np.concatenate(slice_plan)
        _, all_labels = dataloader.load_multitask(all_indices, category="train")
        y_all = all_labels[task].astype(np.int64)
        class_weight = make_class_weight(y_all, nb_classes, device)
        loss_fn = CrossEntropyLoss(weight=class_weight)
        print(f"[info] Using weighted CrossEntropyLoss for task={task}")
    else:
        loss_fn = CrossEntropyLoss()
    # ──────────────────────────────────────────────────────────────────────

    optimizer = make_optimizer(model, args.optimizer, args.learning_rate)

    avg_epochs_per_slice = (
        2 * len(slice_plan) / (len(slice_plan) + 1) * args.epochs / len(slice_plan)
    )

    loaded = False
    elapsed_time = 0.0
    cumulative_train_time = 0.0

    for slice_id in tqdm(range(len(slice_plan)), desc=f"Shard {args.shard}-{task}"):
        current_indices = slice_plan[slice_id]
        slice_hash = get_hash(current_indices)
        final_ckpt = f"containers/{args.container}/cache/{slice_hash}.pt"
        final_time = f"containers/{args.container}/times/{slice_hash}.time"

        if os.path.exists(final_ckpt):
            if slice_id == len(slice_plan) - 1:
                shard_link = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
                if os.path.exists(shard_link) or os.path.islink(shard_link):
                    os.remove(shard_link)
                os.symlink(f"{slice_hash}.pt", shard_link)
            continue

        start_epoch = 0
        slice_epochs = int((slice_id + 1) * avg_epochs_per_slice) - int(
            slice_id * avg_epochs_per_slice
        )

        if not loaded:
            recovery_list = glob(f"containers/{args.container}/cache/{slice_hash}_*.pt")
            if len(recovery_list) > 0:
                model.load_state_dict(torch.load(recovery_list[0], map_location=device))
                start_epoch = int(recovery_list[0].split("_")[-1].split(".")[0])
                time_path = f"containers/{args.container}/times/{slice_hash}_{start_epoch}.time"
                if os.path.exists(time_path):
                    with open(time_path, "r") as f:
                        elapsed_time = float(f.read().strip())
            elif slice_id > 0:
                prev_indices = np.concatenate(slice_plan[:slice_id]) if slice_plan[:slice_id] else np.array([])
                prev_hash = get_hash(prev_indices)
                prev_path = f"containers/{args.container}/cache/{prev_hash}.pt"
                if os.path.exists(prev_path):
                    model.load_state_dict(torch.load(prev_path, map_location=device))
            loaded = True

        for epoch in tqdm(range(start_epoch, slice_epochs), leave=False):
            model.train()
            total = 0
            correct = 0
            running_loss = 0.0
            epoch_start = time()

            for batch_ids in iter_batches(current_indices, args.batch_size, shuffle=True):
                images, labels = dataloader.load_multitask(batch_ids, category="train")
                x = torch.from_numpy(images).to(device)
                y = torch.from_numpy(labels[task]).to(device)

                logits = model.forward_task(x, task)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.shape[0]

            cumulative_train_time += time() - epoch_start
            acc = 100.0 * correct / max(total, 1)
            print(
                f"[Shard {args.shard}][Slice {slice_id}][Epoch {epoch + 1}]"
                f" loss={running_loss:.4f} acc={acc:.2f}%"
            )

            if (
                args.chkpt_interval != -1
                and epoch % args.chkpt_interval == args.chkpt_interval - 1
            ):
                torch.save(
                    model.state_dict(),
                    f"containers/{args.container}/cache/{slice_hash}_{epoch}.pt",
                )
                with open(
                    f"containers/{args.container}/times/{slice_hash}_{epoch}.time", "w"
                ) as f:
                    f.write(f"{cumulative_train_time + elapsed_time}\n")

        torch.save(model.state_dict(), final_ckpt)
        with open(final_time, "w") as f:
            f.write(f"{cumulative_train_time + elapsed_time}\n")

        if slice_id == len(slice_plan) - 1:
            shard_link = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
            if os.path.exists(shard_link) or os.path.islink(shard_link):
                os.remove(shard_link)
            os.symlink(f"{slice_hash}.pt", shard_link)

            time_link = f"containers/{args.container}/times/shard-{args.shard}:{args.label}.time"
            if os.path.exists(time_link) or os.path.islink(time_link):
                os.remove(time_link)
            os.symlink(f"{slice_hash}.time", time_link)


@torch.no_grad()
def test(args):
    task = get_task(args.shard)
    nb_classes = NUM_CLASSES[task]

    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    load_path = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    outputs = np.empty((0, nb_classes))
    test_indices = np.arange(datasetfile["nb_test"])
    _, test_labels = dataloader.load_multitask(test_indices, category="test")

    for batch_ids in iter_batches(test_indices, args.batch_size, shuffle=False):
        images, _ = dataloader.load_multitask(batch_ids, category="test")
        x = torch.from_numpy(images).to(device)
        logits = model.forward_task(x, task)

        if args.output_type == "softmax":
            preds = softmax(logits, dim=1).to("cpu").numpy()
        else:
            argmax_preds = torch.argmax(logits, dim=1)
            preds = one_hot(argmax_preds, nb_classes).to("cpu").numpy()

        outputs = np.concatenate((outputs, preds))

    os.makedirs(f"containers/{args.container}/outputs", exist_ok=True)
    np.save(
        f"containers/{args.container}/outputs/shard-{args.shard}:{args.label}.npy",
        outputs,
    )

    y_true = np.asarray(test_labels[task], dtype=np.int64)
    y_pred = np.argmax(outputs, axis=1).astype(np.int64)
    metrics = compute_multiclass_metrics(y_true, y_pred, nb_classes)

    print("=" * 70)
    print(f"Shard {args.shard} ({task}) metrics")
    print("=" * 70)
    print(f"acc     : {metrics['acc'] * 100:.2f}%")
    print(f"prec    : {metrics['precision'] * 100:.2f}%")
    print(f"bacc    : {metrics['bacc'] * 100:.2f}%")
    print(f"f1      : {metrics['f1'] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--container", required=True)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ver2")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--label", default="0")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--chkpt_interval", type=int, default=5)
    parser.add_argument("--output_type", default="argmax", choices=["argmax", "softmax"])

    # weighted loss cho race (shard 2), mặc định bật
    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        default=True,
        help="Dùng weighted CrossEntropyLoss cho shard 2 (race) để bù lệch class",
    )
    parser.add_argument(
        "--no_class_weight",
        dest="use_class_weight",
        action="store_false",
        help="Tắt weighted loss",
    )

    args = parser.parse_args()

    os.makedirs(f"containers/{args.container}/cache", exist_ok=True)
    os.makedirs(f"containers/{args.container}/times", exist_ok=True)
    os.makedirs(f"containers/{args.container}/outputs", exist_ok=True)

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == "__main__":
    main()