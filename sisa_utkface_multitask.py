import argparse
import importlib
import json
import os
from glob import glob
from time import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot, softmax
from sharded import getShardHash, sizeOfShard
from tqdm import tqdm

from architectures.utkface_multitask import MultiTaskModel


TASKS = ("gender", "age", "race")
NUM_CLASSES = {"gender": 2, "age": 3, "race": 5}


def load_dataset_config(datasetfile_path):
    with open(datasetfile_path, "r") as f:
        datasetfile = json.loads(f.read())
    module_name = ".".join(
        datasetfile_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader


def fetch_multitask_shard_batch(
    container, label, shard, batch_size, dataset, offset=0, until=None
):
    shards = np.load(f"containers/{container}/splitfile.npy", allow_pickle=True)
    requests = np.load(f"containers/{container}/requestfile:{label}.npy", allow_pickle=True)

    datasetfile, dataloader = load_dataset_config(dataset)
    if until is None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = np.setdiff1d(shards[shard][limit - batch_size : limit], requests[shard])
        if indices.size > 0:
            yield dataloader.load_multitask(indices, category="train")
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
        if indices.size > 0:
            yield dataloader.load_multitask(indices, category="train")


def make_optimizer(model, name, learning_rate):
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    raise ValueError("Unsupported optimizer")


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0

    precisions, recalls, f1s = [], [], []
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

    return {
        "acc": acc,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "bacc": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def make_class_weight(labels, num_classes, device):
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def multitask_loss(outputs, labels, loss_fns):
    total = 0.0
    for task in TASKS:
        y = torch.from_numpy(labels[task]).to(outputs[task].device)
        total = total + loss_fns[task](outputs[task], y)
    return total


def train(args):
    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    shard_size = sizeOfShard(args.container, args.shard)
    if shard_size == 0:
        print(f"Shard {args.shard} is empty.")
        return

    slice_size = max(1, shard_size // args.slices)
    avg_epochs_per_slice = 2 * args.slices / (args.slices + 1) * args.epochs / args.slices

    loss_fns = {task: CrossEntropyLoss() for task in TASKS}
    if args.use_class_weight:
        shards = np.load(f"containers/{args.container}/splitfile.npy", allow_pickle=True)
        requests = np.load(
            f"containers/{args.container}/requestfile:{args.label}.npy", allow_pickle=True
        )
        retained = np.setdiff1d(shards[args.shard], requests[args.shard])
        if retained.size > 0:
            _, all_labels = dataloader.load_multitask(retained, category="train")
            race_weight = make_class_weight(all_labels["race"], NUM_CLASSES["race"], device)
            loss_fns["race"] = CrossEntropyLoss(weight=race_weight)
            print(f"[info] Using weighted CrossEntropyLoss for race")

    optimizer = make_optimizer(model, args.optimizer, args.learning_rate)

    loaded = False
    elapsed_time = 0.0
    cumulative_train_time = 0.0

    for sl in tqdm(range(args.slices), desc=f"Shard {args.shard}"):
        slice_hash = getShardHash(
            args.container, args.label, args.shard, until=(sl + 1) * slice_size
        )
        final_ckpt = f"containers/{args.container}/cache/{slice_hash}.pt"
        final_time = f"containers/{args.container}/times/{slice_hash}.time"

        if os.path.exists(final_ckpt):
            if sl == args.slices - 1:
                shard_link = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
                if os.path.exists(shard_link) or os.path.islink(shard_link):
                    os.remove(shard_link)
                os.symlink(f"{slice_hash}.pt", shard_link)
            continue

        start_epoch = 0
        slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)

        if not loaded:
            recovery_list = glob(f"containers/{args.container}/cache/{slice_hash}_*.pt")
            if recovery_list:
                model.load_state_dict(torch.load(recovery_list[0], map_location=device))
                start_epoch = int(recovery_list[0].split("_")[-1].split(".")[0])
                time_path = f"containers/{args.container}/times/{slice_hash}_{start_epoch}.time"
                if os.path.exists(time_path):
                    with open(time_path, "r") as f:
                        elapsed_time = float(f.read().strip())
            elif sl > 0:
                prev_hash = getShardHash(
                    args.container, args.label, args.shard, until=sl * slice_size
                )
                prev_path = f"containers/{args.container}/cache/{prev_hash}.pt"
                if os.path.exists(prev_path):
                    model.load_state_dict(torch.load(prev_path, map_location=device))
            loaded = True

        until = (sl + 1) * slice_size if sl < args.slices - 1 else None

        for epoch in tqdm(range(start_epoch, slice_epochs), leave=False):
            model.train()
            total = 0
            task_correct = {task: 0 for task in TASKS}
            running_loss = 0.0
            epoch_start = time()

            for images, labels in fetch_multitask_shard_batch(
                args.container,
                args.label,
                args.shard,
                args.batch_size,
                args.dataset,
                until=until,
            ):
                x = torch.from_numpy(images).to(device)
                outputs = model(x)
                loss = multitask_loss(outputs, labels, loss_fns)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_size = x.shape[0]
                total += batch_size
                for task in TASKS:
                    y = torch.from_numpy(labels[task]).to(device)
                    preds = torch.argmax(outputs[task], dim=1)
                    task_correct[task] += (preds == y).sum().item()

            cumulative_train_time += time() - epoch_start
            accs = {task: 100.0 * task_correct[task] / max(total, 1) for task in TASKS}
            mean_acc = np.mean(list(accs.values()))
            print(
                f"[Shard {args.shard}][Slice {sl}][Epoch {epoch + 1}] "
                f"loss={running_loss:.4f} "
                f"gender={accs['gender']:.1f}% age={accs['age']:.1f}% "
                f"race={accs['race']:.1f}% mean={mean_acc:.1f}%"
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

        if sl == args.slices - 1:
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
    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    load_path = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    test_indices = np.arange(datasetfile["nb_test"])
    _, test_labels = dataloader.load_multitask(test_indices, category="test")

    outputs_by_task = {task: np.empty((0, NUM_CLASSES[task])) for task in TASKS}

    for start in range(0, len(test_indices), args.batch_size):
        batch_ids = test_indices[start : start + args.batch_size]
        images, _ = dataloader.load_multitask(batch_ids, category="test")
        x = torch.from_numpy(images).to(device)
        logits = model(x)

        for task in TASKS:
            if args.output_type == "softmax":
                preds = softmax(logits[task], dim=1).to("cpu").numpy()
            else:
                argmax_preds = torch.argmax(logits[task], dim=1)
                preds = one_hot(argmax_preds, NUM_CLASSES[task]).to("cpu").numpy()
            outputs_by_task[task] = np.concatenate((outputs_by_task[task], preds))

    os.makedirs(f"containers/{args.container}/outputs", exist_ok=True)
    for task in TASKS:
        np.save(
            f"containers/{args.container}/outputs/shard-{args.shard}:{args.label}-{task}.npy",
            outputs_by_task[task],
        )

    print("=" * 70)
    print(f"Shard {args.shard} test metrics (all 3 tasks)")
    print("=" * 70)
    for task in TASKS:
        y_true = np.asarray(test_labels[task], dtype=np.int64)
        y_pred = np.argmax(outputs_by_task[task], axis=1).astype(np.int64)
        metrics = compute_multiclass_metrics(y_true, y_pred, NUM_CLASSES[task])
        print(
            f"{task:6s}: acc={metrics['acc'] * 100:.2f}% "
            f"prec={metrics['precision'] * 100:.2f}% "
            f"bacc={metrics['bacc'] * 100:.2f}% "
            f"f1={metrics['f1'] * 100:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--container", required=True)
    parser.add_argument("--dataset", default="datasets/UTKFace/datasetfile_ver2")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--label", default="0")
    parser.add_argument("--slices", type=int, default=1, help="SISA slices per data shard")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--chkpt_interval", type=int, default=5)
    parser.add_argument("--output_type", default="argmax", choices=["argmax", "softmax"])

    parser.add_argument("--use_class_weight", action="store_true", default=True)
    parser.add_argument("--no_class_weight", dest="use_class_weight", action="store_false")

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
