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
from tqdm import tqdm

from architectures.celeba_multitask import CelebAMultiTaskModel
from sharded import getShardHash, sizeOfShard

NUM_ATTRIBUTES = 27
TASKS = [f"attr_{i}" for i in range(NUM_ATTRIBUTES)]


def load_dataset_config(datasetfile_path):
    with open(datasetfile_path, "r") as f:
        datasetfile = json.loads(f.read())
    module_name = ".".join(
        datasetfile_path.replace("\\", "/").split("/")[:-1] + [datasetfile["dataloader"]]
    )
    dataloader = importlib.import_module(module_name)
    return datasetfile, dataloader


def fetch_celeba_batch(
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
            yield dataloader.load(indices, category="train")
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
        if indices.size > 0:
            yield dataloader.load(indices, category="train")


def make_optimizer(model, name, learning_rate):
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    raise ValueError("Unsupported optimizer")


def multitask_loss(outputs, labels, loss_fns):
    """Loss cho 27 attributes"""
    total = 0.0
    for i in range(NUM_ATTRIBUTES):
        y = torch.from_numpy(labels[:, i]).to(outputs[i].device).long()
        total += loss_fns[i](outputs[i], y)
    return total

def train(args):
    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CelebAMultiTaskModel(
        input_shape=input_shape, 
        dropout_rate=args.dropout_rate,
        num_attributes=NUM_ATTRIBUTES
    ).to(device)

    shard_size = sizeOfShard(args.container, args.shard)
    if shard_size == 0:
        print(f"Shard {args.shard} is empty.")
        return

    slice_size = max(1, shard_size // args.slices)
    avg_epochs_per_slice = 2 * args.slices / (args.slices + 1) * args.epochs / args.slices

    loss_fns = [CrossEntropyLoss() for _ in range(NUM_ATTRIBUTES)]

    optimizer = make_optimizer(model, args.optimizer, args.learning_rate)

    loaded = False
    elapsed_time = 0.0

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

        train_time = 0.0

        for epoch in tqdm(range(start_epoch, slice_epochs), leave=False):
            model.train()
            running_loss = 0.0

            for images, labels in fetch_celeba_batch(
                args.container,
                args.label,
                args.shard,
                args.batch_size,
                args.dataset,
                until=until,
            ):
                x = torch.from_numpy(images).to(device)

                epoch_start = time()

                outputs = model(x)                    # dict {attr_0: ..., attr_1: ..., ...}
                loss = multitask_loss(outputs, labels, loss_fns)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_time += time() - epoch_start

                running_loss += loss.item()

            print(
                f"[Shard {args.shard}][Slice {sl}][Epoch {epoch + 1}] "
                f"loss={running_loss:.4f}"
                f"Train time={train_time:.2f}s"
            )

            if (
                args.chkpt_interval != -1
                and epoch % args.chkpt_interval == args.chkpt_interval - 1
            ):
                torch.save(
                    model.state_dict(),
                    f"containers/{args.container}/cache/{slice_hash}_{epoch}.pt",
                )

        torch.save(model.state_dict(), final_ckpt)
        with open(final_time, "w") as f:
            f.write(f"{train_time + elapsed_time}\n")

        if sl == args.slices - 1:
            shard_link = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
            if os.path.exists(shard_link) or os.path.islink(shard_link):
                os.remove(shard_link)
            os.symlink(f"{slice_hash}.pt", shard_link)

            time_link = f"containers/{args.container}/times/shard-{args.shard}:{args.label}.time"
            if os.path.exists(time_link) or os.path.islink(time_link):
                os.remove(time_link)
            os.symlink(f"{slice_hash}.time", time_link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--container", required=True)
    parser.add_argument("--dataset", default="datasets/CelebA/datasetfile")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--label", default="0")
    parser.add_argument("--slices", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--chkpt_interval", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(f"containers/{args.container}/cache", exist_ok=True)
    os.makedirs(f"containers/{args.container}/times", exist_ok=True)
    os.makedirs(f"containers/{args.container}/outputs", exist_ok=True)

    if args.train:
        train(args)
    # if args.test:
    #     test(args)   # bạn có thể thêm sau


if __name__ == "__main__":
    main()