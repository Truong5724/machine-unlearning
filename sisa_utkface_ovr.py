import argparse
import glob
import json
import os
from hashlib import sha256
from time import time

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm

from architectures.utkface_ovr import OVRModel, OVR_TASKS


TASK_BY_SHARD = {i: name for i, name in enumerate(OVR_TASKS)}  # 0..9


class BinaryFocalLossWithLogits(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * focal_factor * bce
        return loss.mean()


def get_task(shard):
    if shard not in TASK_BY_SHARD:
        raise ValueError(f"Shard {shard} không hợp lệ cho OVR (0..{len(OVR_TASKS)-1})")
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
        datasetfile_path.replace("\\", "/").split("/")[:-1]
        + [datasetfile["dataloader"]]
    )
    dataloader = __import__(module_name, fromlist=["dummy"])
    return datasetfile, dataloader


def load_slice_plan(container, task):
    path = f"containers/{container}/ovr_slices.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run utkface_ovr_partition.py first."
        )
    data = np.load(path, allow_pickle=True)
    return [np.asarray(x, dtype=np.int64) for x in data[task]]


def load_requests(container, label, shard, num_shards):
    path = f"containers/{container}/requestfile:{label}.npy"
    if not os.path.exists(path):
        return np.array([], dtype=np.int64)
    req = np.load(path, allow_pickle=True)
    req = np.asarray(req, dtype=object).ravel()
    if len(req) <= shard:
        return np.array([], dtype=np.int64)
    return np.asarray(req[shard], dtype=np.int64)


def make_optimizer(model, name, lr):
    if name == "adam":
        return Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    raise ValueError("Unsupported optimizer")


def iter_batches(indices, batch_size, shuffle=True):
    indices = np.asarray(indices, dtype=np.int64)
    if shuffle:
        indices = np.random.permutation(indices)
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def parse_task_set(text):
    if text is None or text.strip() == "":
        return set()
    return {x.strip() for x in text.split(",") if x.strip()}


def train(args):
    task = get_task(args.shard)

    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    # Tính class imbalance để đặt pos_weight cho BCE (one-vs-rest)
    nb_train = datasetfile["nb_train"]
    all_train_idx = np.arange(nb_train, dtype=np.int64)
    all_labels = dataloader.load_ovr_labels(all_train_idx, category="train")
    y_all = np.asarray(all_labels[task], dtype=np.int64)
    pos = y_all.sum()
    neg = len(y_all) - pos
    severe_task_set = parse_task_set(args.focal_tasks)

    if pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(neg) / float(pos)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OVRModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    slice_plan = load_slice_plan(args.container, task)
    requested = load_requests(
        args.container, args.label, args.shard, len(OVR_TASKS)
    )

    if requested.size > 0:
        filtered = []
        for s in slice_plan:
            filtered.append(np.setdiff1d(s, requested))
        slice_plan = [s for s in filtered if len(s) > 0]

    if sum(len(s) for s in slice_plan) == 0:
        print(f"All data removed by unlearning for shard {args.shard} ({task})")
        return

    optimizer = make_optimizer(model, args.optimizer, args.learning_rate)

    use_focal = args.loss_mode == "focal" or (
        args.loss_mode == "auto" and task in severe_task_set
    )
    pos_weight_tensor = torch.tensor(pos_weight_value, device=device)

    if use_focal:
        if args.focal_alpha < 0:
            # Auto alpha: tăng trọng số positive khi lớp positive hiếm.
            alpha = float(neg) / float(max(pos + neg, 1))
            alpha = min(max(alpha, 0.05), 0.95)
        else:
            alpha = args.focal_alpha

        loss_fn = BinaryFocalLossWithLogits(
            gamma=args.focal_gamma,
            alpha=alpha,
            pos_weight=pos_weight_tensor,
        )
        print(
            f"[Shard {args.shard}][{task}] loss=focal gamma={args.focal_gamma} "
            f"alpha={alpha:.4f} pos_weight={pos_weight_value:.4f}"
        )
    else:
        loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        print(
            f"[Shard {args.shard}][{task}] loss=bce pos_weight={pos_weight_value:.4f}"
        )

    avg_epochs_per_slice = (
        2 * len(slice_plan) / (len(slice_plan) + 1) * args.epochs / len(slice_plan)
    )

    loaded = False
    elapsed_time = 0.0
    cumulative_train_time = 0.0

    for slice_id in tqdm(range(len(slice_plan)), desc=f"Shard {args.shard}-{task}"):
        current_indices = np.concatenate(slice_plan[: slice_id + 1])
        slice_hash = get_hash(current_indices)
        ckpt_final = f"containers/{args.container}/cache/{slice_hash}.pt"
        time_final = f"containers/{args.container}/times/{slice_hash}.time"

        if os.path.exists(ckpt_final):
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
            # Recovery checkpoint
            rec_list = glob.glob(
                f"containers/{args.container}/cache/{slice_hash}_*.pt"
            )
            if rec_list:
                model.load_state_dict(
                    torch.load(rec_list[0], map_location=device)
                )
                start_epoch = int(
                    rec_list[0].split("_")[-1].split(".")[0]
                )
                tpath = (
                    f"containers/{args.container}/times/"
                    f"{slice_hash}_{start_epoch}.time"
                )
                if os.path.exists(tpath):
                    with open(tpath, "r") as f:
                        elapsed_time = float(f.read().strip())
            elif slice_id > 0:
                prev_indices = np.concatenate(slice_plan[:slice_id])
                prev_hash = get_hash(prev_indices)
                prev_ckpt = (
                    f"containers/{args.container}/cache/{prev_hash}.pt"
                )
                if os.path.exists(prev_ckpt):
                    model.load_state_dict(
                        torch.load(prev_ckpt, map_location=device)
                    )
            loaded = True

        for epoch in tqdm(range(start_epoch, slice_epochs), leave=False):
            model.train()
            total = 0
            correct = 0
            running_loss = 0.0
            epoch_start = time()

            for batch_ids in iter_batches(
                current_indices, args.batch_size, shuffle=True
            ):
                images, y_dict = dataloader.load_ovr(batch_ids, category="train")
                x = torch.from_numpy(images).to(device)
                y = torch.from_numpy(y_dict[task]).float().to(device)  # (B,)

                logits = model.forward_task(x, task)  # (B,)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                correct += (preds == y.long()).sum().item()
                total += y.size(0)

            cumulative_train_time += time() - epoch_start
            acc = 100.0 * correct / max(total, 1)
            print(
                f"[Shard {args.shard}][Slice {slice_id}][Epoch {epoch+1}] "
                f"loss={running_loss:.4f} acc={acc:.2f}%"
            )

            if (
                args.chkpt_interval != -1
                and epoch % args.chkpt_interval == args.chkpt_interval - 1
            ):
                tmp_ckpt = (
                    f"containers/{args.container}/cache/{slice_hash}_{epoch}.pt"
                )
                tmp_time = (
                    f"containers/{args.container}/times/{slice_hash}_{epoch}.time"
                )
                torch.save(model.state_dict(), tmp_ckpt)
                with open(tmp_time, "w") as f:
                    f.write(f"{cumulative_train_time + elapsed_time}\n")

        torch.save(model.state_dict(), ckpt_final)
        with open(time_final, "w") as f:
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

    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OVRModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    ckpt = f"containers/{args.container}/cache/shard-{args.shard}:{args.label}.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    key = f"nb_{args.eval_split}"
    if key not in datasetfile:
        raise KeyError(
            f"{key} không có trong datasetfile. "
            "Hãy chạy lại prepare_data_ovr.py để tạo split tương ứng."
        )
    nb_eval = datasetfile[key]
    eval_indices = np.arange(nb_eval, dtype=np.int64)

    outputs = []
    for batch_ids in iter_batches(eval_indices, args.batch_size, shuffle=False):
        images, _ = dataloader.load_ovr(batch_ids, category=args.eval_split)
        x = torch.from_numpy(images).to(device)
        logits = model.forward_task(x, task)
        probs = torch.sigmoid(logits).to("cpu").numpy()  # (B,)
        outputs.append(probs.reshape(-1, 1))

    if outputs:
        out_mat = np.concatenate(outputs, axis=0)  # (N,1)
    else:
        out_mat = np.empty((0, 1), dtype=np.float32)

    if args.eval_split == "test":
        out_path = f"containers/{args.container}/outputs/shard-{args.shard}:{args.label}.npy"
    else:
        out_path = (
            f"containers/{args.container}/outputs/"
            f"shard-{args.shard}:{args.label}:{args.eval_split}.npy"
        )
    np.save(out_path, out_mat)
    print(f"Saved outputs: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--dataset", default="datasets/UTKFace/datasetfile_ovr"
    )
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--label", default="0")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--chkpt_interval", type=int, default=5)
    parser.add_argument(
        "--loss_mode",
        default="auto",
        choices=["auto", "bce", "focal"],
        help="Chọn loss: auto (focal cho head severe), bce, focal.",
    )
    parser.add_argument(
        "--focal_tasks",
        default="race_others,age_bin2",
        help="Danh sách head dùng focal khi loss_mode=auto (csv).",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma cho focal loss.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=-1.0,
        help="Alpha cho focal loss; <0 để tự động theo imbalance.",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        choices=["val", "test"],
        help="Split dùng khi chạy --test.",
    )

    args = parser.parse_args()

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == "__main__":
    main()

