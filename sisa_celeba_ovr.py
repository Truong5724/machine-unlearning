import argparse
import glob
import importlib
import importlib.util
import json
import os
import shutil
import copy
from hashlib import sha256
from time import time

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm

from architectures.celeba_ovr import OVRModel, OVR_TASKS


TASK_BY_SHARD = {i: name for i, name in enumerate(OVR_TASKS)}  # 0..26


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

    datasetfile_abs = os.path.abspath(datasetfile_path)
    dataset_dir = os.path.dirname(datasetfile_abs)
    dataloader_name = datasetfile["dataloader"]

    module_name = ".".join(
        datasetfile_path.replace("\\", "/").split("/")[:-1]
        + [dataloader_name]
    )

    try:
        dataloader = importlib.import_module(module_name)
    except Exception:
        py_path = os.path.join(dataset_dir, f"{dataloader_name}.py")
        if not os.path.exists(py_path):
            raise
        spec = importlib.util.spec_from_file_location(dataloader_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load dataloader module from {py_path}")
        dataloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataloader)

    return datasetfile, dataloader


def load_slice_plan(container, task):
    path = f"containers/{container}/ovr_slices.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run celeba_ovr_partition.py first."
        )
    data = np.load(path, allow_pickle=True)
    return [np.asarray(x, dtype=np.int64) for x in data[task]]


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


def get_split_size(datasetfile, split):
    key = f"nb_{split}"
    if key not in datasetfile:
        raise KeyError(f"{key} không có trong datasetfile")
    return int(datasetfile[key])


@torch.no_grad()
def evaluate_split_loss(model, dataloader, task, split, batch_size, device, datasetfile, loss_fn):
    split_size = get_split_size(datasetfile, split)
    if split_size <= 0:
        raise RuntimeError(f"Split '{split}' có 0 samples, không thể early stop")

    split_indices = np.arange(split_size, dtype=np.int64)
    model.eval()

    total_loss = 0.0
    total_count = 0
    for batch_ids in iter_batches(split_indices, batch_size, shuffle=False):
        images, y_dict = dataloader.load_ovr(batch_ids, category=split)
        x = torch.from_numpy(images).to(device)
        y = torch.from_numpy(y_dict[task]).float().to(device)

        logits = model.forward_task(x, task)
        loss = loss_fn(logits, y)

        batch_count = int(y.size(0))
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


def parse_task_set(text):
    if text is None or text.strip() == "":
        return set()
    return {x.strip() for x in text.split(",") if x.strip()}


def save_stable_alias(src_path, alias_path):
    """Create/update alias_path as a real file copy, never a symlink."""
    if os.path.exists(alias_path) or os.path.islink(alias_path):
        os.remove(alias_path)
    shutil.copy2(src_path, alias_path)


def train(args):
    task = get_task(args.shard)

    datasetfile, dataloader = load_dataset_config(args.dataset)
    input_shape = tuple(datasetfile["input_shape"])

    # Tính class imbalance để đặt pos_weight cho BCE (one-vs-rest)
    nb_train = int(getattr(dataloader, "train_size", datasetfile["nb_train"]))
    if nb_train <= 0:
        raise RuntimeError(
            "Train size = 0 trong HDF5. Hay chay lai prepare_data_ovr.py va kiem tra input_dir."
        )
    all_train_idx = np.arange(nb_train, dtype=np.int64)
    all_labels = dataloader.load_ovr_labels(all_train_idx, category="train")
    y_all = np.asarray(all_labels[task], dtype=np.int64)
    pos = y_all.sum()
    neg = len(y_all) - pos
    severe_task_set = parse_task_set(args.focal_tasks)

    if pos == 0:
        pos_weight_value = 1.0
        pos_ratio = 0.0
    else:
        pos_weight_value = float(neg) / float(pos)
        pos_ratio = float(pos) / float(pos + neg)

    # Tự động quyết định dùng focal theo mức imbalance của từng thuộc tính.
    # Nếu pos_ratio rất nhỏ hoặc rất lớn => imbalance nghiêm trọng => focal thường hữu ích hơn.
    # Có thể xem như tự "tìm thuộc tính đang bị imbalance".
    focal_auto_by_ratio = (
        args.focal_auto_min_pos_ratio > 0
        and (pos_ratio < args.focal_auto_min_pos_ratio or pos_ratio > (1.0 - args.focal_auto_min_pos_ratio))
    )
    # Bổ sung theo pos_weight (neg/pos)
    focal_auto_by_weight = (
        args.focal_auto_min_pos_weight > 1.0
        and pos_weight_value >= args.focal_auto_min_pos_weight
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OVRModel(input_shape=input_shape, dropout_rate=args.dropout_rate).to(device)

    slice_plan = load_slice_plan(args.container, task)
    if sum(len(s) for s in slice_plan) == 0:
        print(f"Empty slice plan for shard {args.shard} ({task})")
        return

    optimizer = make_optimizer(model, args.optimizer, args.learning_rate)

    use_focal = args.loss_mode == "focal" or (
        args.loss_mode == "auto"
        and (task in severe_task_set or focal_auto_by_ratio or focal_auto_by_weight)
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

    early_stop_enabled = args.early_stop_patience > 0
    if early_stop_enabled:
        if args.early_stop_split not in {"train", "val", "test"}:
            raise ValueError("early_stop_split phải là train, val hoặc test")
        split_size = get_split_size(datasetfile, args.early_stop_split)
        if split_size <= 0:
            raise RuntimeError(
                f"Split '{args.early_stop_split}' có 0 samples, không thể early stop"
            )
        print(
            f"[Shard {args.shard}][{task}] early_stop=on split={args.early_stop_split} "
            f"patience={args.early_stop_patience} min_delta={args.early_stop_min_delta:.6f} "
            f"restore_best={bool(args.early_stop_restore_best)}"
        )
    else:
        print(f"[Shard {args.shard}][{task}] early_stop=off")

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
                shard_ckpt = f"containers/{args.container}/cache/shard-{args.shard}.pt"
                save_stable_alias(ckpt_final, shard_ckpt)

                time_link = f"containers/{args.container}/times/shard-{args.shard}.time"
                if os.path.exists(time_final):
                    save_stable_alias(time_final, time_link)
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

        if early_stop_enabled:
            best_state = copy.deepcopy(model.state_dict())
            best_val_loss = float("inf")
            best_epoch = start_epoch - 1
            patience_left = args.early_stop_patience

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
            epoch_val_loss = None
            if early_stop_enabled:
                epoch_val_loss = evaluate_split_loss(
                    model,
                    dataloader,
                    task,
                    args.early_stop_split,
                    args.batch_size,
                    device,
                    datasetfile,
                    loss_fn,
                )
                if epoch_val_loss < (best_val_loss - args.early_stop_min_delta):
                    best_state = copy.deepcopy(model.state_dict())
                    best_val_loss = epoch_val_loss
                    best_epoch = epoch
                    patience_left = args.early_stop_patience
                else:
                    patience_left -= 1
            print(
                f"[Shard {args.shard}][Slice {slice_id}][Epoch {epoch+1}] "
                f"loss={running_loss:.4f} acc={acc:.2f}%"
                + (f" val_loss={epoch_val_loss:.6f}" if epoch_val_loss is not None else "")
            )

            if early_stop_enabled and patience_left <= 0:
                print(
                    f"[Shard {args.shard}][Slice {slice_id}] Early stopping at epoch {epoch+1}. "
                    f"Best {args.early_stop_split}_loss={best_val_loss:.6f} at epoch {best_epoch+1}"
                )
                break

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

        if early_stop_enabled and args.early_stop_restore_best:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), ckpt_final)
        with open(time_final, "w") as f:
            f.write(f"{cumulative_train_time + elapsed_time}\n")

        if slice_id == len(slice_plan) - 1:
            shard_ckpt = f"containers/{args.container}/cache/shard-{args.shard}.pt"
            save_stable_alias(ckpt_final, shard_ckpt)

            time_link = f"containers/{args.container}/times/shard-{args.shard}.time"
            save_stable_alias(time_final, time_link)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train CelebA OVR models với SISA framework."
    )
    parser.add_argument(
        "--container",
        required=True,
        help="Tên container (vd: celeba_ovr)"
    )
    parser.add_argument(
        "--shard",
        required=True,
        type=int,
        help="Shard ID (0 .. 26, tương ứng với 27 attributes)"
    )
    parser.add_argument(
        "--dataset",
        default="datasets/celebA/datasetfile_ovr",
        help="Path tới datasetfile_ovr"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Số epoch cho mỗi slice"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        help="Optimizer: adam hoặc sgd"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3,
        help="Dropout rate"
    )
    parser.add_argument(
        "--loss_mode",
        default="auto",
        help="Loss: bce, focal, hoặc auto (dùng focal cho imbalanced tasks)"
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=-1,
        help="Focal loss alpha (âm = auto, dương = cố định)"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma"
    )
    parser.add_argument(
        "--focal_auto_min_pos_ratio",
        type=float,
        default=0.2,
        help="Trong chế độ auto: dùng focal nếu pos_ratio < min hoặc > (1-min)"
    )
    parser.add_argument(
        "--focal_auto_min_pos_weight",
        type=float,
        default=3.0,
        help="Trong chế độ auto: dùng focal nếu pos_weight >= ngưỡng này"
    )
    parser.add_argument(
        "--focal_tasks",
        default="",
        help="Danh sách task cần dùng focal loss (nếu loss_mode=auto), phân cách bằng ','"
    )
    parser.add_argument(
        "--chkpt_interval",
        type=int,
        default=-1,
        help="Lưu checkpoint mỗi N epoch (âm = không lưu)"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Số epoch không cải thiện val loss trước khi dừng sớm (0 = tắt)"
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Mức giảm loss tối thiểu để xem là có cải thiện"
    )
    parser.add_argument(
        "--early_stop_split",
        default="val",
        help="Split dùng để theo dõi loss khi early stopping: train|val|test"
    )
    parser.add_argument(
        "--early_stop_restore_best",
        action="store_true",
        help="Khôi phục best checkpoint theo early_stop_split trước khi lưu cuối"
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
