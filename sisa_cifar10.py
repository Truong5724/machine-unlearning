import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateauScheduler
from torch.nn.functional import one_hot, softmax
from sharded import sizeOfShard, getShardHash, fetchShardBatch, fetchValBatch, fetchTestBatch
import os
from glob import glob
from time import time
import json
from tqdm import tqdm
import argparse
import random
import torchvision.transforms as transforms

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="purchase", help="Architecture to use, default purchase"
)

parser.add_argument(
    "--train", action="store_true", help="Perform SISA training on the shard"
)
parser.add_argument("--test", action="store_true", help="Compute shard predictions")

parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Train for the specified number of epochs, default 20",
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Size of the batches, relevant for both train and test, default 16",
)
parser.add_argument(
    "--dropout_rate",
    default=0.2,
    type=float,
    help="Dropout rate, if relevant, default 0.2",
)
parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001"
)

parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")

parser.add_argument(
    "--output_type",
    default="argmax",
    help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax",
)

parser.add_argument("--container", help="Name of the container")
parser.add_argument("--shard", type=int, help="Index of the shard to train/test")
parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
)
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)

parser.add_argument(
    "--chkpt_interval",
    default=1,
    type=int,
    help="Interval (in epochs) between two chkpts, -1 to disable chackpointing, default 1",
)
parser.add_argument(
    "--label",
    default="latest",
    help="Label to be used on simlinks and outputs, default latest",
)

args = parser.parse_args()

# Import the architecture.
from importlib import import_module
model_lib = import_module("architectures.{}".format(args.model))

# Retrive dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())
input_shape = tuple(datasetfile["input_shape"])
nb_classes = datasetfile["nb_classes"]

# Use GPU if available.
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # pylint: disable=no-member

set_seed(1)
model = model_lib.Model(input_shape, nb_classes, dropout_rate=args.dropout_rate)
model.to(device)

# Instantiate loss and optimizer.
loss_fn = CrossEntropyLoss()
            
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == 'max':
            improvement = metric - self.best_score
        else:
            improvement = self.best_score - metric

        if improvement > self.min_delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

# Augmentation config.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

if args.train:
    shard_size = sizeOfShard(args.container, args.shard)
    slice_size = shard_size // args.slices
    avg_epochs_per_slice = args.epochs

    loaded = False

    for sl in tqdm(range(args.slices)):
        set_seed((sl + 1) * 100)
        
        # Get slice hash using sharded lib.
        slice_hash = getShardHash(
            args.container, args.label, args.shard, until=(sl + 1) * slice_size
        )

        # Initialize state.
        elapsed_time = 0
        start_epoch = 0
        slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(
            sl * avg_epochs_per_slice
        )

        # If weights are already in memory (from previous slice), skip loading.
        if not loaded:
            # Look for a recovery checkpoint for the slice.
            recovery_list = glob(
                "containers/{}/cache/{}_*.pt".format(args.container, slice_hash)
            )
            if len(recovery_list) > 0:
                print(
                    "Recovery mode for shard {} on slice {}".format(args.shard, sl)
                )

                # Load weights.
                model.load_state_dict(torch.load(recovery_list[0]))
                start_epoch = int(
                    recovery_list[0].split("/")[-1].split(".")[0].split("_")[1]
                )

                # Load time
                with open(
                    "containers/{}/times/{}_{}.time".format(
                        args.container, slice_hash, start_epoch
                    ),
                    "r",
                ) as f:
                    elapsed_time = float(f.read())

            else:
                # If model weights of the slice exist, skip the slice.
                if os.path.exists(
                    "containers/{}/cache/{}.pt".format(args.container, slice_hash)
                ):
                    print("Model weights for shard {} on slice {} exists".format(args.shard, sl))
                    continue
                
                if sl > 0:
                    previous_slice_hash = getShardHash(
                        args.container, args.label, args.shard, until=sl * slice_size
                    )

                    # Load weights.
                    model.load_state_dict(
                        torch.load(
                            "containers/{}/cache/{}.pt".format(
                                args.container, previous_slice_hash
                            )
                        )
                    )
                
                    print("Load model weights from slice {} for shard {} successfully".format(sl - 1, args.shard))

            # Mark model as loaded for next slices.
            loaded = True

        # Init optimizer
        if args.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == "sgd":
            optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        else:
            raise "Unsupported optimizer"

        # Init Scheduler
        reduce_lr = ReduceLROnPlateauScheduler(optimizer,
                                                mode='min', 
                                                factor=0.5, 
                                                patience=5, 
                                                min_lr=1e-5)

        # Init EarlyStopping
        early_stopping = EarlyStopping(patience=20, min_delta=0.002, mode='min')

        # Actual training.
        train_time = 0.0

        for epoch in tqdm(range(start_epoch, slice_epochs)):
            model.train()

            train_loss = 0.0
            train_batches = 0

            epoch_start_time = time()

            for images, labels in fetchShardBatch(
                args.container,
                args.label,
                args.shard,
                args.batch_size,
                args.dataset,
                until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
            ):
                # Convert data to torch format and send to selected device.
                images = torch.from_numpy(images)

                images = torch.stack([
                    train_transform(img)
                    for img in images
                ])

                gpu_images = images.to(device)
                gpu_labels = torch.from_numpy(labels).to(device)

                forward_start_time = time()

                # Perform basic training step.
                logits = model(gpu_images)
                loss = loss_fn(logits, gpu_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_time += time() - forward_start_time

                # Calculate train loss for the epoch.
                train_loss += loss.item()
                train_batches += 1

            train_loss /= train_batches

            ### VALIDATION (to monitor learning rate)
            model.eval()   

            correct = 0
            total = 0

            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():  
                for val_images, val_labels in fetchValBatch(args.dataset, args.batch_size):
                    gpu_val_images = torch.from_numpy(val_images).to(device)
                    gpu_val_labels = torch.from_numpy(val_labels).to(device)

                    outputs = model(gpu_val_images)
                    preds = torch.argmax(outputs, dim=1)

                    # Calculate validation loss for the epoch.
                    loss = loss_fn(outputs, gpu_val_labels)

                    val_loss += loss.item()
                    val_batches += 1

                    correct += (preds == gpu_val_labels).sum().item()
                    total += gpu_val_labels.size(0)

            val_loss /= val_batches
            val_acc = 100 * correct / total
            print(f" [Epoch {epoch+1}] - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} - Val accuracy : {val_acc:.2f}%")

            # Check early stopping
            if early_stopping(val_loss):
                print("Early stopping triggered!")
                break

            # Cập nhật ReduceLROnPlateau scheduler dựa trên val_loss
            reduce_lr.step(val_loss)

            # Create a checkpoint every chkpt_interval.
            if (
                args.chkpt_interval != -1
                and epoch % args.chkpt_interval == args.chkpt_interval - 1
            ):
                # Save weights
                torch.save(
                    model.state_dict(),
                    "containers/{}/cache/{}_{}.pt".format(
                        args.container, slice_hash, epoch
                    ),
                )

                # Save time
                with open(
                    "containers/{}/times/{}_{}.time".format(
                        args.container, slice_hash, epoch
                    ),
                    "w",
                ) as f:
                    f.write("{}\n".format(train_time + elapsed_time))

                # Remove previous checkpoint.
                if os.path.exists(
                    "containers/{}/cache/{}_{}.pt".format(
                        args.container, slice_hash, epoch - args.chkpt_interval
                    )
                ):
                    os.remove(
                        "containers/{}/cache/{}_{}.pt".format(
                            args.container, slice_hash, epoch - args.chkpt_interval
                        )
                    )
                if os.path.exists(
                    "containers/{}/times/{}_{}.time".format(
                        args.container, slice_hash, epoch - args.chkpt_interval
                    )
                ):
                    os.remove(
                        "containers/{}/times/{}_{}.time".format(
                            args.container, slice_hash, epoch - args.chkpt_interval
                        )
                    )

            # When training is complete, save slice.
            torch.save(
                model.state_dict(),
                "containers/{}/cache/{}.pt".format(args.container, slice_hash),
            )
            with open(
                "containers/{}/times/{}.time".format(args.container, slice_hash), "w"
            ) as f:
                f.write("{}\n".format(train_time + elapsed_time))

            # Remove previous checkpoint.
            if os.path.exists(
                "containers/{}/cache/{}_{}.pt".format(
                    args.container, slice_hash, args.epochs - args.chkpt_interval
                )
            ):
                os.remove(
                    "containers/{}/cache/{}_{}.pt".format(
                        args.container, slice_hash, args.epochs - args.chkpt_interval
                    )
                )
            if os.path.exists(
                "containers/{}/times/{}_{}.time".format(
                    args.container, slice_hash, args.epochs - args.chkpt_interval
                )
            ):
                os.remove(
                    "containers/{}/times/{}_{}.time".format(
                        args.container, slice_hash, args.epochs - args.chkpt_interval
                    )
                )

        # If this is the last slice, create a symlink attached to it.
        if sl == args.slices - 1:
            os.symlink(
                "{}.pt".format(slice_hash),
                "containers/{}/cache/shard-{}:{}.pt".format(
                    args.container, args.shard, args.label
                ),
            )
            if not os.path.exists(
                "containers/{}/times/shard-{}:{}.time".format(
                    args.container, args.shard, args.label
                )
            ):
                os.symlink(
                    "null.time",
                    "containers/{}/times/shard-{}:{}.time".format(
                        args.container, args.shard, args.label
                    ),
                )


if args.test:
    # Load model weights from shard checkpoint (last slice).
    model.load_state_dict(
        torch.load(
            "containers/{}/cache/shard-{}:{}.pt".format(
                args.container, args.shard, args.label
            )
        )
    )

    model.eval()

    # Compute predictions batch per batch.
    outputs = np.empty((0, nb_classes))

    with torch.no_grad():
        for images, _ in fetchTestBatch(args.dataset, args.batch_size):
            # Convert data to torch format and send to selected device.
            gpu_images = torch.from_numpy(images).to(device)  # pylint: disable=no-member

            if args.output_type == "softmax":
                # Actual batch prediction.
                logits = model(gpu_images)
                predictions = softmax(logits, dim=1).to("cpu")  # Send back to cpu.

                # Convert back to numpy and concatenate with previous batches.
                outputs = np.concatenate((outputs, predictions.numpy()))

            else:
                # Actual batch prediction.
                logits = model(gpu_images)
                predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

                # Convert to one hot, send back to cpu, convert back to numpy and concatenate with previous batches.
                out = one_hot(predictions, nb_classes).to("cpu")
                outputs = np.concatenate((outputs, out.numpy()))

    # Save outputs in numpy format.
    outputs = np.array(outputs)
    np.save(
        "containers/{}/outputs/shard-{}:{}.npy".format(
            args.container, args.shard, args.label
        ),
        outputs,
    )
