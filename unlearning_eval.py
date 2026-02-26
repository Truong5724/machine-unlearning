"""
Unlearning evaluation for CIFAR-10 (single-file, hardcoded config except request label).

Behavior:
- Hardcoded: container, model name, number of shards, strategy (proportional), batch size.
- CLI: only `--label` to pick which `requestfile:<label>.npy` to use.
- Assumes `datasets/CIFAR-10/cifar10_train.npy` saved as `np.save('cifar10_train.npy', {'X': X, 'y': y})` (user-tested format).
- Assumes `containers/<container>/requestfile:<label>.npy` and `containers/<container>/splitfile.npy` are 2D arrays (no branching logic).
"""

import argparse
import os
import numpy as np
import torch
from torch.nn.functional import softmax
import importlib

# ----- Hardcoded configuration (CIFAR-10) -----
CONTAINER = "cifar10"
MODEL_NAME = "cifar10"
SHARDS = 5
STRATEGY = "proportional"  
BATCH_SIZE = 32
NB_CLASSES = 10
INPUT_SHAPE = [3, 32, 32]

# ----- CLI: only request label -----
parser = argparse.ArgumentParser()
parser.add_argument("--label", default="latest", help="Requestfile label to use (overrides default)")
cli = parser.parse_args()
LABEL = cli.label

print(f"Container: {CONTAINER}")
print(f"Model: {MODEL_NAME}")
print(f"Shards: {SHARDS}")
print(f"Request label: {LABEL}")
print(f"Strategy: {STRATEGY}")
print("=" * 60)

# ----- Setup -----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_lib = importlib.import_module(f"architectures.{MODEL_NAME}")
container_path = f"containers/{CONTAINER}"
cache_dir = os.path.join(container_path, "cache")

# ----- STEP 1: Load CIFAR-10 data (dict {'X','y'}) -----
print("\n[Step 1] Loading CIFAR-10 data and labels...")
train_npy = "datasets/CIFAR-10/cifar10_train.npy"
if not os.path.exists(train_npy):
    raise FileNotFoundError(f"CIFAR-10 train file not found: {train_npy}")

# per user's note, file loads as array-object where the dict is at [0]['y'] etc
cifar10_obj = np.load(train_npy, allow_pickle=True).reshape((1,))[0]
X = np.array(cifar10_obj["X"])  
y = np.array(cifar10_obj["y"]).astype(np.int64)

print(f"Loaded CIFAR-10: X {X.shape}, y {y.shape}")

# ----- STEP 2: Load requestfile and map to indices -----
print("\n[Step 2] Loading requestfile and mapping to indices...")
requestfile_path = os.path.join(container_path, f"requestfile:{LABEL}.npy")
if not os.path.exists(requestfile_path):
    raise FileNotFoundError(f"Requestfile not found: {requestfile_path}")

requests = np.load(requestfile_path, allow_pickle=True)
print(f"Requestfile shape: {requests.shape}")

# Assume requests is a 2D numeric array: flatten and uniq
all_unlearn_indices = np.unique(
    np.concatenate([np.array(r, dtype=np.int64) for r in requests])
)
print(f"Total unlearning indices: {len(all_unlearn_indices)}")

unlearn_data = X[all_unlearn_indices]
unlearn_labels = y[all_unlearn_indices]

print(f"Unlearning data shape: {unlearn_data.shape}")
print(f"Unlearning labels shape: {unlearn_labels.shape}")
print(f"Label distribution: {np.bincount(unlearn_labels)}")

# ----- STEP 3: Load models & predict -----
print("\n[Step 3] Loading models and predicting...")

all_shard_outputs = []

for sid in range(SHARDS):
    print(f"  Shard {sid + 1}/{SHARDS}...", end=" ")
    model = model_lib.Model(input_shape=INPUT_SHAPE, nb_classes=NB_CLASSES)
    model.to(device)

    model_path = os.path.join(cache_dir, f"shard-{sid}:{LABEL}.pt")
    if not os.path.exists(model_path):
        print("SKIP (model not found)")
        continue

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    shard_outputs = np.empty((0, NB_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(unlearn_data), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(unlearn_data))
            batch = unlearn_data[i:j].astype(np.float32) / 255.0
            tensor = torch.from_numpy(batch).to(device)
            
            logits = model(tensor)
            probs = softmax(logits, dim=1).cpu().numpy()
            shard_outputs = np.concatenate((shard_outputs, probs), axis=0)

    all_shard_outputs.append(shard_outputs)
    print(f"OK ({shard_outputs.shape[0]} predictions)")

all_shard_outputs = np.array(all_shard_outputs)  # shape (num_shards_used, num_samples, nb_classes)
print(f"Prediction array shape: {all_shard_outputs.shape}")

# ----- STEP 4: Aggregation (proportional) -----
print("\n[Step 4] Aggregating predictions...")
split_path = os.path.join(container_path, "splitfile.npy")
if not os.path.exists(split_path):
    raise FileNotFoundError(f"Splitfile not found: {split_path}")

split = np.load(split_path, allow_pickle=True)
# split is 2D array/list of shard indices; compute shard sizes
shard_sizes = np.array([len(s) for s in split], dtype=float)
weights = shard_sizes / shard_sizes.sum()
print(f"Weights (proportional): {weights}")

votes = np.argmax(
    np.tensordot(weights.reshape(1, -1), all_shard_outputs, axes=1), axis=2
).reshape((all_shard_outputs.shape[1],))
print(f"Final predictions shape: {votes.shape}")

# ----- STEP 5: Compute accuracy -----
print("\n[Step 5] Computing accuracy...")
accuracy = np.mean(votes == unlearn_labels)
correct = int(np.sum(votes == unlearn_labels))

per_class_acc = {}
for cls in range(NB_CLASSES):
    mask = unlearn_labels == cls
    if np.sum(mask) > 0:
        per_class_acc[cls] = float(np.mean(votes[mask] == unlearn_labels[mask]))

print("\n" + "=" * 60)
print("UNLEARNING EVALUATION RESULTS:")
print("=" * 60)
print(f"Total samples: {len(unlearn_labels)}")
print(f"Correct predictions: {correct}")
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nPer-class accuracy:")
for cls, acc in per_class_acc.items():
    print(f"  Class {cls}: {acc:.4f}")
print("=" * 60)

# ----- Save results -----
results_file = os.path.join(container_path, f"unlearning_eval_{LABEL}.txt")
predictions_file = os.path.join(container_path, f"unlearning_predictions_{LABEL}.npy")

np.save(predictions_file, votes)
with open(results_file, "w") as f:
    f.write("Unlearning Evaluation Results (CIFAR-10)\n")
    f.write(f"Container: {CONTAINER}\n")
    f.write(f"Label: {LABEL}\n")
    f.write(f"Strategy: {STRATEGY}\n")
    f.write(f"Shards: {SHARDS}\n")
    f.write(f"Total samples: {len(unlearn_labels)}\n")
    f.write(f"Correct: {correct}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\nPer-class accuracy:\n")
    for cls, acc in per_class_acc.items():
        f.write(f"  Class {cls}: {acc:.4f}\n")
    f.write(f"Predictions saved to: {predictions_file}\n")

print(f"\nResults saved to: {results_file}")
print(f"Predictions saved to: {predictions_file}")

