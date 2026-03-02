"""
evaluate_forgot.py - Đánh giá model trên tập FORGOT

Mục đích: Kiểm tra xem model có thực sự quên dữ liệu không
- Model baseline (label=0): Accuracy cao trên forgot set
- Model unlearned (label=100): Accuracy THẤP trên forgot set (chứng tỏ đã quên!)

Usage:
    python evaluate_forgot.py --container utkface --label 100 --shards 4
"""

import numpy as np
import torch
import json
import argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--container', default='utkface', help='Container name')
parser.add_argument('--label', type=int, required=True, help='Label to evaluate')
parser.add_argument('--shards', type=int, required=True, help='Number of shards')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--model', default='utkface', help='Model architecture')
parser.add_argument('--dataset', default='datasets/UTKFace/datasetfile', help='Dataset file')
args = parser.parse_args()

# Load dataset metadata
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())

input_shape = tuple(datasetfile["input_shape"])
nb_classes = datasetfile["nb_classes"]

# Load dataloader
module_path = '.'.join(args.dataset.split('/')[:-1] + [datasetfile['dataloader']])
dataloader = import_module(module_path)

# Load model architecture
model_lib = import_module(f"architectures.{args.model}")

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("EVALUATE ON FORGOT SET")
print("=" * 70)
print(f"Container: {args.container}")
print(f"Label: {args.label}")
print(f"Shards: {args.shards}")
print()

# Load splitfile và requestfile
splitfile = np.load(f'containers/{args.container}/splitfile.npy', allow_pickle=True)
requestfile = np.load(f'containers/{args.container}/requestfile:{args.label}.npy', allow_pickle=True)

# Tổng hợp tất cả forgot indices
all_forgot_indices = []
for shard_idx in range(args.shards):
    forgot_in_shard = requestfile[shard_idx]
    all_forgot_indices.extend(forgot_in_shard)

all_forgot_indices = np.array(all_forgot_indices)

if len(all_forgot_indices) == 0:
    print("⚠️  No forgot data (label=0 is baseline)")
    print(f"   Use label>0 to evaluate unlearning")
    exit(0)

print(f"📊 Forgot set size: {len(all_forgot_indices)} samples")
print()

# Evaluate từng shard
print("🔄 Evaluating each shard on forgot set...")
print()

shard_predictions = []
shard_accuracies = []

for shard_idx in range(args.shards):
    print(f"Shard {shard_idx}:")
    
    # Load model
    checkpoint = f"containers/{args.container}/cache/shard-{shard_idx}:{args.label}.pt"
    
    try:
        model = model_lib.Model(input_shape, nb_classes, dropout_rate=0.4)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"  ❌ Checkpoint not found: {checkpoint}")
        continue
    
    # Get forgot indices từ shard này
    forgot_in_shard = requestfile[shard_idx]
    
    if len(forgot_in_shard) == 0:
        print(f"  ⚠️  No forgot data in this shard")
        continue
    
    print(f"  Forgot samples: {len(forgot_in_shard)}")
    
    # Load forgot data
    X_forgot, y_forgot = dataloader.load(forgot_in_shard, category='train')
    
    # Predict
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(X_forgot), args.batch_size):
            batch_X = X_forgot[i:i+args.batch_size]
            batch_y = y_forgot[i:i+args.batch_size]
            
            gpu_X = torch.from_numpy(batch_X).to(device)
            gpu_y = torch.from_numpy(batch_y).to(device)
            
            logits = model(gpu_X)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == gpu_y).sum().item()
            total += len(batch_y)
    
    accuracy = correct / total * 100
    shard_accuracies.append(accuracy)
    
    print(f"  Accuracy on forgot set: {accuracy:.2f}%")
    print()

# Overall accuracy
if len(shard_accuracies) > 0:
    avg_accuracy = np.mean(shard_accuracies)
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Average accuracy on forgot set: {avg_accuracy:.2f}%")
    print()
    
    # Interpretation
    print("💡 INTERPRETATION:")
    if avg_accuracy > 80:
        print("  ❌ Model still remembers forgot data well!")
        print("     Unlearning may NOT be effective")
    elif avg_accuracy > 60:
        print("  ⚠️  Model partially remembers forgot data")
        print("     Unlearning has some effect")
    elif avg_accuracy > 40:
        print("  ✅ Model performance degraded significantly")
        print("     Unlearning is working!")
    else:
        print("  ✅✅ Model has largely forgotten the data")
        print("     Strong unlearning effect!")
    
    print()
    print("📊 COMPARISON GUIDE:")
    print("  Baseline (label=0): Should be ~92% on forgot set")
    print(f"  Unlearned (label={args.label}): {avg_accuracy:.2f}% on forgot set")
    print(f"  Difference: {92 - avg_accuracy:.2f}% ← Unlearning effect!")
    
else:
    print("❌ No results - check if models are trained")

print("=" * 70)