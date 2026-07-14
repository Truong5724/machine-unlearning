UTKFace Multi-task SISA
======================

Design:
- shard 0 => task gender, 2 slices: female / male
- shard 1 => task age, 5 slices: equal-width bins on [0..116]
- shard 2 => task race, 5 slices: white / black / asian / indian / others

Requirements:
- Rebuild UTKFace HDF5 by running datasets/UTKFace/prepare_data_multitask.py
  (new files include age/gender/race labels)

Run:
1) Prepare data ver2
  python datasets/UTKFace/prepare_data_multitask.py --img_dir datasets/UTKFace/UTKFace

2) Initialize partition
   bash example-scripts/utkface-sharding/init_multitask.sh utkface 0

3) Train all 3 shards
   bash example-scripts/utkface-sharding/train_multitask.sh utkface 0

4) Predict all 3 shards
   bash example-scripts/utkface-sharding/predict_multitask.sh utkface 0 argmax

5) Evaluate with data_ver2.sh
  python example-scripts/utkface-sharding/data_multitask.py --label 0 --container utkface --dataset datasets/UTKFace/datasetfile_ver2

Unlearn 1 slice (đúng ý tưởng "quên theo nhóm"):
- Gender shard (2 slices): slice 0=female, slice 1=male
  bash example-scripts/utkface-sharding/unlearn_shard_multitask.sh utkface forget-gender-slice1 0 1
- Age shard (5 slices): slice 0..4 là các bins tuổi đều trên [0..116]
  bash example-scripts/utkface-sharding/unlearn_shard_multitask.sh utkface forget-age-slice2 1 2
- Race shard (5 slices): slice 0..4 tương ứng race 0..4
  bash example-scripts/utkface-sharding/unlearn_shard_multitask.sh utkface forget-race-slice4 2 4

Then retrain only that shard with the same label:
  python sisa_utkface_multitask.py --train --container utkface --dataset datasets/UTKFace/datasetfile_ver2 --shard <0|1|2> --label <forget-label>

Then predict + evaluate again:
  bash example-scripts/utkface-sharding/predict_multitask.sh utkface <forget-label> argmax
  python example-scripts/utkface-sharding/data_multitask.py --label <forget-label> --container utkface --dataset datasets/UTKFace/datasetfile_ver2
