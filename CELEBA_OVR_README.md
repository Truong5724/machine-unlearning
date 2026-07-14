# CelebA OVR Guide

Tai lieu nay huong dan chay toan bo pipeline CelebA OVR theo thu tu:
1. prepare_data_ovr.py
2. init_ovr.sh
3. train_ovr.sh
4. predict_ovr.sh (val->test)
5. data_ovr.sh (chi tong hop)

Luu y:
- Pipeline hien tai su dung chien luoc drop-model (bo thuoc tinh = bo shard/model do).
- Khong dung requestfile cho CelebA OVR.
- Moi thuoc tinh la bai toan nhi phan doc lap: sigmoid -> so sanh threshold -> yes/no.

## 1) Files lien quan

- Du lieu: datasets/celebA/prepare_data_ovr.py
- Dataloader: datasets/celebA/dataloader_ovr.py
- Model: architectures/celeba_ovr.py
- Partition: celeba_ovr_partition.py
- Train shard: sisa_celeba_ovr.py
- Aggregate/eval: aggregation_ovr_celebA.py
- Script data: example-scripts/celeba-sharding/data_ovr.sh
- Script init: example-scripts/celeba-sharding/init_ovr.sh
- Script train: example-scripts/celeba-sharding/train_ovr.sh
- Script predict: example-scripts/celeba-sharding/predict_ovr.sh

## 2) Buoc prepare data OVR (co train/val/test)

Chay tu root repo:

```bash
cd /home/tri/machine-unlearning

python datasets/celebA/prepare_data_ovr.py \
  --input_dir datasets/celebA/img_align_celeba \
  --attr_file datasets/celebA/list_attr_celeba.txt \
  --output_dir datasets/celebA \
  --train_samples 50000 \
  --val_samples 10000 \
  --test_samples 10000 \
  --seed 42
```

Sau khi chay xong se co:
- datasets/celebA/celeba_ovr_train.h5
- datasets/celebA/celeba_ovr_val.h5
- datasets/celebA/celeba_ovr_test.h5
- datasets/celebA/datasetfile_ovr

## 3) Buoc init partition OVR

```bash
bash example-scripts/celeba-sharding/init_ovr.sh \
  celeba_ovr \
  datasets/celebA/datasetfile_ovr \
  2 \
  42
```

Y nghia tham so:
- arg1: ten container
- arg2: datasetfile
- arg3: so slices moi shard
- arg4: seed

Mac dinh script tao 27 shard (0..26), moi shard ung voi 1 thuoc tinh.
Mac dinh moi shard co 2 slices (arg3 = 2).

## 4) Buoc train OVR

Train tat ca shard 0..26:

```bash
bash example-scripts/celeba-sharding/train_ovr.sh \
  celeba_ovr \
  0-26 \
  datasets/celebA/datasetfile_ovr
```

Ban co the doi hyper-params qua bien moi truong:

```bash
EPOCHS=8 \
BATCH_SIZE=64 \
LEARNING_RATE=0.0005 \
LOSS_MODE=auto \
FOCAL_TASKS=mustache,goatee,sideburns,double_chin,bags_under_eyes \
bash example-scripts/celeba-sharding/train_ovr.sh \
  celeba_ovr \
  0-26 \
  datasets/celebA/datasetfile_ovr
```

Train mot doan shard:

```bash
bash example-scripts/celeba-sharding/train_ovr.sh \
  celeba_ovr \
  10-18 \
  datasets/celebA/datasetfile_ovr
```

Train theo dung vi du ban can (0-2):

```bash
bash example-scripts/celeba-sharding/train_ovr.sh \
  celeba_ovr \
  0-2 \
  datasets/celebA/datasetfile_ovr
```

Script train_ovr.sh hien chap nhan shard_spec linh hoat:
- Range: 0-2
- Danh sach: 0,3,8
- Hon hop: 0-2,6,9-11

## 5) Buoc predict (val -> test + tune threshold)

predict_ovr.sh la script chay val/test:
- tune threshold rieng tung task tren val
- danh gia tren test
- luu thresholds ra file JSON

```bash
bash example-scripts/celeba-sharding/predict_ovr.sh \
  celeba_ovr \
  datasets/celebA/datasetfile_ovr \
  f1 \
  val \
  test \
  \
  \
  outputs/predict_val_test.json
```

Y nghia tham so predict_ovr.sh:
- arg1: container
- arg2: datasetfile
- arg3: objective tune (f1 hoac bacc)
- arg4: tune split (thuong la val)
- arg5: eval split (thuong la test)
- arg6: include_tasks CSV (de trong = all)
- arg7: exclude_tasks CSV
- arg8: save_json path (optional)

## 6) Buoc data (chi tong hop ket qua)

data_ovr.sh chi tong hop metric bang thresholds da duoc luu tu buoc predict.

Mac dinh doc thresholds tai thu muc:
- containers/celeba_ovr/outputs/thresholds/

Moi task co 1 file rieng, vi du:
- containers/celeba_ovr/outputs/thresholds/thresholds:young.json
- containers/celeba_ovr/outputs/thresholds/thresholds:male.json

```bash
bash example-scripts/celeba-sharding/data_ovr.sh \
  celeba_ovr \
  datasets/celebA/datasetfile_ovr \
  test
```

Chi dinh file thresholds + include/exclude:

```bash
bash example-scripts/celeba-sharding/data_ovr.sh \
  celeba_ovr \
  datasets/celebA/datasetfile_ovr \
  test \
  containers/celeba_ovr/outputs/thresholds \
  male,young,smiling \
  \
  outputs/data_aggregate.json
```

Y nghia tham so data_ovr.sh:
- arg1: container
- arg2: datasetfile
- arg3: split aggregate (train/val/test)
- arg4: thresholds_path (optional, mac dinh thu muc thresholds cua container; co the la file legacy JSON hoac thu muc threshold moi)
- arg5: include_tasks CSV (de trong = all)
- arg6: exclude_tasks CSV
- arg7: save_json path (optional)

## 7) Drop thuoc tinh (drop model)

Voi chien luoc hien tai, bo thuoc tinh nghia la bo shard/model do.
Khong co retrain cho thuoc tinh bi bo.

Vi du bo task smiling (shard 2):
- Cach 1: khong train shard 2.
- Cach 2: neu da train, xoa symlink shard do:

```bash
rm -f containers/celeba_ovr/cache/shard-2.pt
rm -f containers/celeba_ovr/times/shard-2.time
```

Khi aggregate/eval, script se tu dong bao missing task va bo qua.

## 8) Mot workflow day du de chay nhanh

```bash
cd /home/tri/machine-unlearning

python datasets/celebA/prepare_data_ovr.py \
  --train_samples 50000 \
  --val_samples 10000 \
  --test_samples 10000 \
  --seed 42

bash example-scripts/celeba-sharding/init_ovr.sh \
  celeba_ovr datasets/celebA/datasetfile_ovr 2 42

bash example-scripts/celeba-sharding/train_ovr.sh \
  celeba_ovr 0-26 datasets/celebA/datasetfile_ovr

bash example-scripts/celeba-sharding/predict_ovr.sh \
  celeba_ovr datasets/celebA/datasetfile_ovr f1 val test

bash example-scripts/celeba-sharding/data_ovr.sh \
  celeba_ovr datasets/celebA/datasetfile_ovr test
```

## 9) Troubleshooting

Loi khong tim thay celeba_ovr_val.h5:
- Ban chua chay lai prepare_data_ovr.py phien ban co val split.

Loi shard range:
- train_ovr.sh chi chap nhan shard trong 0-26.

CUDA out of memory:
- Giam BATCH_SIZE, hoac giam EPOCHS.

Metric thap:
- Dung predict_ovr.sh de tune threshold tren val truoc, sau do data_ovr.sh de tong hop.
