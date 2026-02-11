# CelebA Dataset cho Machine Unlearning

## Tổng quan

CelebA (Celebrity Attributes) là dataset gồm 202,599 ảnh khuôn mặt người nổi tiếng với 40 thuộc tính nhị phân mỗi ảnh.

## Các thuộc tính trong CelebA

CelebA có **40 thuộc tính nhị phân**, mỗi thuộc tính có giá trị 0 (không có) hoặc 1 (có).

### Các thuộc tính được đề xuất cho Binary Classification

Dựa trên phân bố dữ liệu, các thuộc tính sau **cân bằng tốt** và phù hợp cho binary classification:

1. **Smiling** ⭐ (Phổ biến nhất)
   - Phân bố: ~50-50
   - Dễ train, kết quả tốt
   - **Đề xuất cho khóa luận**

2. **Male**
   - Phân bố: ~50-50
   - Rõ ràng, dễ đánh giá

3. **Young**
   - Phân bố: ~50-50
   - Phù hợp cho bài toán phân loại

4. **Attractive**
   - Phân bố: ~40-60
   - Có thể hơi lệch nhưng vẫn train được

5. **Blond_Hair**
   - Phân bố: ~20-80 (lệch)
   - Vẫn có thể dùng nhưng cần cân nhắc

## Cách sử dụng

### Bước 1: Tải dataset

```bash
cd datasets/celebA
python download.py
```

### Bước 2: Chuẩn bị dữ liệu

#### Option 1: Chọn thuộc tính tương tác
```bash
python prepare_data.py
```
Script sẽ hiển thị:
- Tất cả 40 thuộc tính và phân bố của chúng
- Top 15 thuộc tính cân bằng nhất
- Menu để chọn thuộc tính

#### Option 2: Chọn thuộc tính trực tiếp
```bash
# Dùng thuộc tính "Smiling" (đề xuất)
python prepare_data.py --attribute Smiling

# Hoặc dùng thuộc tính khác
python prepare_data.py --attribute Male
python prepare_data.py --attribute Young
```

#### Option 3: Chỉ xem danh sách thuộc tính
```bash
python prepare_data.py --list
```

### Bước 3: Train model

Sau khi prepare xong, bạn có thể train model:

```bash
# Từ thư mục gốc của project
python sisa.py --model celeba --train --dataset datasets/celebA/datasetfile \
    --epochs 30 --batch_size 32 --learning_rate 0.0001
```

## Gợi ý cho Khóa luận Machine Unlearning

### So sánh với CIFAR-10

| Dataset | Loại | Số lớp | Kích thước ảnh | Đặc điểm |
|---------|------|--------|----------------|----------|
| **CIFAR-10** | Multi-class | 10 | 32x32 | Phân loại đối tượng |
| **CelebA** | Binary | 2 | 64x64 | Phân loại thuộc tính khuôn mặt |

### Lý do chọn "Smiling" cho khóa luận

1. ✅ **Cân bằng tốt**: Phân bố ~50-50, dễ train
2. ✅ **Phổ biến**: Được dùng nhiều trong nghiên cứu
3. ✅ **Rõ ràng**: Dễ đánh giá và giải thích
4. ✅ **Tương phản với CIFAR-10**: 
   - CIFAR-10: Multi-class classification (10 lớp)
   - CelebA: Binary classification (2 lớp)
   - Giúp đánh giá kỹ thuật unlearning trên nhiều loại bài toán khác nhau

### Các thuộc tính khác có thể thử

Nếu muốn thử nghiệm với thuộc tính khác:

```bash
# Các thuộc tính cân bằng tốt
python prepare_data.py --attribute Male
python prepare_data.py --attribute Young
python prepare_data.py --attribute Attractive

# Các thuộc tính khác (có thể lệch hơn)
python prepare_data.py --attribute Blond_Hair
python prepare_data.py --attribute Wearing_Hat
python prepare_data.py --attribute Eyeglasses
```

## Cấu trúc file sau khi prepare

```
datasets/celebA/
├── celeba_train.npy      # Training data (162,770 ảnh)
├── celeba_test.npy       # Test data (39,829 ảnh)
├── datasetfile           # Metadata (chứa thông tin thuộc tính đã chọn)
└── dataloader.py         # Load dữ liệu
```

File `datasetfile` sẽ chứa:
```json
{
    "nb_train": 162770,
    "nb_test": 39829,
    "input_shape": [3, 64, 64],
    "nb_classes": 2,
    "dataloader": "dataloader",
    "attribute": "Smiling",
    "attribute_distribution": {
        "class_0": 101177,
        "class_1": 101422,
        "ratio_1_percent": 50.1
    }
}
```

## Lưu ý

1. **Thời gian prepare**: Việc load và resize 202,599 ảnh có thể mất 10-20 phút
2. **Dung lượng**: Dataset sau khi prepare sẽ chiếm khoảng 2-3 GB
3. **Thuộc tính**: Mỗi lần chạy `prepare_data.py` với thuộc tính khác sẽ tạo dataset mới. Nếu muốn đổi thuộc tính, cần xóa file `.npy` cũ hoặc đổi tên.

## Troubleshooting

### Lỗi: "Không tìm thấy list_attr_celeba.txt"
→ Chạy `python download.py` trước

### Lỗi: "Không tìm thấy img_align_celeba"
→ Chạy `python download.py` và đợi tải xong

### Muốn đổi thuộc tính
→ Xóa `celeba_train.npy` và `celeba_test.npy`, sau đó chạy lại `prepare_data.py` với thuộc tính mới
