"""
Chuẩn bị CelebA dataset cho SISA training - PHIÊN BẢN TỐI ƯU CHO COLAB
- Sử dụng HDF5 thay vì .npy để tránh load toàn bộ vào RAM
- Load và xử lý ảnh theo batch để tiết kiệm memory
- Hỗ trợ resume nếu bị gián đoạn

Yêu cầu: pip install h5py tqdm Pillow
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import h5py
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable


def load_attributes(attr_file):
    """Load attribute labels from list_attr_celeba.txt"""
    with open(attr_file, 'r') as f:
        lines = f.readlines()

    # First line is number of images
    num_images = int(lines[0].strip())
    # Second line is attribute names
    attr_names = lines[1].strip().split()

    # Rest are image attributes
    attributes = []
    filenames = []
    for line in lines[2:]:
        parts = line.strip().split()
        filenames.append(parts[0])
        # Convert -1/1 to 0/1
        attrs = [(int(x) + 1) // 2 for x in map(int, parts[1:])]
        attributes.append(attrs)

    return filenames, np.array(attributes, dtype=np.int64), attr_names


def load_images_batch(img_dir, filenames, target_size=(64, 64), batch_size=1000):
    """
    Generator: Tải ảnh theo batch để tiết kiệm RAM

    Yields:
        batch_images: numpy array shape (batch_size, 3, H, W)
        batch_indices: list các index đã load thành công
    """
    batch_images = []
    batch_indices = []
    failed = []

    for idx, fname in enumerate(filenames):
        img_path = os.path.join(img_dir, fname)
        try:
            if not os.path.exists(img_path):
                failed.append(fname)
                continue

            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)  # HWC -> CHW

            batch_images.append(img_array)
            batch_indices.append(idx)

            # Khi đủ batch_size hoặc là ảnh cuối, yield batch
            if len(batch_images) == batch_size or idx == len(filenames) - 1:
                yield np.array(batch_images, dtype=np.uint8), batch_indices
                batch_images = []
                batch_indices = []

        except Exception as e:
            print(f"\n⚠️  Lỗi khi tải {fname}: {e}")
            failed.append(fname)
            continue

    if failed:
        print(f"\n⚠️  Không tải được {len(failed)} ảnh")


def save_to_hdf5(h5_file, filenames, labels, img_dir, target_size=(64, 64), batch_size=1000):
    """
    Lưu ảnh vào HDF5 file theo batch
    HDF5 cho phép truy cập random access mà không cần load toàn bộ vào RAM
    """
    total_images = len(filenames)

    # Tạo datasets trong HDF5
    # chunks=(1, 3, 64, 64) cho phép đọc từng ảnh riêng lẻ hiệu quả
    images_dataset = h5_file.create_dataset(
        'images',
        shape=(total_images, 3, target_size[0], target_size[1]),
        dtype='uint8',
        chunks=(1, 3, target_size[0], target_size[1]),
        compression='gzip',  # Nén để tiết kiệm disk space
        compression_opts=4   # Level 4: cân bằng giữa tốc độ và tỷ lệ nén
    )

    labels_dataset = h5_file.create_dataset(
        'labels',
        shape=(total_images,),
        dtype='int64'
    )

    # Load và lưu theo batch
    print(f"Đang lưu {total_images} ảnh vào HDF5...")
    total_saved = 0

    for batch_images, batch_indices in tqdm(
        load_images_batch(img_dir, filenames, target_size, batch_size),
        total=total_images // batch_size + 1,
        desc="Processing batches"
    ):
        # Lưu batch vào HDF5
        for i, idx in enumerate(batch_indices):
            images_dataset[idx] = batch_images[i]
            labels_dataset[idx] = labels[idx]

        total_saved += len(batch_indices)

    print(f"✅ Đã lưu {total_saved}/{total_images} ảnh")
    return total_saved


def analyze_attributes(attributes, attr_names):
    """Phân tích và hiển thị phân bố của các thuộc tính"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH CÁC THUỘC TÍNH CELEBA (40 thuộc tính)")
    print("=" * 80)
    print(f"{'Thuộc tính':<20} {'Class 0':<12} {'Class 1':<12} {'Tỷ lệ 1':<10} {'Đánh giá'}")
    print("-" * 80)

    recommended = ["Smiling", "Male", "Young", "Attractive", "Blond_Hair",
                   "Wearing_Hat", "Wearing_Necktie", "High_Cheekbones"]

    attr_stats = []
    for i, attr_name in enumerate(attr_names):
        labels = attributes[:, i]
        count_0 = np.sum(labels == 0)
        count_1 = np.sum(labels == 1)
        ratio_1 = count_1 / len(labels) * 100

        if 30 <= ratio_1 <= 70:
            rating = "✅ Tốt"
        elif 20 <= ratio_1 < 30 or 70 < ratio_1 <= 80:
            rating = "⚠️  Khá"
        else:
            rating = "❌ Lệch"

        if attr_name in recommended:
            rating += " (Đề xuất)"

        attr_stats.append({
            'name': attr_name,
            'count_0': count_0,
            'count_1': count_1,
            'ratio_1': ratio_1,
            'rating': rating
        })

    attr_stats.sort(key=lambda x: abs(x['ratio_1'] - 50))

    print("\n📊 TOP 15 THUỘC TÍNH CÂN BẰNG NHẤT:")
    for stat in attr_stats[:15]:
        print(f"{stat['name']:<20} {stat['count_0']:<12} {stat['count_1']:<12} "
              f"{stat['ratio_1']:>5.1f}%    {stat['rating']}")

    print("\n" + "=" * 80)
    return attr_stats


def main():
    parser = argparse.ArgumentParser(
        description='Chuẩn bị CelebA dataset - TỐI ƯU CHO COLAB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python prepare_data_optimized.py                    # Menu chọn thuộc tính
  python prepare_data_optimized.py --attribute Smiling # Chọn "Smiling"
  python prepare_data_optimized.py --batch_size 500    # Giảm batch size nếu RAM ít
        """
    )
    parser.add_argument(
        '--attribute', '-a',
        type=str,
        default=None,
        help='Tên thuộc tính (ví dụ: Smiling, Male, Young)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='Chỉ liệt kê thuộc tính, không prepare'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Số ảnh load mỗi batch (default: 1000, giảm nếu RAM ít)'
    )
    args = parser.parse_args()

    # Cấu hình
    img_dir = "img_align_celeba/img_align_celeba"
    attr_file = "list_attr_celeba.txt"
    target_size = (64, 64)
    train_size = 162770

    # Kiểm tra file
    if not os.path.exists(attr_file):
        raise FileNotFoundError(
            f"Không tìm thấy {attr_file}. Hãy chạy download.py trước!")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(
            f"Không tìm thấy {img_dir}. Hãy chạy download.py trước!")

    print("=" * 80)
    print("CHUẨN BỊ CELEBA - TỐI ƯU CHO COLAB (HDF5)")
    print("=" * 80)

    print("\nĐang tải thuộc tính...")
    filenames, attributes, attr_names = load_attributes(attr_file)
    print(f"✅ Đã tải {len(filenames)} mẫu với {len(attr_names)} thuộc tính")

    # Phân tích thuộc tính
    attr_stats = analyze_attributes(attributes, attr_names)

    if args.list:
        print("\n✅ Đã liệt kê thuộc tính.")
        return

    # Chọn thuộc tính
    if args.attribute:
        target_attr = args.attribute
    else:
        print("\n" + "=" * 80)
        print("CHỌN THUỘC TÍNH")
        print("=" * 80)
        print("\nThuộc tính đề xuất:")
        recommended = ["Smiling", "Male", "Young", "Attractive", "Blond_Hair"]
        for i, attr in enumerate(recommended, 1):
            if attr in attr_names:
                idx = attr_names.index(attr)
                labels = attributes[:, idx]
                ratio = np.sum(labels == 1) / len(labels) * 100
                print(f"  {i}. {attr:<20} (Tỷ lệ positive: {ratio:.1f}%)")

        print("\nNhấn Enter để dùng 'Smiling'")
        user_input = input("Chọn (số/tên): ").strip()

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(recommended):
                target_attr = recommended[idx]
            else:
                target_attr = "Smiling"
        elif user_input:
            target_attr = user_input
        else:
            target_attr = "Smiling"

    if target_attr not in attr_names:
        print(f"\n❌ Không tìm thấy '{target_attr}', dùng 'Smiling'")
        target_attr = "Smiling"

    attr_idx = attr_names.index(target_attr)
    labels = attributes[:, attr_idx]

    # Thống kê
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    ratio_1 = count_1 / len(labels) * 100

    print("\n" + "=" * 80)
    print(f"THUỘC TÍNH: '{target_attr}'")
    print("=" * 80)
    print(f"   Class 0: {count_0:,} ({100-ratio_1:.1f}%)")
    print(f"   Class 1: {count_1:,} ({ratio_1:.1f}%)")
    print("=" * 80)

    # Chia train/test
    train_filenames = filenames[:train_size]
    test_filenames = filenames[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    # Lưu train set vào HDF5
    print("\n📦 Đang tạo celeba_train.h5...")
    if not os.path.exists('celeba_train.h5'):
        with h5py.File('celeba_train.h5', 'w') as h5f:
            # Lưu metadata
            h5f.attrs['attribute'] = target_attr
            h5f.attrs['target_size'] = target_size
            h5f.attrs['num_samples'] = len(train_filenames)

            # Lưu ảnh và labels
            save_to_hdf5(h5f, train_filenames, train_labels, img_dir,
                         target_size, args.batch_size)
        print("✅ Đã lưu celeba_train.h5")
    else:
        print("✅ celeba_train.h5 đã tồn tại")

    # Lưu test set
    print("\n📦 Đang tạo celeba_test.h5...")
    if not os.path.exists('celeba_test.h5'):
        with h5py.File('celeba_test.h5', 'w') as h5f:
            h5f.attrs['attribute'] = target_attr
            h5f.attrs['target_size'] = target_size
            h5f.attrs['num_samples'] = len(test_filenames)

            save_to_hdf5(h5f, test_filenames, test_labels, img_dir,
                         target_size, args.batch_size)
        print("✅ Đã lưu celeba_test.h5")
    else:
        print("✅ celeba_test.h5 đã tồn tại")

    # Tạo datasetfile
    print("\n📄 Đang tạo datasetfile...")
    with h5py.File('celeba_train.h5', 'r') as h5f:
        actual_train_size = h5f.attrs['num_samples']
    with h5py.File('celeba_test.h5', 'r') as h5f:
        actual_test_size = h5f.attrs['num_samples']

    dataset_info = {
        "nb_train": int(actual_train_size),
        "nb_test": int(actual_test_size),
        "input_shape": [3, target_size[0], target_size[1]],
        "nb_classes": 2,
        "dataloader": "dataloader",
        "attribute": target_attr,
        "storage_format": "hdf5",
        "attribute_distribution": {
            "class_0": int(count_0),
            "class_1": int(count_1),
            "ratio_1_percent": float(ratio_1)
        }
    }

    with open("datasetfile", "w") as f:
        json.dump(dataset_info, f, indent=4)
    print("✅ Đã tạo datasetfile")

    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT!")
    print("=" * 80)
    print(f"📊 Train: {actual_train_size:,} | Test: {actual_test_size:,}")
    print(f"🎯 Task: '{target_attr}' - Binary Classification")
    print(f"💾 Lưu trữ: HDF5 (tiết kiệm RAM, hỗ trợ lazy loading)")
    print("\n💡 Bước tiếp theo:")
    print("   1. Đổi tên dataloader.py → dataloader_old.py")
    print("   2. Đổi tên dataloader_optimized.py → dataloader.py")
    print("   3. Chạy SISA training như bình thường!")
    print("=" * 80)


if __name__ == "__main__":
    main()
