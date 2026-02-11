"""
Chuẩn bị CelebA dataset cho SISA training
- Tải và xử lý ảnh và thuộc tính
- Tạo train/test split (chuẩn: 162,770 train, còn lại test)
- Lưu dưới dạng .npy để load nhanh

CelebA có 40 thuộc tính nhị phân. Script này cho phép chọn thuộc tính để train.
Các thuộc tính phổ biến và tốt cho binary classification:
- Smiling: Phổ biến nhất, cân bằng tốt (~50-50)
- Male: Cân bằng tốt
- Young: Cân bằng tốt
- Attractive: Có thể hơi lệch
- Eyeglasses: Ít dữ liệu positive
- Blond_Hair: Cân bằng tương đối
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    # Fallback nếu không có tqdm
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


def load_images(img_dir, filenames, target_size=(64, 64), max_images=None):
    """Tải và resize ảnh"""
    if max_images:
        filenames = filenames[:max_images]

    images = []
    failed = []
    print(f"Đang tải {len(filenames)} ảnh...")

    for fname in tqdm(filenames, desc="Loading images"):
        img_path = os.path.join(img_dir, fname)
        try:
            if not os.path.exists(img_path):
                failed.append(fname)
                continue
            img = Image.open(img_path).convert('RGB')
            # LANCZOS cho chất lượng tốt hơn
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
            images.append(img_array)
        except Exception as e:
            print(f"\n⚠️  Lỗi khi tải {fname}: {e}")
            failed.append(fname)
            continue

    if failed:
        print(f"\n⚠️  Không tải được {len(failed)} ảnh")

    if len(images) == 0:
        raise ValueError(
            "Không tải được ảnh nào! Kiểm tra lại đường dẫn img_dir.")

    return np.array(images, dtype=np.uint8)


def analyze_attributes(attributes, attr_names):
    """Phân tích và hiển thị phân bố của các thuộc tính"""
    print("\n" + "=" * 80)
    print("PHÂN TÍCH CÁC THUỘC TÍNH CELEBA (40 thuộc tính)")
    print("=" * 80)
    print(f"{'Thuộc tính':<20} {'Class 0':<12} {'Class 1':<12} {'Tỷ lệ 1':<10} {'Đánh giá'}")
    print("-" * 80)

    # Các thuộc tính được đề xuất (cân bằng tốt, dễ train)
    recommended = ["Smiling", "Male", "Young", "Attractive", "Blond_Hair",
                   "Wearing_Hat", "Wearing_Necktie", "High_Cheekbones"]

    attr_stats = []
    for i, attr_name in enumerate(attr_names):
        labels = attributes[:, i]
        count_0 = np.sum(labels == 0)
        count_1 = np.sum(labels == 1)
        ratio_1 = count_1 / len(labels) * 100

        # Đánh giá: tốt nếu tỷ lệ 1 trong khoảng 30-70%
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

    # Sắp xếp theo tỷ lệ gần 50% nhất (cân bằng nhất)
    attr_stats.sort(key=lambda x: abs(x['ratio_1'] - 50))

    # Hiển thị top 15 thuộc tính cân bằng nhất
    print("\n📊 TOP 15 THUỘC TÍNH CÂN BẰNG NHẤT (tỷ lệ gần 50-50):")
    for stat in attr_stats[:15]:
        print(f"{stat['name']:<20} {stat['count_0']:<12} {stat['count_1']:<12} "
              f"{stat['ratio_1']:>5.1f}%    {stat['rating']}")

    print("\n" + "=" * 80)
    return attr_stats


def main():
    parser = argparse.ArgumentParser(
        description='Chuẩn bị CelebA dataset cho training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python prepare_data.py                    # Hiển thị tất cả thuộc tính và chọn tương tác
  python prepare_data.py --attribute Smiling # Chọn thuộc tính "Smiling"
  python prepare_data.py --attribute Male   # Chọn thuộc tính "Male"
  python prepare_data.py --list             # Chỉ liệt kê thuộc tính, không prepare
        """
    )
    parser.add_argument(
        '--attribute', '-a',
        type=str,
        default=None,
        help='Tên thuộc tính để sử dụng (ví dụ: Smiling, Male, Young). Nếu không chỉ định, sẽ hiển thị menu chọn.'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='Chỉ liệt kê các thuộc tính và phân bố, không prepare dataset'
    )
    args = parser.parse_args()

    # Cấu hình
    img_dir = "img_align_celeba"
    attr_file = "list_attr_celeba.txt"
    target_size = (64, 64)  # Resize về 64x64 để train nhanh hơn

    # Kiểm tra file cần thiết
    if not os.path.exists(attr_file):
        raise FileNotFoundError(
            f"Không tìm thấy {attr_file}. Hãy chạy download.py trước!")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(
            f"Không tìm thấy {img_dir}. Hãy chạy download.py trước!")

    # CelebA split chuẩn: 162,770 train, 19,867 val, 19,962 test
    # Chúng ta dùng: 162,770 train, 39,829 test (gộp val+test)
    train_size = 162770

    print("=" * 80)
    print("CHUẨN BỊ CELEBA DATASET CHO MACHINE UNLEARNING")
    print("=" * 80)

    print("\nBước 1: Đang tải thuộc tính...")
    filenames, attributes, attr_names = load_attributes(attr_file)
    print(f"✅ Đã tải {len(filenames)} mẫu với {len(attr_names)} thuộc tính")

    # Phân tích các thuộc tính
    attr_stats = analyze_attributes(attributes, attr_names)

    # Chọn thuộc tính
    if args.list:
        print("\n✅ Đã liệt kê tất cả thuộc tính. Sử dụng --attribute để chọn thuộc tính cụ thể.")
        return

    if args.attribute:
        target_attr = args.attribute
    else:
        # Hiển thị menu chọn
        print("\n" + "=" * 80)
        print("CHỌN THUỘC TÍNH ĐỂ TRAIN")
        print("=" * 80)
        print("\nCác thuộc tính được đề xuất (cân bằng tốt):")
        recommended = ["Smiling", "Male", "Young", "Attractive", "Blond_Hair"]
        for i, attr in enumerate(recommended, 1):
            if attr in attr_names:
                idx = attr_names.index(attr)
                labels = attributes[:, idx]
                ratio = np.sum(labels == 1) / len(labels) * 100
                print(f"  {i}. {attr:<20} (Tỷ lệ positive: {ratio:.1f}%)")

        print("\nHoặc nhập tên thuộc tính khác từ danh sách trên.")
        print("Nhấn Enter để dùng mặc định: Smiling")
        user_input = input("\nChọn thuộc tính (số hoặc tên): ").strip()

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(recommended):
                target_attr = recommended[idx]
            else:
                print("⚠️  Số không hợp lệ, dùng mặc định: Smiling")
                target_attr = "Smiling"
        elif user_input:
            target_attr = user_input
        else:
            target_attr = "Smiling"

    # Kiểm tra thuộc tính có tồn tại không
    if target_attr not in attr_names:
        print(f"\n❌ Không tìm thấy thuộc tính '{target_attr}'")
        print(f"   Các thuộc tính có sẵn: {', '.join(attr_names[:10])}...")
        print(f"   Sử dụng 'Smiling' làm mặc định")
        target_attr = "Smiling"

    attr_idx = attr_names.index(target_attr)
    labels = attributes[:, attr_idx]

    # Hiển thị thống kê thuộc tính đã chọn
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    ratio_1 = count_1 / len(labels) * 100

    print("\n" + "=" * 80)
    print(f"ĐÃ CHỌN THUỘC TÍNH: '{target_attr}'")
    print("=" * 80)
    print(f"   Class 0 (Negative): {count_0:,} ({100-ratio_1:.1f}%)")
    print(f"   Class 1 (Positive): {count_1:,} ({ratio_1:.1f}%)")
    if 30 <= ratio_1 <= 70:
        print(f"   ✅ Phân bố cân bằng tốt, phù hợp cho binary classification")
    elif 20 <= ratio_1 < 30 or 70 < ratio_1 <= 80:
        print(f"   ⚠️  Phân bố hơi lệch, vẫn có thể train được")
    else:
        print(f"   ❌ Phân bố rất lệch, có thể ảnh hưởng đến chất lượng model")
    print("=" * 80)

    # Chia train/test
    train_filenames = filenames[:train_size]
    test_filenames = filenames[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    # Tải ảnh train (có thể mất 10-20 phút)
    print("\nBước 3: Đang tải ảnh training...")
    if not os.path.exists('celeba_train.npy'):
        train_images = load_images(img_dir, train_filenames, target_size)
        print(f"   Shape ảnh train: {train_images.shape}")
        np.save('celeba_train.npy', {'X': train_images, 'y': train_labels})
        print("✅ Đã lưu celeba_train.npy")
    else:
        print("✅ celeba_train.npy đã tồn tại, bỏ qua...")

    # Tải ảnh test
    print("\nBước 4: Đang tải ảnh test...")
    if not os.path.exists('celeba_test.npy'):
        test_images = load_images(img_dir, test_filenames, target_size)
        print(f"   Shape ảnh test: {test_images.shape}")
        np.save('celeba_test.npy', {'X': test_images, 'y': test_labels})
        print("✅ Đã lưu celeba_test.npy")
    else:
        print("✅ celeba_test.npy đã tồn tại, bỏ qua...")

    # Tạo datasetfile
    if not os.path.exists("datasetfile"):
        # Đọc lại để lấy số lượng chính xác
        if os.path.exists('celeba_train.npy'):
            train_data = np.load('celeba_train.npy', allow_pickle=True).item()
            actual_train_size = len(train_data['X'])
        else:
            actual_train_size = train_size

        if os.path.exists('celeba_test.npy'):
            test_data = np.load('celeba_test.npy', allow_pickle=True).item()
            actual_test_size = len(test_data['X'])
        else:
            actual_test_size = len(test_filenames)

        dataset_info = {
            "nb_train": actual_train_size,
            "nb_test": actual_test_size,
            "input_shape": [3, target_size[0], target_size[1]],
            "nb_classes": 2,  # Binary classification
            "dataloader": "dataloader",
            "attribute": target_attr,  # Lưu thuộc tính đã chọn
            "attribute_distribution": {
                "class_0": int(count_0),
                "class_1": int(count_1),
                "ratio_1_percent": float(ratio_1)
            }
        }

        with open("datasetfile", "w") as f:
            json.dump(dataset_info, f, indent=4)
        print("\n✅ Đã tạo datasetfile (đã lưu thông tin thuộc tính)")
    else:
        print("\n✅ datasetfile đã tồn tại")
        # Cập nhật thông tin thuộc tính nếu cần
        with open("datasetfile", "r") as f:
            dataset_info = json.load(f)
        if "attribute" not in dataset_info or dataset_info["attribute"] != target_attr:
            dataset_info["attribute"] = target_attr
            dataset_info["attribute_distribution"] = {
                "class_0": int(count_0),
                "class_1": int(count_1),
                "ratio_1_percent": float(ratio_1)
            }
            with open("datasetfile", "w") as f:
                json.dump(dataset_info, f, indent=4)
            print("   (Đã cập nhật thông tin thuộc tính)")

    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT CHUẨN BỊ CELEBA DATASET!")
    print("=" * 80)
    print(f"📊 Dataset:")
    print(f"   Train: {train_size:,} ảnh")
    print(f"   Test: {len(test_filenames):,} ảnh")
    print(f"   Kích thước ảnh: {target_size}")
    print(f"\n🎯 Task: Binary Classification - '{target_attr}'")
    print(f"   Class 0: {count_0:,} ({100-ratio_1:.1f}%)")
    print(f"   Class 1: {count_1:,} ({ratio_1:.1f}%)")
    print("\n💡 Gợi ý cho Machine Unlearning:")
    print(
        f"   - Thuộc tính '{target_attr}' {'cân bằng tốt' if 30 <= ratio_1 <= 70 else 'hơi lệch'}")
    print(f"   - Model binary classification sẽ dễ train và đánh giá")
    print(f"   - Có thể so sánh với CIFAR-10 (multi-class) để đánh giá kỹ thuật unlearning")
    print("=" * 80)


if __name__ == "__main__":
    main()
