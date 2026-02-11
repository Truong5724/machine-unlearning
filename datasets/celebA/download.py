"""
Script tự động tải CelebA dataset
Hỗ trợ tải từ Google Drive (chính thức) hoặc Kaggle

CelebA dataset chứa:
- img_align_celeba.zip: ~1.3GB, 202,599 ảnh
- list_attr_celeba.txt: 40 thuộc tính nhị phân cho mỗi ảnh
- identity_CelebA.txt: nhãn identity
- list_bbox_celeba.txt: bounding boxes
- list_landmarks_align_celeba.txt: facial landmarks
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

def check_dependencies():
    """Kiểm tra và cài đặt dependencies cần thiết"""
    missing = []
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        import gdown
    except ImportError:
        missing.append("gdown")
    
    if missing:
        print(f"⚠️  Thiếu các thư viện: {', '.join(missing)}")
        print("Đang cài đặt...")
        import subprocess
        for lib in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print("✅ Đã cài đặt xong!\n")

def download_with_gdown(file_id, output_path):
    """Tải file từ Google Drive sử dụng gdown"""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Đang tải {output_path} từ Google Drive...")
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tải với gdown: {e}")
        return False

def download_with_requests(url, output_path, chunk_size=8192):
    """Tải file sử dụng requests (fallback)"""
    try:
        import requests
        print(f"Đang tải {output_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rTiến độ: {percent:.1f}%", end='', flush=True)
        print()  # New line
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tải với requests: {e}")
        return False

def extract_zip(zip_path, extract_to=None):
    """Giải nén file zip"""
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    print(f"Đang giải nén {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Đã giải nén xong!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi giải nén: {e}")
        return False

def verify_files():
    """Kiểm tra các file đã tải về"""
    required_files = {
        "img_align_celeba": "Thư mục ảnh",
        "list_attr_celeba.txt": "File thuộc tính",
        "identity_CelebA.txt": "File identity",
        "list_bbox_celeba.txt": "File bounding box",
        "list_landmarks_align_celeba.txt": "File landmarks"
    }
    
    all_ok = True
    for file_path, desc in required_files.items():
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                num_files = len([f for f in os.listdir(file_path) if f.endswith('.jpg')])
                print(f"✅ {desc}: {file_path} ({num_files} ảnh)")
            else:
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"✅ {desc}: {file_path} ({size:.2f} MB)")
        else:
            print(f"❌ Thiếu: {file_path} ({desc})")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("SCRIPT TẢI CELEBA DATASET")
    print("=" * 60)
    
    # Kiểm tra dependencies
    check_dependencies()
    
    # Google Drive file IDs (official CelebA dataset)
    # Lưu ý: Các ID này có thể cần cập nhật nếu link thay đổi
    files_to_download = {
        "img_align_celeba.zip": {
            "gdrive_id": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",  # img_align_celeba.zip
            "description": "Ảnh CelebA (202,599 ảnh, ~1.3GB)"
        },
        "list_attr_celeba.txt": {
            "gdrive_id": "0B7EVK8r0v71pblRyaVFSWGxPY0U",  # list_attr_celeba.txt
            "description": "File thuộc tính"
        },
        "identity_CelebA.txt": {
            "gdrive_id": "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",  # identity_CelebA.txt
            "description": "File identity"
        },
        "list_bbox_celeba.txt": {
            "gdrive_id": "0B7EVK8r0v71pbThiMUVxZ2ZPYVk",  # list_bbox_celeba.txt
            "description": "File bounding box"
        },
        "list_landmarks_align_celeba.txt": {
            "gdrive_id": "0B7EVK8r0v71pd0FJY3Blby1HUTQ",  # list_landmarks_align_celeba.txt
            "description": "File landmarks"
        }
    }
    
    # Tải các file còn thiếu
    for filename, info in files_to_download.items():
        if os.path.exists(filename) or (filename == "img_align_celeba.zip" and os.path.exists("img_align_celeba")):
            print(f"✅ {filename} đã tồn tại, bỏ qua...")
            continue
        
        print(f"\n📥 Đang tải {info['description']}...")
        success = download_with_gdown(info['gdrive_id'], filename)
        
        if not success:
            print(f"⚠️  Không thể tải {filename} từ Google Drive")
            print(f"   Vui lòng tải thủ công từ: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
            continue
    
    # Giải nén img_align_celeba.zip nếu cần
    if os.path.exists("img_align_celeba.zip") and not os.path.exists("img_align_celeba"):
        extract_zip("img_align_celeba.zip")
        # Xóa file zip sau khi giải nén để tiết kiệm dung lượng (tùy chọn)
        # os.remove("img_align_celeba.zip")
    
    # Kiểm tra lại
    print("\n" + "=" * 60)
    print("KIỂM TRA CÁC FILE ĐÃ TẢI:")
    print("=" * 60)
    all_ok = verify_files()
    
    if all_ok:
        print("\n✅ Tất cả các file đã sẵn sàng!")
        print("Bạn có thể chạy: python prepare_data.py")
    else:
        print("\n⚠️  Một số file còn thiếu. Vui lòng tải thủ công từ:")
        print("   https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
        print("   hoặc")
        print("   https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")

if __name__ == "__main__":
    main()
