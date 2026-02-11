"""
Download CelebA dataset using torchvision (simple and reliable)

This script uses torchvision.datasets.CelebA to download the dataset automatically.
Much simpler and more reliable than manual downloads!
"""

import os
import sys

try:
    import torchvision.transforms as transforms
    from torchvision.datasets import CelebA
except ImportError:
    print("Installing torchvision...")
    os.system(f"{sys.executable} -m pip install torchvision -q")
    import torchvision.transforms as transforms
    from torchvision.datasets import CelebA


def download_celeba(root_dir=".", split="all"):
    """Download CelebA dataset using torchvision
    
    Args:
        root_dir: Directory to store dataset (default: current dir)
        split: 'train', 'valid', 'test', or 'all'
    
    Returns:
        CelebA dataset object
    
    Note:
        First run may ask you to accept CelebA terms of use.
        You'll need to visit a link and copy a token.
    """
    print(f"Downloading CelebA dataset (split={split})...")
    print("This may take 10-30 minutes depending on your internet speed...\n")
    
    dataset = CelebA(
        root=root_dir,
        split=split,
        download=True,
        transform=transforms.ToTensor()
    )
    
    print(f"\n✅ Successfully downloaded {len(dataset)} samples!")
    return dataset


def verify_download(root_dir="."):
    """Verify downloaded files exist"""
    img_dir = os.path.join(root_dir, "celeba", "img_align_celeba")
    attr_file = os.path.join(root_dir, "celeba", "list_attr_celeba.txt")
    
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    
    all_ok = True
    
    if os.path.exists(img_dir):
        num_imgs = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        print(f"✅ Images directory: {img_dir}")
        print(f"   Found {num_imgs} images")
    else:
        print(f"❌ Images directory not found: {img_dir}")
        all_ok = False
    
    if os.path.exists(attr_file):
        size_mb = os.path.getsize(attr_file) / (1024 * 1024)
        print(f"✅ Attributes file: {attr_file}")
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"❌ Attributes file not found: {attr_file}")
        all_ok = False
    
    return all_ok


def main():
    print("="*60)
    print("CELEBA DOWNLOAD SCRIPT (using torchvision)")
    print("="*60)
    
    root_dir = "."
    
    try:
        # Download dataset
        dataset = download_celeba(root_dir, split="all")
        
        # Verify
        if verify_download(root_dir):
            print("\n✅ Download completed successfully!")
            print("\nNext steps:")
            print("1. cd datasets/test")
            print("2. python ../celebA/prepare_data.py")
            return 0
        else:
            print("\n❌ Download verification failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        print("\nTroubleshooting:")
        print("- Make sure you have ~15GB free space")
        print("- Check your internet connection")
        print("- Try running the script again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
