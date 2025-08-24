import os
import shutil
import random
import glob
from PIL import Image
import cv2


def setup_directories(base_path):
    """Create necessary directories"""
    dirs = [
        os.path.join(base_path, 'train', 'with_mask'),
        os.path.join(base_path, 'train', 'without_mask'),
        os.path.join(base_path, 'val', 'with_mask'),
        os.path.join(base_path, 'val', 'without_mask')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created successfully")


def clean_and_validate_images(directory):
    """Remove corrupted images and validate file formats"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    removed_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()

            # Check file extension
            if file_ext not in valid_extensions:
                os.remove(file_path)
                removed_count += 1
                continue

            # Try to open and validate image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity
            except:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                removed_count += 1

    print(f"✅ Cleaned dataset: removed {removed_count} invalid/corrupted files")


def move_to_val(source_dir, val_dir, val_ratio=0.2):
    """Move files from train to validation with proper validation"""
    if not os.path.exists(source_dir):
        print(f"❌ Source directory {source_dir} doesn't exist")
        return

    os.makedirs(val_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir)
             if os.path.isfile(os.path.join(source_dir, f))]

    if len(files) == 0:
        print(f"❌ No files found in {source_dir}")
        return

    random.shuffle(files)
    val_count = int(len(files) * val_ratio)

    if val_count == 0:
        print(f"⚠️  Too few files in {source_dir} to create validation split")
        return

    val_files = files[:val_count]
    moved_count = 0

    for f in val_files:
        source_path = os.path.join(source_dir, f)
        dest_path = os.path.join(val_dir, f)
        try:
            shutil.move(source_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {f}: {e}")

    print(f"✅ Moved {moved_count} files from {os.path.basename(source_dir)} to validation")


def check_dataset_stats(base_path):
    """Check and display dataset statistics"""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        for class_name in ['with_mask', 'without_mask']:
            class_dir = os.path.join(base_path, split, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir)
                             if os.path.isfile(os.path.join(class_dir, f))])
                print(f"  {class_name}: {count} images")
            else:
                print(f"  {class_name}: Directory not found")

    # Calculate total and ratios
    train_with = len(glob.glob(os.path.join(base_path, 'train', 'with_mask', '*')))
    train_without = len(glob.glob(os.path.join(base_path, 'train', 'without_mask', '*')))
    val_with = len(glob.glob(os.path.join(base_path, 'val', 'with_mask', '*')))
    val_without = len(glob.glob(os.path.join(base_path, 'val', 'without_mask', '*')))

    total_train = train_with + train_without
    total_val = val_with + val_without
    total_all = total_train + total_val

    print(f"\nTOTAL IMAGES: {total_all}")
    print(
        f"Train/Val Split: {total_train}/{total_val} ({total_train / total_all * 100:.1f}%/{total_val / total_all * 100:.1f}%)")

    if train_with > 0 and train_without > 0:
        ratio = train_with / train_without
        print(f"Class Balance (with/without): {ratio:.2f}")
        if 0.8 <= ratio <= 1.2:
            print("✅ Dataset is well balanced")
        else:
            print("⚠️  Dataset imbalance detected - consider balancing")


def resize_images(directory, target_size=(224, 224)):
    """Resize all images to target size"""
    resized_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        resized_img = cv2.resize(img, target_size)
                        cv2.imwrite(file_path, resized_img)
                        resized_count += 1
                except:
                    print(f"Error resizing {file_path}")

    print(f"✅ Resized {resized_count} images to {target_size}")


def main():
    base_path = 'images/img'  # Updated path to match your structure

    print("🚀 Starting dataset preparation...")

    # Step 1: Setup directories
    setup_directories(base_path)

    # Step 2: Clean corrupted images
    print("\n📋 Cleaning dataset...")
    clean_and_validate_images(os.path.join(base_path, 'train'))

    # Step 3: Create train/val split
    print("\n📂 Creating train/validation split...")
    classes = ['with_mask', 'without_mask']

    for c in classes:
        source = os.path.join(base_path, 'train', c)
        val = os.path.join(base_path, 'val', c)
        move_to_val(source, val, val_ratio=0.2)

    # Step 4: Resize images (optional)
    print("\n🖼️  Resizing images...")
    resize_images(base_path)

    # Step 5: Show statistics
    check_dataset_stats(base_path)

    print("\n✅ Dataset preparation complete!")
    print("Your dataset is now ready for training!")


if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    main()