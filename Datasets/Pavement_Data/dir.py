import os
import random
import shutil
import yaml
from tqdm import tqdm

def create_yolo_split(
    input_base_dir, # e.g., C:\augmented_dataset
    output_yolo_dataset_dir, # e.g., C:\yolo_dataset_split
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    class_names=None # List of your class names, e.g., ['pavement', 'obstacle']
):
    """
    Splits a dataset (images and YOLO annotations) into train, val, and test sets
    and organizes them in YOLO format.

    Args:
        input_base_dir (str): Path to the base directory of your combined dataset,
                              expected to contain 'images' and 'annotations' subfolders.
        output_yolo_dataset_dir (str): Path to the desired output directory for the YOLO split.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        class_names (list): A list of class names in order of their IDs. Required for dataset.yaml.
    """

    if not (abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    if class_names is None:
        raise ValueError("Class names must be provided for the dataset.yaml file.")

    input_images_dir = os.path.join(input_base_dir, 'images')
    input_annotations_dir = os.path.join(input_base_dir, 'annotations') # Use 'annotations' as per your structure

    if not os.path.exists(input_images_dir):
        print(f"Error: Input images directory not found at {input_images_dir}")
        return
    if not os.path.exists(input_annotations_dir):
        print(f"Error: Input annotations directory not found at {input_annotations_dir}")
        return

    # Create output directories
    os.makedirs(output_yolo_dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dataset_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dataset_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dataset_dir, 'labels', 'test'), exist_ok=True)

    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    random.shuffle(image_files) # Shuffle to ensure random distribution

    total_files = len(image_files)
    if total_files == 0:
        print(f"No image files found in {input_images_dir}. Exiting.")
        return

    num_train = int(total_files * train_ratio)
    num_val = int(total_files * val_ratio)
    num_test = total_files - num_train - num_val # Remaining files go to test to handle rounding

    train_files = image_files[:num_train]
    val_files = image_files[num_train : num_train + num_val]
    test_files = image_files[num_train + num_val :]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    print(f"Total files: {total_files}")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    # Copy files to their respective directories
    for split_name, file_list in splits.items():
        print(f"\nCopying {split_name} files...")
        for img_filename in tqdm(file_list, desc=f"Copying {split_name} images/labels"):
            base_name, _ = os.path.splitext(img_filename)
            
            # Source paths
            src_image_path = os.path.join(input_images_dir, img_filename)
            src_annotation_path = os.path.join(input_annotations_dir, base_name + '.txt')

            # Destination paths
            dest_image_path = os.path.join(output_yolo_dataset_dir, 'images', split_name, img_filename)
            dest_annotation_path = os.path.join(output_yolo_dataset_dir, 'labels', split_name, base_name + '.txt')

            # Copy image
            shutil.copyfile(src_image_path, dest_image_path)
            
            # Copy annotation if it exists
            if os.path.exists(src_annotation_path):
                shutil.copyfile(src_annotation_path, dest_annotation_path)
            else:
                print(f"Warning: Annotation file not found for {img_filename} at {src_annotation_path}. Skipping.")


    # Create dataset.yaml file
    dataset_yaml_content = {
        'path': os.path.abspath(output_yolo_dataset_dir), # Absolute path to the dataset root
        'train': '../images/train',
        'val': '../images/val',
        'test': '../images/test', # Optional, useful for final evaluation
        'nc': len(class_names),
        'names': class_names
    }

    yaml_file_path = os.path.join(output_yolo_dataset_dir, 'dataset.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(dataset_yaml_content, f, default_flow_style=False)

    print(f"\nDataset split complete!")
    print(f"YOLO format dataset created at: {output_yolo_dataset_dir}")
    print(f"Configuration file generated: {yaml_file_path}")
    print("\nNext steps: You can now point your YOLO training script to this dataset.yaml file.")


if __name__ == "__main__":
    # --- IMPORTANT: Configure your directories and classes here ---
    
    # This should be the base directory where your *combined* images and annotations are.
    # If you followed the previous step, this would be your 'augmented_dataset' folder.
    input_dataset_base = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\data\augmented_dataset_annotations' 
    
    # This will be the new directory where the YOLO-formatted split dataset will be created.
    output_yolo_dataset_base = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\data\augmented_yolo_dataset'

    # !!! IMPORTANT !!! Define your class names in the correct order of their IDs
    # E.g., if class ID 0 is 'sidewalk' and ID 1 is 'road' based on your annotation files
    my_class_names = ['pavement', 'road'] # <<< REPLACE WITH YOUR ACTUAL CLASS NAMES

    # Define split ratios
    train_ratio = 0.8
    val_ratio = 0.15 # Using 15% for validation
    test_ratio = 0.05 # Using 5% for testing (adjust as needed)

    create_yolo_split(
        input_dataset_base,
        output_yolo_dataset_base,
        train_ratio,
        val_ratio,
        test_ratio,
        my_class_names
    )