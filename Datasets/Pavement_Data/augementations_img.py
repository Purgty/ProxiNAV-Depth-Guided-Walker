import cv2
import os
import albumentations as A
from tqdm import tqdm
import numpy as np

def yolo_to_albumentations(bbox, img_width, img_height):
    """
    Converts a YOLO format bounding box to Albumentations format (normalized x_min, y_min, x_max, y_max).
    YOLO format: [class_id, x_center, y_center, width, height] (normalized)
    Albumentations format: [x_min, y_min, x_max, y_max, class_id] (normalized)
    """
    class_id, x_center, y_center, width, height = bbox

    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)

    return [x_min, y_min, x_max, y_max, class_id]

def albumentations_to_yolo(bbox, img_width, img_height):
    """
    Converts an Albumentations format bounding box (normalized x_min, y_min, x_max, y_max) to YOLO format.
    Albumentations format: [x_min, y_min, x_max, y_max, class_id] (normalized)
    YOLO format: [class_id, x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, x_max, y_max, class_id = bbox

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Clip values to be within [0, 1] range to avoid issues with transformations
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)

    return [int(class_id), x_center, y_center, width, height]


def load_yolo_annotations(annotation_path):
    """Loads YOLO annotations from a .txt file."""
    bboxes = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split(' ')))
                # YOLO format: class_id x_center y_center width height
                bboxes.append([int(parts[0]), parts[1], parts[2], parts[3], parts[4]])
    return bboxes

def save_yolo_annotations(annotation_path, bboxes):
    """Saves YOLO annotations to a .txt file."""
    with open(annotation_path, 'w') as f:
        for bbox in bboxes:
            # bbox is [class_id, x_center, y_center, width, height]
            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")


def apply_augmentations_to_dataset(
    input_base_dir, # e.g., C:\
    output_base_dir, # e.g., C:\augmented_dataset
    num_augmentations_per_image=5
):
    """
    Applies augmentations to images and their corresponding YOLO annotations.

    Args:
        input_base_dir (str): The base directory containing 'images' and 'annotations' folders.
        output_base_dir (str): The base directory where augmented images and annotations will be saved.
        num_augmentations_per_image (int): Number of augmented versions to create for each original image.
    """
    input_images_dir = os.path.join(input_base_dir, 'images')
    input_annotations_dir = os.path.join(input_base_dir, 'annotations')

    output_images_dir = os.path.join(output_base_dir, 'images')
    output_annotations_dir = os.path.join(output_base_dir, 'annotations')

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # Define the augmentation pipeline including bbox_params
    transform = A.Compose([
        # Lighting Augmentations (stronger emphasis)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

        # Shear Augmentation (using Affine transform)
        A.Affine(shear=(-10, 10), interpolation=cv2.INTER_LINEAR, p=0.5, fit_output=False), # fit_output=False keeps original image size

        # Other Augmentations (in smaller quantities/probabilities)
        A.HorizontalFlip(p=0.25),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT, # Use constant border mode to fill empty areas
            value=0, # Fill with black pixels
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.15),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'])) # Specify YOLO format and label_fields

    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print(f"No image files found in {input_images_dir}.")
        return

    print(f"Found {len(image_files)} images in {input_images_dir}. Augmenting...")

    for image_filename in tqdm(image_files, desc="Augmenting Dataset"):
        base_name, ext = os.path.splitext(image_filename)
        image_path = os.path.join(input_images_dir, image_filename)
        annotation_path = os.path.join(input_annotations_dir, base_name + '.txt')

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Albumentations expects RGB

        # Load YOLO annotations for the current image
        # Each bbox is [class_id, x_center, y_center, width, height] (normalized)
        bboxes_yolo_raw = load_yolo_annotations(annotation_path)
        
        # Albumentations expects labels to be separate or as part of bbox tuple
        # We'll pass class_ids separately
        class_ids = [bbox[0] for bbox in bboxes_yolo_raw]
        # Albumentations BboxParams(format='yolo') expects [x_c, y_c, w, h] only
        bboxes_for_aug = [bbox[1:] for bbox in bboxes_yolo_raw]


        for i in range(num_augmentations_per_image):
            # Apply transformations
            augmented = transform(image=img, bboxes=bboxes_for_aug, class_ids=class_ids)
            augmented_image = augmented['image']
            augmented_bboxes_yolo = augmented['bboxes'] # These are already in YOLO format if BboxParams.format='yolo'
            augmented_class_ids = augmented['class_ids']

            # Filter out any bounding boxes that might have become invalid (e.g., too small, out of bounds)
            # Albumentations might return boxes with invalid coordinates or zero area after transformation.
            valid_bboxes = []
            for bbox_yolo, class_id in zip(augmented_bboxes_yolo, augmented_class_ids):
                xc, yc, w, h = bbox_yolo
                # Basic validation: check if width/height are reasonable and within bounds
                if w > 0.001 and h > 0.001 and 0 <= xc <= 1 and 0 <= yc <= 1:
                    valid_bboxes.append([class_id, xc, yc, w, h])

            # Save augmented image
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            output_image_filename = f"{base_name}_aug{i}{ext}"
            output_image_path = os.path.join(output_images_dir, output_image_filename)
            cv2.imwrite(output_image_path, augmented_image_bgr)

            # Save augmented annotations
            output_annotation_filename = f"{base_name}_aug{i}.txt"
            output_annotation_path = os.path.join(output_annotations_dir, output_annotation_filename)
            save_yolo_annotations(output_annotation_path, valid_bboxes)

    print(f"Augmentation complete. Augmented dataset saved to {output_base_dir}")

if __name__ == "__main__":
    # --- IMPORTANT: Configure your directories here ---
    # Input base directory (e.g., where C:\images and C:\annotations are located)
    input_base_dir = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\data' 
    
    # Output base directory for the augmented dataset
    output_base_dir = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\data\augmented_dataset_annotations' 

    # Number of augmented versions to create for each original image
    num_aug_per_img = 5 

    apply_augmentations_to_dataset(
        input_base_dir,
        output_base_dir,
        num_aug_per_img
    )