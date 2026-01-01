import cv2
import os
import numpy as np
import random # For picking random images to display

# --- Helper Functions (yolo_to_pixels, load_yolo_annotations - unchanged) ---
def yolo_to_pixels(bbox_normalized, img_width, img_height):
    """
    Converts a YOLO format bounding box (normalized) to pixel coordinates (x_min, y_min, x_max, y_max).
    YOLO format: [class_id, x_center, y_center, width, height] (normalized)
    Output: [x_min_px, y_min_px, x_max_px, y_max_px, class_id] (pixel coordinates)
    """
    class_id, x_center, y_center, width, height = bbox_normalized

    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    x_min_px = int(x_center_px - (width_px / 2))
    y_min_px = int(y_center_px - (height_px / 2))
    x_max_px = int(x_center_px + (width_px / 2))
    y_max_px = int(y_center_px + (height_px / 2))

    return [x_min_px, y_min_px, x_max_px, y_max_px, int(class_id)]

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

# --- Modified Function for Visualization ---
def visualize_yolo_annotations(image_path, annotation_path, class_names=None, window_name="Annotated Image"):
    """
    Loads an image and its YOLO annotations, draws bounding boxes, and displays it.

    Args:
        image_path (str): Path to the image file.
        annotation_path (str): Path to the corresponding YOLO annotation .txt file.
        class_names (list, optional): A list of class names where index corresponds to class_id.
                                     E.g., ['pavement', 'obstacle'].
        window_name (str): The name of the OpenCV window to display the image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Skipping visualization.")
        return

    img_height, img_width, _ = img.shape
    bboxes_yolo = load_yolo_annotations(annotation_path)

    if not bboxes_yolo:
        print(f"No annotations found for {os.path.basename(image_path)}. Displaying image only.")
        cv2.imshow(window_name, img)
        cv2.waitKey(0) # Keep waitKey here to pause for each image
        # cv2.destroyWindow(window_name) # <<< REMOVED THIS LINE
        return

    for bbox_yolo_normalized in bboxes_yolo:
        class_id_float, x_center, y_center, width, height = bbox_yolo_normalized
        class_id = int(class_id_float) # Ensure class_id is an integer

        # Convert normalized YOLO bbox to pixel coordinates
        x_min, y_min, x_max, y_max, _ = yolo_to_pixels([class_id, x_center, y_center, width, height], img_width, img_height)

        # Draw rectangle
        color = (0, 255, 0) # Green color for bounding boxes
        thickness = 2
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Put text label
        label = f"Class {class_id}"
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        
        # Calculate text size to adjust position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw text background rectangle for better readability
        cv2.rectangle(img, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
        cv2.putText(img, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black text


    cv2.imshow(window_name, img)
    print(f"Displaying: {os.path.basename(image_path)}")
    cv2.waitKey(0) # Keep waitKey here to pause for each image
    # cv2.destroyWindow(window_name) # <<< REMOVED THIS LINE


# --- Main Visualization Script ---
if __name__ == "__main__":
    # --- IMPORTANT: Configure your directories here ---
    input_base_dir = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\data' # The base directory containing your 'images' and 'annotations' folders
    output_base_dir = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\data\augmented_dataset_annotations' # The base directory where augmented data was saved

    original_images_dir = os.path.join(input_base_dir, 'images')
    original_annotations_dir = os.path.join(input_base_dir, 'annotations')

    augmented_images_dir = os.path.join(output_base_dir, 'images')
    augmented_annotations_dir = os.path.join(output_base_dir, 'annotations')

    # Define your class names (replace with your actual class names)
    # E.g., class_id 0 might be 'pavement', class_id 1 might be 'obstacle'
    my_class_names = ['pavement', 'road'] # Example classes

    # --- Visualize a few original images ---
    print("\n--- Visualizing Original Images and Annotations ---")
    original_image_files = [f for f in os.listdir(original_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Pick a few random images to display
    num_to_display = min(5, len(original_image_files)) # Display up to 5 original images
    if num_to_display > 0:
        images_to_show = random.sample(original_image_files, num_to_display)
        for img_filename in images_to_show:
            base_name, _ = os.path.splitext(img_filename)
            image_path = os.path.join(original_images_dir, img_filename)
            annotation_path = os.path.join(original_annotations_dir, base_name + '.txt')
            
            visualize_yolo_annotations(image_path, annotation_path, my_class_names, f"Original: {img_filename}")
    else:
        print("No original images to display.")

    # --- Visualize a few augmented images ---
    print("\n--- Visualizing Augmented Images and Annotations ---")
    augmented_image_files = [f for f in os.listdir(augmented_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Pick a few random augmented images to display
    num_to_display_aug = min(5, len(augmented_image_files)) # Display up to 5 augmented images
    if num_to_display_aug > 0:
        images_to_show_aug = random.sample(augmented_image_files, num_to_display_aug)
        for img_filename in images_to_show_aug:
            base_name, _ = os.path.splitext(img_filename)
            image_path = os.path.join(augmented_images_dir, img_filename)
            annotation_path = os.path.join(augmented_annotations_dir, base_name + '.txt')
            
            visualize_yolo_annotations(image_path, annotation_path, my_class_names, f"Augmented: {img_filename}")
    else:
        print("No augmented images to display.")


    # --- FINAL CLEANUP OF ALL WINDOWS ---
    print("\nClosing all visualization windows...")
    cv2.destroyAllWindows()
    print("Visualization complete.")