import os
import cv2

def resize_folder():
    image_folder = "images"
    target_size = (640, 640)

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            file_path = os.path.join(image_folder, filename)
            temp_path = os.path.join(image_folder, f"temp_{filename}")

        try:
            # Read the image
            img = cv2.imread(file_path)

            if img is None:
                print(f"Could not read image: {filename}")
                continue

            resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Save to temp files first
            cv2.imwrite(temp_path, resized)

            # Replace old file with the resized one
            os.remove(file_path)
            os.rename(temp_path, file_path)
            print(f"resized and replaced image: {filename}")

        except Exception as e:
            print(f"Could not resize image {filename}: {str(e)}")