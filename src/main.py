import numpy as np
import cv2
import os
from resize import resize_folder  # Imported but not used; you can call resize_folder() if needed before processing

resize_folder()

prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_folder = 'images/'

# Load the model and set up points once (outside the loop)
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)  # cluster center point
points = points.transpose().reshape(2, 313, 1, 1)  # 1 by 1 convolutional kernel
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Get list of image files in the folder
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

if not image_files:
    print("No image files found in the 'images/' folder.")
else:
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Load the image
        bw_image = cv2.imread(image_path)
        if bw_image is None:
            print(f"Could not read image: {image_path}")
            continue
        
        # Normalize and convert to LAB
        normalized = bw_image.astype("float32") / 255.0
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        
        # Resize to 224x224 for the model
        resized = cv2.resize(lab, (224, 224))
        
        # Split the lightness channel
        L = cv2.split(resized)[0]
        L -= 50
        
        # Predict color channels
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # color channels
        
        # Resize back to original size
        ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
        L = cv2.split(lab)[0]
        
        # Concatenate and convert back to BGR
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = (255.0 * colorized).astype("uint8")
        
        # Display the pair
        cv2.imshow("BW image", bw_image)
        cv2.imshow("Colorized", colorized)
        print(f"Displaying: {image_file}. Press any key to continue to the next image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# LAB COLOR SCHEME -> L = LIGHTNESS A* B*
