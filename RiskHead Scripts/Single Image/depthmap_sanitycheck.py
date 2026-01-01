import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Output/plain_inference/images/frame_000069.png")
depth = np.load("Output/plain_inference/depth_maps/frame_000069.npy")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("RGB")
plt.imshow(img[:,:,::-1])
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Depth")
plt.imshow(depth, cmap="inferno")
plt.axis("off")

plt.show()
