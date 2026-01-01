import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread(r'Output2\plain_inference\images\image_000001.png')
O = np.load(r'Output2\plain_inference\obstacle_maps\image_000001.npy')
S = np.load(r'Output2\plain_inference\walkability_maps\image_000001.npy')

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("RGB"); plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2); plt.title("Obstacle"); plt.imshow(O, cmap="hot")
plt.subplot(1,3,3); plt.title("Walkable"); plt.imshow(S, cmap="gray")
plt.show()