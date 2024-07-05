#%%
# from PIL import Image
import numpy as np

# Load and check the original label image
original_label_path = "../dataloader_test/v2/small_800-400_R07/labels/train/lab_train_0_1_bird_view_frame_SEG.png"
original_label = Image.open(original_label_path)
original_label = np.array(original_label)

og_img_path = "../dataloader_test/v2/small_800-400_R07/imgs/train/train_0_0_bird_view_frame_RGB.jpg"
img = Image.open(og_img_path)
img = np.array(img)


print(img.shape)
print(original_label.shape)
# Check the mode of the label image
#%%

import numpy as np
import matplotlib.pyplot as plt

img = np.load("bad_img.npy")
img = img.squeeze()
plt.imshow(img)
plt.show()
