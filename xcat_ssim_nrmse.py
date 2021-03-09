import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse


def img_to_np_array(path_to_img, dimX, dimY, dimZ):
    f = open(path_to_img, 'rb')
    img_str = f.read()
    img_arr = np.fromstring(img_str, np.float32)
    img = np.reshape(img_arr, (dimX, dimY, dimZ))
    f.close()
    return img


path_to_img1 = 'Data/xcat_ref_image.i33'
path_to_img2 = 'Data/xcat_castor_rec_image.img'

img1 = img_to_np_array(path_to_img1, 100, 100, 100)
img2 = img_to_np_array(path_to_img2, 100, 100, 100)

img1_cropped = img1[8:88, 17:97, 10:90]
img2_cropped = img2[8:88, 17:97, 10:90]

img1_norm = img1 / np.sum(img1_cropped)
img2_norm = img2 / np.sum(img2_cropped)

nrmse_mean = normalized_root_mse(img1_norm, img2_norm, normalization='mean')
print('nrmse_mean: ', nrmse_mean)


ssim, ssim_img = structural_similarity(img1_norm, img2_norm, win_size=11, gaussian_weights='True', sigma=1.5,
                                       use_sample_covariance='False', K1=pow(10, -64), K2=pow(10, -64), full=True)

mssim = np.mean(ssim_img)
print('mssim: ', mssim) # Mean SSIM over the whole image


# Show SSIM map

img_coronal = ssim_img[:, 50, :]
img_transverse = np.flip(ssim_img[32, :, :], 0)
img_sagital = ssim_img[:, :, 50]

color_map = 'viridis'
opts = {'vmin': 0, 'vmax': 1}
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
axes[0].imshow(img_transverse, cmap=color_map, **opts)
axes[0].set_axis_off()
axes[1].imshow(img_coronal, cmap=color_map, **opts)
axes[1].set_axis_off()
im = axes[2].imshow(img_sagital, cmap=color_map, **opts)
axes[2].set_axis_off()
fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.85,
                    wspace=0.05, hspace=0.02)

cb_ax = fig.add_axes([0.88, 0.1, 0.025, 0.81])
cbar = fig.colorbar(im, cax=cb_ax, shrink=0.5)
plt.show()
