import cv2
import numpy as np


def addfog_channels(image, fog_intensity=0.5, fog_color_intensity=255):
    """
    对图像 RGB 通道应用雾效。
    参数:
        image: 输入图像（numpy数组）。
        fog_intensity: 雾的强度（0到1）。
        fog_color_intensity: 雾的颜色强度（0到255）.不宜过小, 建议大于180
    """
    fog_intensity = np.clip(fog_intensity, 0, 1)
    fog_layer = np.ones_like(image) * fog_color_intensity
    fogged_image = cv2.addWeighted(image, 1 - fog_intensity, fog_layer, fog_intensity, 0)

    return fogged_image

if __name__ == '__main__':
    exr_file_path = "output/haze_images/DJI_20240308113002_0205_V.JPG"

    image = cv2.imread(exr_file_path)
    (row, col, chs) = image.shape
    fog_intensity = 0.5
    fogged_image = addfog_channels(image, fog_intensity)

    cv2.imwrite(f"output/haze_images/{exr_file_path.split('/')[-1].split('.')[0]}_hazed.jpg", fogged_image)
    img = cv2.resize(image, (int(col / 4), int(row / 4)))
    cv2.imwrite(f"output/haze_images/{exr_file_path.split('/')[-1].split('.')[0]}_resized.jpg", img)
