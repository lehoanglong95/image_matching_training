import cv2
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

def on_press(event):
    if event.key == "x":
        plt.close("all")

if __name__ == '__main__':
    df = pd.read_parquet("./input_file/very_small_val_set_pair.parquet")[["first_image_file_name", "second_image_file_name", "pair_result"]]
    base_dir = "/Users/lap02387/pms/image_matching/images"
    for idx, row in df.iterrows():
        fig, ax = plt.subplots(1, 2)
        fig.canvas.mpl_connect("key_press_event", on_press)
        first_file_name = row["first_image_file_name"]
        second_file_name = row["second_image_file_name"]
        img_1 = io.imread(f"{base_dir}/{first_file_name}")
        img_2 = io.imread(f"{base_dir}/{second_file_name}")
        ax[0].imshow(img_1)
        ax[1].imshow(img_2)
        ax[0].set_title(row["pair_result"])
        plt.show()


