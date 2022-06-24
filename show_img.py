import cv2
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

di = defaultdict(set)
labeled = set()

def on_press(event, url_l, index):
    di[url_l[0]].add(url_l[1])
    labeled.add(url_l[0])
    labeled.add(url_l[1])
    if event.key == "x":
        dfdf = pd.DataFrame()
        dfdf["source"] = [url_l[0]]
        dfdf["candidate"] = [list(di[url_l[0]])]
        dfdf.to_parquet(f"input_file/test_set_df_list/test_set_relabel_{index}.parquet")
        plt.close("all")
    if event.key == "0":
        di[url_l[0]].add(url_l[2])
        labeled.add(url_l[2])
    if event.key == "1":
        di[url_l[0]].add(url_l[3])
        labeled.add(url_l[3])
    if event.key == "2":
        di[url_l[0]].add(url_l[4])
        labeled.add(url_l[4])
    if event.key == "3":
        di[url_l[0]].add(url_l[5])
        labeled.add(url_l[5])
    if event.key == "4":
        di[url_l[0]].add(url_l[6])
        labeled.add(url_l[6])
    if event.key == "5":
        di[url_l[0]].add(url_l[7])
        labeled.add(url_l[7])
    if event.key == "6":
        di[url_l[0]].add(url_l[8])
        labeled.add(url_l[8])
    if event.key == "7":
        di[url_l[0]].add(url_l[9])
        labeled.add(url_l[9])
    if event.key == "8":
        di[url_l[0]].add(url_l[10])
        labeled.add(url_l[10])
    if event.key == "9":
        di[url_l[0]].add(url_l[11])
        labeled.add(url_l[11])
    if event.key == "a": #10
        di[url_l[0]].add(url_l[12])
        labeled.add(url_l[12])
    if event.key == "s": #11
        di[url_l[0]].add(url_l[13])
        labeled.add(url_l[13])
    if event.key == "d": #12
        di[url_l[0]].add(url_l[14])
        labeled.add(url_l[14])
    if event.key == "f": #13
        di[url_l[0]].add(url_l[15])
        labeled.add(url_l[15])
    if event.key == "g": #14
        di[url_l[0]].add(url_l[16])
        labeled.add(url_l[16])
    if event.key == "h": #15
        di[url_l[0]].add(url_l[17])
        labeled.add(url_l[17])
    if event.key == "j": #16
        di[url_l[0]].add(url_l[18])
        labeled.add(url_l[18])
    if event.key == "k": #17
        di[url_l[0]].add(url_l[19])
        labeled.add(url_l[19])
    if event.key == "l": #18
        di[url_l[0]].add(url_l[20])
        labeled.add(url_l[20])
    if event.key == ";": #19
        di[url_l[0]].add(url_l[21])
        labeled.add(url_l[21])
    if event.key == "'": #20
        di[url_l[0]].add(url_l[22])
        labeled.add(url_l[22])

def to_list(val):
    return [val]

def load_labeled():
    # list total labeled data
    files = glob.glob("/Users/lap02387/PycharmProjects/image_matching_training/input_file/test_set_df_list/*")
    df_l = []
    for file in files:
        df_l.append(pd.read_parquet(file))
    df = pd.concat(df_l)
    df = df.reset_index()
    del df["index"]
    source = df.source.values
    candidate = df.candidate.values
    s = set()
    for e in source:
        s.add(e)
    for e in candidate:
        for ee in e:
            s.add(ee)
    return s

if __name__ == '__main__':
    error_indices_1_neighbor = np.load("./output/error_indices_1.npy")[0]
    labeled = load_labeled()
    # df = pd.read_parquet("./input_file/final_product_image_with_label_test_set.parquet")[["normalized_url_image", "label"]]
    df = pd.read_parquet("./output/similarity_df_20.parquet")
    df = df.reset_index()
    del df["index"]
    del df["duplicated"]
    # df = df.loc[error_indices_1_neighbor]
    # df["normalized_url_image"] = df["normalized_url_image"].apply(to_list)
    # print(len(df))
    # new_df = df.groupby("label").agg({"normalized_url_image": "sum"})
    # print(len(new_df))
    base_dir = "/Users/lap02387/pms/image_matching/images"
    for idx, row in df.iterrows():
        print(len(labeled))
        if row["normalized_url_image"] in labeled:
            print(idx)
            continue
        fig, ax = plt.subplots(4, 6)
        url_l = []
        url_l.append(row["normalized_url_image"])
        # print(len(labeled))
        url_l.append(row["correct_image"])
        for iiii in range(20):
            url_l.append(row[f"duplicated_{iiii}"])
        fig.canvas.mpl_connect("key_press_event", lambda event: on_press(event, url_l, idx))
        label = row["normalized_url_image"]
        duplicated = row["correct_image"]
        img_1 = io.imread(f"{base_dir}/{label}")
        img_2 = io.imread(f"{base_dir}/{duplicated}")
        ax[0][0].imshow(img_1)
        ax[0][0].set_title("source")
        ax[0][1].imshow(img_2)
        ax[0][1].set_title("label")
        for ii in range(20):
            temp_file = row[f"duplicated_{ii}"]
            temp_img = io.imread(f"{base_dir}/{temp_file}")
            iii = (ii + 2) // 6
            jjj = (ii + 2) % 6
            ax[iii][jjj].imshow(temp_img)
            ax[iii][jjj].set_title(f"candidate_{ii}")
        plt.show()


