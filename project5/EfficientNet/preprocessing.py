import cv2
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
import gc


def resize_image(img, org_width, org_height, new_width, new_height):
    img = 255 - img
    img = (img * (225 / img.max())).astype(np.uint8)
    img = img.reshape(org_height, org_width)
    image_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image_resized

def save_img(train_dir, img, org_width, org_height, new_width, new_height, image_id):
    image_resized = resize_image(img, org_width, org_height, new_width, new_height)
    cv2.imwrite(train_dir + str(image_id) + '.png', image_resized)

def generate_images(data_dir, train_dir, org_width, org_height, new_width, new_height):
    for i in tqdm(range(0, 4)):
        df = pd.read_parquet(data_dir + '/train_image_data_' + str(i) + '.parquet')
        image_ids = df['image_id'].values
        df = df.drop(['image_id'], axis=1)
        for image_id, index in zip(image_ids, range(df.shape[0])):
            save_img(train_dir, df.loc[df.index[index]].values, org_width, org_height, new_width, new_height, image_id)
        del df
        gc.collect()