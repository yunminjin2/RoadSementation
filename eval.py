import cv2
from PIL import Image
import os
import random

import torch
import matplotlib
import matplotlib.pyplot as plt


import myNN
import util


ROOT = "./data_road/testing/image_2/"
MODEL_PATH = "model_road_detect.ckpt"

model = None

def single_image(img_name):
    img = Image.open(img_name)
    img = util.preprocess_img(img, resize=True)
    
    data = util.toTensor(img)

    out = model(data)
    out = util.postprocess_out(out)

    result = util.paint_mask(img, out)
    cv2.imshow("result", result)
    cv2.waitKey()

def multi_image():
    fig, ax = plt.subplots(2, 5, figsize=(10, 5))

    files = os.listdir(ROOT)
    
    for row in range(2):
        for i in range(5):
            img_name = files[int(random.random() * len(files))]
            img = Image.open(ROOT + img_name).convert("RGB")
            img = util.preprocess_img(img, resize=True)

            data = util.toTensor(img)
            
            out = model(data)

            out = util.postprocess_out(out)
            out = util.paint_mask(img, out)
            ax[row, i].set_title(img_name)
            ax[row, i].imshow(out)
    
    plt.show()

if __name__=="__main__":
    model = util.load_model(MODEL_PATH)
    model.eval()

    #single_image(ROOT + "um_000076.png")
    multi_image()
    
   

