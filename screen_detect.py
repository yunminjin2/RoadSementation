import os
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import torch
import threading
import myNN
import util




BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YOUTUBE_GRAB_AREA = (0, 300, 1200, 800)
GAME_GRAB_AREA = (0, 0, 1100, 900)

def roadDetect(frame, model):
    data = util.toTensor(frame)
            
    out = model(data)

    out = util.postprocess_out(out)

    out = util.paint_mask(frame, out, threshold=0.5)
   

    
    return out


if __name__ == "__main__":
    model = util.load_model("./weights/model_road_detect.ckpt")
    model.eval()

    while(True):
        # Grab Image of screen
        screen = np.array(ImageGrab.grab(bbox = YOUTUBE_GRAB_AREA))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        FRAME = cv2.resize(frame, myNN.INPUT_IMAGE_SHAPE)

        
        out = roadDetect(FRAME, model)
        cv2.imshow("result", out)


        #object_lists = []
        #myCar.drive(lines, object_lists)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    