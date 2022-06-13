import torch
import myNN
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def paint_mask(img, mask, vis_channel = 1, threshold = 0.5):
    img = np.array(img)
    if img.dtype == 'uint8':
            visImage = img.copy().astype('f4')/255
    else:
        visImage = img.copy()

    channelPart = visImage[:, :, vis_channel] * (mask > threshold) - 1
    channelPart[channelPart < 0] = 0
    visImage[:, :, vis_channel] = visImage[:, :, vis_channel] * (mask <= threshold) + (mask > threshold) * 1 + channelPart

    return visImage

def preprocess_img(data, resize=False):
    if resize:
        data = data.resize(myNN.INPUT_IMAGE_SHAPE)

    return data

def toTensor(data):
    data = np.array(data)
    data = torch.tensor(data)
    data = data.permute(2, 0, 1)[None,...]

    data = data.cuda().float()

    return data


def postprocess_out(out):
    out = out[0].cpu().detach().numpy()
    out -= np.min(out)
    out /= np.max(out)
    out = 1 - out
    return out 


def load_model(path):
    model = myNN.myNN()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path))

    return model