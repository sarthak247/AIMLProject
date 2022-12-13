import os
import argparse
import time

import numpy as np

# from lib import ReCoNetModel
from model import ReCoNet
import torch
import cv2
from utils import nhwc_to_nchw, nchw_to_nhwc, postprocess_reconet, preprocess_for_reconet



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", help="Path to input video file")
    # parser.add_argument("--output", help="Path to output video file")
    # parser.add_argument("--model", help="Path to model file")
    # parser.add_argument("--use-cpu", action='store_true', help="Use CPU instead of GPU")
    # parser.add_argument("--gpu-device", type=int, default=None, help="GPU device index")
    # parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    # parser.add_argument("--fps", type=int, default=None, help="FPS of output video")
    # parser.add_argument("--frn", action='store_true', help="Use Filter Response Normalization and TLU ")

    # args = parser.parse_args()

    # batch_size = args.batch_size

    model = ReCoNet(frn=True)
    model.load_state_dict(torch.load('models_full/starry_night/model.pth', map_location='cuda:0'))
    model.eval()
    model = model.to(device = 'cuda')


    frame = cv2.imread('test_image.jpeg')
    orig_ndim = frame.ndim
    frame = cv2.cvtColor(cv2.UMat(frame), cv2.COLOR_BGR2RGB)
    frame = cv2.UMat.get(frame)
    frame = frame[None, ...]
    frame = torch.from_numpy(frame)
    frame = frame.to(device = 'cuda')
    frame= nhwc_to_nchw(frame)
    frame = frame.to(torch.float32) / 255
    with torch.no_grad():
        frame = frame.to(device = 'cuda')
        frame = preprocess_for_reconet(frame)
        styled_frame = model(frame)
        styled_frame = postprocess_reconet(styled_frame)
        styled_frame = styled_frame.cpu()
        styled_frame = torch.clamp(styled_frame * 255, 0, 255).to(torch.uint8)
        styled_frame = nchw_to_nhwc(styled_frame)
        styled_frame = styled_frame.numpy()
        if orig_ndim == 3:
                styled_frame = styled_frame[0]
        styled_frame = cv2.cvtColor(cv2.UMat(styled_frame), cv2.COLOR_RGB2BGR)
        cv2.imwrite('style_frame_frn.jpeg', styled_frame)