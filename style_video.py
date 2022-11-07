import os
import argparse
import time

import numpy as np

# from lib import ReCoNetModel
from model import ReCoNet
from ffmpeg_tools import VideoReader, VideoWriter
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
    model.load_state_dict(torch.load('models/starry_night_3_frn/model.pth', map_location='cuda:0'))
    model.eval()
    model = model.to(device = 'cuda')


    vidcap = cv2.VideoCapture('test.mp4')
    success,image = vidcap.read()
    height, width, layers = image.shape
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter('output.avi', 0, 25, (width, height))
    count = 0
    t1 = time.time()
    while success:
        count += 1
        print(count)
        success, frame = vidcap.read()
        if success:
            orig_ndim = frame.ndim
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
            cv2.imwrite('frames_frn/' + str(count) + '.png', styled_frame)
        # wait 20 milliseconds between frames and break the loop if the `q` key is pressed
        if cv2.waitKey(20) == ord('q'):
            break

# we also need to close the video and destroy all Windows
    vidcap.release()
    output.release()
    cv2.destroyAllWindows()
    t2 = time.time()
    print(count/((t2-t1)))