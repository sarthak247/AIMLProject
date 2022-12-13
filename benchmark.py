from model import ReCoNet
import torch
# from utils import nhwc_to_nchw, nchw_to_nhwc, postprocess_reconet, preprocess_for_reconet
import time
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


def benchmark(model, input_shape, dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
    return np.mean(timings)*1000, nruns/np.sum(timings)

if __name__ == '__main__':
    model = ReCoNet(frn=True)
    model.load_state_dict(torch.load('models/starry_night_3_frn/model.pth', map_location='cuda:0'))
    model.eval()
    model = model.to(device = 'cuda')

    cudnn.benchmark = True

    shapes = [2**i for i in range(5, 10)]
    times = []
    fps = []
    for shape in shapes:
        print(f'Input shape: {shape} x {shape}', )
        output = benchmark(model, input_shape=(1, 3, shape, shape))
        times.append(output[0])
        fps.append(output[1])
    
    # Shape vs Time Plot
    shapes = [str(shape) for shape in shapes]
    plt.bar(shapes, times, data=times)
    plt.xlabel('Input shape (Resolution)')
    plt.ylabel('Time per frame (in ms)')
    plt.title('Resolution vs Time Plot')
    for i in range(len(shapes)):
        plt.annotate(str(round(times[i], 2)), xy=(shapes[i],times[i]), ha='center', va='bottom')
    plt.savefig('benchmarks.png')
    plt.show()
    # plt.plot(shapes, times, 'r-')
    # plt.xlabel('Input shape')
    # plt.ylabel('Time per frame (in ms)')
    # plt.title('Shape vs Time Plot')
    # plt.savefig('benchmarks.png')
    # plt.show()

    # Shape vs FPS plot
    plt.clf()
    plt.bar(shapes, fps, data=times)
    plt.xlabel('Input shape (Resolution)')
    plt.ylabel('FPS')
    plt.title('Resolution vs FPS Plot')
    for i in range(len(shapes)):
        plt.annotate(str(round(fps[i], 2)), xy=(shapes[i],fps[i]), ha='center', va='bottom')
    plt.savefig('benchmarks2.png')
    plt.show()
    # plt.clf()
    # plt.plot(shapes, fps, 'r-')
    # plt.xlabel('Input shape')
    # plt.ylabel('FPS')
    # plt.title('Shape vs FPS Plot')
    # plt.savefig('benchmarks2.png')
    # plt.show()

