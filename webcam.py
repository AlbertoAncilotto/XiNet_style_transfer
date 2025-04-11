import cv2
from transformer import TransformerNetworkXiNet
import torch
import utils
from model_profiler import model_info
import numpy as np


STYLE_TRANSFORM_PATHS = [  "checkpoints/spaghetti256_a75/model/checkpoint_0.pth"]
PRESERVE_COLOR = [False, False, False, False]
WIDTH = 160
HEIGHT = 160

def webcam(style_transform_paths, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    print("Loading Transformer Network")
    nets = []
    for style_transform_path in style_transform_paths:
        net = TransformerNetworkXiNet(lite=False, alpha=0.75, bn_instead_of_in=True)
        model_info(net, img_size=256)
        net.load_state_dict(torch.load(style_transform_path))
        net.eval()
        nets.append(net.to(device))
        print("Done Loading Transformer Network")


    # Set webcam settings
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture('picture_frame\BNS2023.mp4')
    cam.set(3, width)
    model_id=0
    # Main loop
    added_noise=(np.random.rand(height, width, 3)-0.5)*20
    with torch.no_grad():
        while True:
            # Get webcam input
            ret_val, img = cam.read()

            # Mirror 
            img = cv2.resize(img, (width, height)).astype(np.float32)
            # img+= added_noise
            img = cv2.flip(img, 1)/300#/260.0

            # Generate image
            content_tensor = utils.itot(img).to(device)
            content_tensor = torch.clip(content_tensor, 0.0, 1.0)*255
            print('content batch stats:',torch.min(content_tensor), torch.max(content_tensor), torch.mean(content_tensor))
            generated_tensor = nets[model_id](content_tensor)     
            print('generated_tensor batch stats:',torch.min(generated_tensor), torch.max(generated_tensor), torch.mean(generated_tensor))
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR[model_id]):
                generated_image = utils.transfer_color((img*255.0).astype(np.uint8), generated_image.astype(np.uint8))

            # Show webcam
            cv2.imshow('Demo webcam', generated_image.astype(np.uint8))
            key = cv2.waitKey(1)
            if key != -1: 
                if key == 27:
                    break  # esc to quit
                else:
                    model_id+=1
                    model_id%=len(STYLE_TRANSFORM_PATHS)
                    print('loading model', model_id)
                    net=nets[model_id]
            
    cam.release()
    cv2.destroyAllWindows()
    modelpath = 'style_transfer.onnx'
    input_names = [ "input1" ]
    output_names = [ "output1" ]
    # Free-up memories
    torch.onnx.export(net, content_tensor, modelpath, export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True, 
                  input_names=input_names, output_names=output_names
                  )

webcam(STYLE_TRANSFORM_PATHS, WIDTH, HEIGHT)
