import cv2
import numpy as np
import onnxruntime as ort
import utils

STYLE_TRANSFORM_PATHS = ['style_transfer_gogh_256a100.onnx']
PRESERVE_COLOR = [False]
WIDTH = 256
HEIGHT = 256


def webcam(style_transform_paths, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Load ONNX model
    ort_sessions=[]
    for P in STYLE_TRANSFORM_PATHS:
        ort_sessions.append(ort.InferenceSession(P))
    input_name = ort_sessions[0].get_inputs()[0].name
    output_name = ort_sessions[0].get_outputs()[0].name

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    model_id = 0

    # Main loop
    while True:
        # Get webcam input
        ret_val, img = cam.read()

        # Mirror 
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (width, height))

        # Generate image
        
        content_tensor = np.transpose(img, (2, 0, 1))
        content_tensor = np.expand_dims(content_tensor, axis=0).astype(np.float32)/255.0
        
        generated_tensor = ort_sessions[model_id].run([output_name], {input_name: content_tensor})[0]
        generated_image = generated_tensor.squeeze()
        generated_image = generated_image.transpose(1, 2, 0)

        if PRESERVE_COLOR[model_id]:
            generated_image = utils.transfer_color(img, generated_image)

        generated_image = generated_image / 255

        # Show webcam
        cv2.imshow('Demo webcam', generated_image)
        key = cv2.waitKey(1)
        if key != -1: 
            if key == 27:
                break  # esc to quit
            else:
                model_id += 1
                model_id %= len(style_transform_paths)
                print('loading model', model_id)

    cam.release()
    cv2.destroyAllWindows()


webcam(STYLE_TRANSFORM_PATHS, WIDTH, HEIGHT)