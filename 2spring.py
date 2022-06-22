import numpy as np
import torch
from zmq import device

from model import CustomNet
from PIL import Image

def _output(img_pth, model_pth):
    model = CustomNet()
    model_fn = model_pth
    model.load_state_dict(torch.load(model_fn), strict=False)

    img = Image.open(img_pth)
    img = np.array(img)
    img = torch.FloatTensor(img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    
    y_hat = model(img)
    y_hat = torch.argmax(y_hat, dim=-1)
    y_hat = y_hat.tolist()
    y_hat = max(y_hat, key=y_hat.count)
    y_hat = abs(int(y_hat))

    if y_hat == 0:
        emotion_object = {
            "angry"
        }
        
        return emotion_object
    elif y_hat == 1:
        emotion_object = {
            "fear"
        }
        
        return emotion_object
    elif y_hat == 2:
        emotion_object = {
            "suprised"
        }
        
        return emotion_object
    elif y_hat == 3:
        emotion_object = {
            "happy"
        }
        
        return emotion_object
    elif y_hat == 4:
        emotion_object = {
            "sad"
        }
        
        return emotion_object
    else:
        emotion_object = {
            "netural"
        }
        
        return emotion_object


if __name__ == '__main__':
    main()