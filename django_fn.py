import numpy as np
import torch
import requests
from io import BytesIO

from .model import CustomNet
from PIL import Image

def _output(img_url, model_pth):
    img_url = requests.get(img_url)  # requests를 이용해 img를 url로 받아온다.
    model = CustomNet()  # CustomNet 모델을 불러온다.
    model_fn = model_pth  # model_pth=best_model.pth를 model_fn에 할당한다.
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_fn, device), strict=False)  # model의 weight 값을 model_fn으로 설정한다.

    img = Image.open(BytesIO(img_url.content))  # img_url을 통해 이미지를 img에 할당한다.
    img = np.array(img)  # img의 dtype을 numpy array로 변경한다.
    img = torch.FloatTensor(img)  # img의 dtype을 tensor로 변경한다.
    img = img.permute(2, 0, 1)  # batch_size와 channel수를 바꾼다.
    img = img.unsqueeze(0)  # img의 shape 제일 앞에 1차원을 추가한다.
    
    y_hat = model(img)  # 받아온 img를 model에 넣어 결과 값을 y_hat에 할당한다.
    y_hat = torch.argmax(y_hat, dim=-1)  # y_hat에 argmax를 적용한다.
    y_hat = y_hat.tolist()  # y_hat을 list로 변경합니다.
    y_hat = max(y_hat, key=y_hat.count)  # 값들 중 가장 개수가 많은 값을 y_hat에 할당한다.
    y_hat = abs(int(y_hat))  # y_hat을 float -> int, 절대값을 적용한다.

    # y_hat의 값에 따라 다른 json을 return 한다.
    if y_hat == 0:
        emotion_object = {
            "emotion": "angry"
        }
        
        return emotion_object
    elif y_hat == 1:
        emotion_object = {
            "emotion": "fear"
        }
        
        return emotion_object
    elif y_hat == 2:
        emotion_object = {
            "emotion": "suprised"
        }
        
        return emotion_object
    elif y_hat == 3:
        emotion_object = {
            "emotion": "happy"
        }
        
        return emotion_object
    elif y_hat == 4:
        emotion_object = {
            "emotion": "sad"
        }
        
        return emotion_object
    else:
        emotion_object = {
            "emotion": "netural"
        }
        
        return emotion_object