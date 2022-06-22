import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json

from PIL import Image

from model import CustomNet
from data_loader import EmotionDataset


# matplotlib을 통해 이미지를 출력하고, label값을 표시한다.
def show_img(img, label, num):
    z = np.array(img['pixelist'][num])
    zz = z.reshape(48, 48)
    plt.imshow(zz, interpolation='nearest', cmap='gray')
    plt.show()
    print(label[num])


# test_loader를 이용한 정확도 확인
def test(model, test_loader, device):
    model = model.to(device)
    model.eval()
    crit = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in iter(test_loader):
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            
            correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
            total_cnt = float(x.size(0))

            accuracy = correct_cnt / total_cnt
            print("Accuracy: %.4f" % accuracy)


# backend 연동을 위한 함수
def y2backend(img_pth, model_pth, model, device):
    model = model.to(device)
    model.load_state_dict(torch.load(model_pth), strict=False)
    img = Image.open(img_pth)
    img = np.array(img)
    img = torch.FloatTensor(img)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)
    img = img.to(device)

    y_hat = model(img) # output 생성
    y_hat = torch.argmax(y_hat, dim=-1)
    y_hat = y_hat.tolist()
    y_hat = max(y_hat, key=y_hat.count)
    print(y_hat)
    y_hat = abs(int(y_hat)) # 나온 값을 절대값 및 int 형으로 변경

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


# train_loss와 train_acc에 대하여 시각화를 해주는 함수
def train_acc_loss_plot(train_loss, train_acc, label):
    fig, axs = plt.subplot(1, 2, figsize=(20, 8))
    axs[0].plot(train_loss, label=label)
    axs[0].set_title('Train Loss')
    axs[1].plot(train_acc, label=label)
    axs[1].set_title("Train ACC")


# test_loss와 test_acc에 대하여 시각화를 해주는 함수
def test_acc_loss_plot(test_loss, test_acc, label):
    fig, axs = plt.subplot(1, 2, figsize=(20, 8))
    axs[0].plot(test_loss, label=label)
    axs[0].set_title('Test Loss')
    axs[1].plot(test_acc, label=label)
    axs[1].set_title("Test ACC")
