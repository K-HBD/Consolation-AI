import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import json

from PIL import Image


# matplotlib을 통해 이미지를 출력하고, label값을 표시한다.
def show_img(img, label, num):
    z = np.array(img['pixelist'][num])
    zz = z.reshape(48, 48)
    plt.imshow(zz, interpolation='nearest', cmap='gray')
    plt.show()
    print(label[num])


# test_loader를 이용한 정확도 확인
def test(model, crit, test_dataset, test_loader, device):
    model = model.to(device)
    model.eval()
    correct_cnt = 0

    with torch.no_grad():
        for x, y in iter(test_loader):
            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            
            c_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
            t_cnt = float(x.size(0))

            correct_cnt += c_cnt
            accuracy = c_cnt / t_cnt
            print("Accuracy: %.4f" % accuracy)

        total_cnt = len(test_dataset)
        print(f"correct_cnt: {correct_cnt}/{total_cnt}")


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
