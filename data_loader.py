import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


# 데이터를 불러오는 함수
def load_data(data_pth):
    df = pd.read_csv(data_pth)
    df['pixelist'] = [[int(y) for y in x.split()] for x in df['pixels']] # pixels의 type을 list로 바꾼 pixelist 컬럼 추가

    return df


# train, valid, test로 split하는 함수
def split_data(df):
    df_train = df[df['Usage'] == 'Training']
    df_valid = df[df['Usage'] == 'PublicTest']
    df_test = df[df['Usage'] == 'PrivateTest']

    return df_train, df_valid, df_test


# label을 얻는 함수
def get_labels(df_train, df_valid, df_test):
    train_labels = df_train['emotion'].tolist()
    valid_labels = df_valid['emotion'].tolist()
    test_labels = df_test['emotion'].tolist()

    return train_labels, valid_labels, test_labels


# torch에서 제공하는 Dataset을 이용하여 dataset class 정의
class EmotionDataset(Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.data = data
        self.transforms = transforms

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        x = self.data.iloc[index]['pixelist']
        x = np.array(x).reshape(48, 48, 1)
        x = torch.FloatTensor(x)
        x = x.numpy()

        y = self.data.iloc[index]['emotion']

        if self.transforms:
            x = self.transforms(x)
            x = torch.cat((x, x, x), 0)

        return x, y


# transform을 정의하는 함수
def get_transform(train=True): # train=True -> Train_data, train=False -> Valid_data, test_data
    if train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            transforms.RandomHorizontalFlip()
        ])

        return train_transform

    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        return test_transform


# dataloader를 return하는 함수
def get_loaders():
    df = load_data('./data/fer2013.csv')

    df_train, df_valid, df_test = split_data(df)

    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    train_dataset = EmotionDataset(df_train, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    valid_dataset = EmotionDataset(df_valid, test_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=0)

    test_dataset = EmotionDataset(df_test, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, train_dataset, valid_loader, valid_dataset, test_loader, test_dataset
    