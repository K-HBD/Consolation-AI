import torch
import torch.nn as nn
import torch.optim as optim

from model import CustomNet
from trainer import Trainer
from data_loader import get_loaders
from utils import test


def main():
    # 하이퍼 파라미터 정의 
    model = CustomNet()
    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    crit = nn.CrossEntropyLoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    epochs = 50

    # loaders 불러오기
    train_loader, train_dataset, valid_loader, valid_dataset, test_loader, test_dataset = get_loaders()
    
    trainer = Trainer(model, optimizer, crit, device, epochs)
    trainer.train(train_loader, train_dataset, valid_loader, valid_dataset)

    torch.save({
        'model': trainer.model.state_dict()
    }, 'best_model.pth')

    test(model, crit, test_dataset, test_loader, device)


if __name__ == '__main__':
    main()