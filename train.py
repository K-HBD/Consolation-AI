import torch
import torch.nn as nn
import torch.optim as optim

from model import CustomNet
from trainer import Trainer
from data_loader import get_loaders
from utils import y2backend, test


def main(image):
    # 하이퍼 파라미터 정의 
    model = CustomNet()
    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    crit = nn.CrossEntropyLoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # loaders 불러오기
    train_loader, valid_loader, test_loader = get_loaders()
    
    trainer = Trainer(model, optimizer, crit, device)
    trainer.train(train_loader, valid_loader)

    torch.save({
        'model': trainer.model.state_dict()
    }, 'best_model.pth')

    test(model, test_loader, device)    
    y2backend('./test/sad.jpeg', 'best_model.pth', model=model, device=device)



if __name__ == '__main__':
    main()