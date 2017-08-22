import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

def get_mnist_mean_std():
    MNIST = datasets.MNIST('data', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    _sum, _squred_sum = 0, 0
    for idx, data in enumerate(MNIST):
        img, label = data
        _batch, _h, _w = img.size()
        _sum += torch.sum(img)
        _squred_sum += torch.sum(img ** 2)
    mean = _sum / len(MNIST) / _h / _w
    var = _squred_sum / len(MNIST) / _h / _w - mean ** 2
    std = var ** 0.5
    print(f'mean : {mean}, std : {std}')
    return mean, std

def z_normalize_tensor(tensor, mean, std):
    return (tensor - mean) / std

def get_mnist_loader(train=True, bsize=64):
    MNIST = datasets.MNIST('data', train=train, transform=transforms.ToTensor(), target_transform=None, download=True)
    loader = torch.utils.data.DataLoader(MNIST, batch_size=bsize, shuffle=True)
    return loader

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.input_mean, self.input_std = get_mnist_mean_std()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        nn.init.normal(self.fc1.weight, mean=0, std=(28*28)**(-0.5))
        nn.init.normal(self.fc2.weight, mean=0, std=(1024)**(-0.5))
        nn.init.normal(self.fc3.weight, mean=0, std=(512)**(-0.5))
    
    def forward(self, inp):
        inp = z_normalize_tensor(inp, self.input_mean, self.input_std)
        inp = inp.view(-1, 28 * 28)
        inp = F.selu(self.fc1(inp))
        inp = F.selu(self.fc2(inp))
        inp = F.selu(self.fc3(inp))
        inp = self.fc4(inp)
        return inp

def train(model, loader, optim, crit):
    learning_sum = 0
    learning_size = 0

    for bid, data in enumerate(loader):
        optim.zero_grad()

        img, label = data
        learning_size += img.size()[0]
        img, label = Variable(img.cuda()), Variable(label.cuda())

        pred = model(img)

        loss = crit(pred, label)
        loss.backward()
        optim.step()

        _, pred_id = torch.max(pred, 1)
        corrects = pred_id == label
        learning_sum += torch.sum(corrects.float()).data[0]

        if bid % 100 == 0:
            print(f'[batch {bid} / {len(loader)}] Training Accuracy : {learning_sum / learning_size}')

def evaluation(model, loader):
    learning_sum = 0
    learning_size = 0

    for bid, data in enumerate(loader):
        img, label = data
        learning_size += img.size()[0]
        img, label = Variable(img.cuda()), Variable(label.cuda())

        pred = model(img)

        _, pred_id = torch.max(pred, 1)
        corrects = pred_id == label
        learning_sum += torch.sum(corrects.float()).data[0]

    print(f'Test Accuracy : {learning_sum / learning_size}')

def main():
    snn = SNN()
    snn.cuda()
    optim = torch.optim.Adam(snn.parameters())
    crit = nn.CrossEntropyLoss()

    for epoch in range(10):
        snn.train()
        loader = get_mnist_loader(train=True)
        print(f'{"=" * 10} EPOCH {epoch} {"=" * 10}')
        train(snn, loader, optim, crit)
    
        snn.eval()
        loader = get_mnist_loader(train=False)
        evaluation(snn, loader)

if __name__ == '__main__':
    main()