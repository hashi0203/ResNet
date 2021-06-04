import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from model import *

import os
import argparse
import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# --lr オプションで開始時の学習率を指定
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# --resume, -r オプションで開始時のモデルを指定
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
# --summary, -s オプションで torchsummary を表示するかを指定
parser.add_argument('--summary', '-s', action='store_true', help='show torchsummary')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_accuracy = 0  # テストの正解率の最大値
start_epoch = 0  # 0 か checkpoint のエポック数からスタート
n_epoch = 250 # epoch の回数

print('==> Preparing data..')
#変換器の作成
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # torch.Tensor へ変換
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 正規化する
])
transform_test = transforms.Compose([
    transforms.ToTensor(), # torch.Tensor へ変換
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 正規化する
])

#訓練データのダウンロードと変換
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#訓練データのローダ(読み込み器)の作成
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

#テストデータのダウンロードと変換
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#テストデータのローダ(読み込み器)の作成
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = ResNet101(10).to(device) # ResNet101 の方が精度が高かったのでこちらを使用
if device == 'cuda':
    net = torch.nn.DataParallel(net) # DataParallel を使って高速化
    cudnn.benchmark = True

if args.summary:
    # summary の表示
    summary(net, (3, 32, 32))

if args.resume:
    # checkpoint の読み込み
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_accuracy = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # ローダからデータを読み込む; データは [inputs, labels] の形で取得される．
        inputs, targets = inputs.to(device), targets.to(device)
        # 勾配を0に初期化する（逆伝播に備える）．
        optimizer.zero_grad()
        # 順伝播 + 逆伝播 + 最適化
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 統計を表示する．
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 99:
            print('[%3d, %5d] train loss: %.3f train accuracy: %.2f %%' %
                (epoch + 1, batch_idx + 1, train_loss / (batch_idx+1), 100.*correct/total))

    return train_loss/len(trainloader), 100.*correct/total

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, targets) in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    test_loss /= len(testloader)
    print('Epoch: %4d  test  loss: %.3f test  accuracy: %.2f %%' %
        (epoch+1, test_loss, accuracy))
    return test_loss, accuracy

if not os.path.isdir('log'):
    os.mkdir('log')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

t = datetime.datetime.now().strftime('%m%d-%H%M')
LOG_FILE = './log/log-%s.txt' % t
MODEL_FILE = './checkpoint/ckpt-%s.pth' % t
f = open(LOG_FILE, 'w')
f.write('train_loss, train_accuracy, test_loss, test_accuracy\n')
f.close()

for epoch in range(start_epoch, start_epoch+n_epoch):
    train_loss, train_accuracy = train(epoch)
    test_loss, test_accuracy = test(epoch)

    f = open(LOG_FILE, 'a')
    f.write('    %.4f,         %.3f,    %.4f,        %.3f\n' %
        (train_loss, train_accuracy, test_loss, test_accuracy))
    f.close()

    # checkpoint の保存
    if test_accuracy > best_accuracy:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_accuracy,
            'epoch': epoch,
        }
        torch.save(state, MODEL_FILE)
        best_accuracy = test_accuracy

    scheduler.step()
