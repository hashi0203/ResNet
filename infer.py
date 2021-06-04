import torch
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import *

import os
import argparse
import datetime
import glob

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Inference')
# checkpoint を指定
parser.add_argument('ckpt', type=str, help='specify the checkpoint to use')
parser.add_argument('--image', type=str, nargs='*', help='specify the images to infer')
args = parser.parse_args()

images = args.image
if images is None:
    images = glob.glob("./image/*")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = ResNet101(10).to(device) # ResNet101 の方が精度が高かったのでこちらを使用
if device == 'cuda':
    net = torch.nn.DataParallel(net) # DataParallel を使って高速化
    cudnn.benchmark = True

# checkpoint の読み込み
print('==> Loading checkpoint..')
checkpoint = torch.load(args.ckpt)
net.load_state_dict(checkpoint['net'])
net.eval()

def infer(imgs):
    global classes
    with torch.no_grad():
        imgs = imgs.to(device)
        outputs = net(imgs)

        _, predicted = outputs.max(1)

    ans = [classes[p] for p in predicted]
    return ans

def load_image(image):
    img = Image.open(image) # 画像読み込み
    img = img.convert('RGB') # RGB に変換
    return img.resize((32, 32), Image.LANCZOS) # 32 x 32 に圧縮

def cvt_image(image):
    img = load_image(image)
    img = torchvision.transforms.functional.to_tensor(img) # tensor に変換
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return norm(img) # 正規化

print('==> Inferring..')
# 入力画像データの生成
imgs = torch.stack([cvt_image(image) for image in images], dim=0)
# 推論
ans = infer(imgs)

# 答えをプリントする
l = len(max(images, key=len))
for i in range(len(ans)):
    print('%s: %s' % (images[i].ljust(l), ans[i]))

# 画像を保存する
def imsave(images, ans):
    imgs = list(map(load_image, images))
    t = datetime.datetime.now().strftime('%m%d-%H%M')
    cols = min(2, len(imgs))
    rows = (len(imgs)+1) // 2
    axes=[]

    fig=plt.figure()
    for (i, img) in enumerate(imgs):
        axes.append( fig.add_subplot(rows, cols, i+1) )
        axes[-1].set_title("Predicted: %s" % ans[i])
        plt.imshow(img)
    fig.tight_layout()
    if not os.path.isdir('answer'):
        os.mkdir('answer')
    plt.savefig("./answer/answer-%s.png" % t)

print('==> Saving answer in the answer directory..')
imsave(images, ans)