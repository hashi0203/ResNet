import torch
from model import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch Checkpoint Conversion from CPU to GPU')
parser.add_argument('ckpt_in', type=str, help='specify the checkpoint to convert')
parser.add_argument('ckpt_out', type=str, help='specify the output checkpoint filename')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: to convert checkpoints, you need a GPU!'

print('==> Building model..')
net = ResNet101(10) # ResNet101 の方が精度が高かったのでこちらを使用

# checkpoint の読み込み
print('==> Loading checkpoint..')
checkpoint = torch.load(args.ckpt_in)
net.load_state_dict(checkpoint['net'])
best_accuracy = checkpoint['acc']
start_epoch = checkpoint['epoch']
net = torch.nn.DataParallel(net) # DataParallel を使って高速化
state = {
    'net': net.state_dict(),
    'acc': best_accuracy,
    'epoch': start_epoch,
}
print('==> Saving checkpoint in %s..' % args.ckpt_out)
torch.save(state, args.ckpt_out)
