import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Visualizer')
parser.add_argument('file', metavar='FILE', type=str, nargs='*', help='input file')

args = parser.parse_args()

for file in args.file:
    f = open(file, 'r')
    s = f.read()
    f.close()

    s = s.splitlines()[1:]
    s = [ss.split() for ss in s]
    s = np.array([[float(sss[:-1]) for sss in ss] for ss in s]).T
    epochs = np.arange(1, s.shape[1]+1)

    d = file[-13:-4]
    if not os.path.isdir('graph'):
        os.mkdir('graph')

    plt.figure()
    plt.plot(epochs, s[0], label="train", color='tab:blue')
    am = np.argmin(s[0])
    plt.plot(epochs[am], s[0][am], color='tab:blue', marker='x')
    plt.text(epochs[am], s[0][am]-0.02, '%.4f' % s[0][am], horizontalalignment="center", verticalalignment="top")

    plt.plot(epochs, s[2], label="test", color='tab:orange')
    am = np.argmin(s[2])
    plt.plot(epochs[am], s[2][am], color='tab:orange', marker='x')
    plt.text(epochs[am], s[2][am]+0.02, '%.4f' % s[2][am], horizontalalignment="center", verticalalignment="bottom")

    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('./graph/loss-%s.png' % d)


    plt.figure()
    plt.plot(epochs, s[1], label="train")
    am = np.argmax(s[1])
    plt.plot(epochs[am], s[1][am], color='tab:blue', marker='x')
    plt.text(epochs[am], s[1][am]+0.5, '%.3f' % s[1][am], horizontalalignment="center", verticalalignment="bottom")

    plt.plot(epochs, s[3], label="test")
    am = np.argmax(s[3])
    plt.plot(epochs[am], s[3][am], color='tab:orange', marker='x')
    plt.text(epochs[am], s[3][am]-0.5, '%.3f' % s[3][am], horizontalalignment="center", verticalalignment="top")

    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig('./graph/accuracy-%s.png' % d)