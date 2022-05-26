import numpy as np
import os
import struct
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

# 标签one-hot处理
def onehot(targets):
    num=len(targets)
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid的一阶导数
def Dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


class NN(object):
    def __init__(self, l0, l1, l2):
        self.lr = 0.1                                        # 学习率
        self.W1 = np.random.randn(l0, l1) * 0.01             # 初始化
        self.b1 = np.random.randn(l1) * 0.01
        self.W2 = np.random.randn(l1, l2) * 0.01
        self.b2 = np.random.randn(l2) * 0.01

    # 前向传播
    def forward(self, X, y):
        self.X = X                                           # m x 784
        self.z1 = np.dot(X, self.W1) + self.b1               # m x 500, 等于中间层层数
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2         # m x 10
        self.a2 = sigmoid(self.z2)
        loss = np.sum((self.a2 - y) * (self.a2 - y)) / 2     # 均方差
        self.d2 = (self.a2 - y) * Dsigmoid(self.z2)          # m x 10 , 用于反向传播
        return loss, self.a2

    # 反向传播
    def backward(self):
        dW2 = np.dot(self.a1.T, self.d2) / 3                  # 500 x 10, batchsize=3
        db2 = np.sum(self.d2, axis=0) / 3                     # 10
        d1 = np.dot(self.d2, self.W2.T) * Dsigmoid(self.z1)   # m x 500, 用于反向传播
        dW1 = np.dot(self.X.T, d1) / 3                        # 784x 500
        db1 = np.sum(d1, axis=0) / 3                          # 500

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


def val():
    r = np.load("data.npz")
    nn = NN(784, 500, 10)
    nn.W1 = r["w1"]
    nn.b1 = r["b1"]
    nn.W2 = r["w2"]
    nn.b2 = r["b2"]
    _, result = nn.forward(test_image, test_labels)
    result = np.argmax(result, axis=1)
    precison = np.sum(result==test_labels) / 10000
    print("Precison:", precison)


def load_mnist(path='./mnist', kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)

    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    # 读入magic是一个文件协议的描述,也是调用fromfile 方法将字节读入NumPy的array之前在文件缓冲中的item数(n).

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    labels=onehot(labels)
    return images, labels


if __name__ == '__main__':

    # Mnist手写数字集
    train_image,train_labels = load_mnist(path='./mnist',kind='train')
    test_image,test_labels=load_mnist(path='./mnist',kind='test')

    nn = NN(784, 500, 10)
    losses=[]
    batchsize=4
    for epoch in range(5):
        for i in range(0, 10000-batchsize):
            x = train_image[i:i+batchsize]
            y = train_labels[i:i+batchsize]
            loss, _ = nn.forward(x, y)
            print("Epoch:", epoch, "-", i, ":", "{:.3f}".format(loss))
            nn.backward()
            losses.append(loss)
        np.savez("data.npz", w1=nn.W1, b1=nn.b1, w2=nn.W2, b2=nn.b2)

    plt.plot([i for i in range(len(losses))],losses)
    plt.savefig('./results.png')
    plt.show()