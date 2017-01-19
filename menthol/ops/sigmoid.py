import numpy as np


class sigmoid(object):

    def __init__(self):
        self.result = None

    def forward(self, x):
        self.result = 1 / (1 + np.e**(-x))
        return self.result

    def backward(self, L):
        loss = L * (self.result * (1 - self.result))
        self.result = None
        return loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sig = sigmoid()

    # xs = []
    # forrs = []
    # backs = []

    # for i in np.arange(-6, 5, 0.1):
        # xs.append(i)
        # forrs.append(sig.forward(i))
        # backs.append(sig.backward(1.0))

    xs = np.arange(-6, 5, 0.1)
    forrs = sig.forward(xs)
    backs = sig.backward(np.ones(xs.shape))

    plt.plot(xs, forrs, 'r')
    plt.plot(xs, backs, 'b')

    plt.show()
