import numpy as np
import random

class Pocket():
    def __init__(self, data, test, lr=1):
        self.X = np.concatenate((np.ones((data.shape[0],1),dtype=np.float), data[:,:4]), axis = 1)
        self.test = np.concatenate((np.ones((data.shape[0],1),dtype=np.float), test[:,:4]), axis = 1)
        self.Y = data[:,4]
        self.Yt = test[:,4]
        self.W = np.zeros(5)
        self.size = data.shape[0]
        self.lr = lr
        self.order = [i for i in range(data.shape[0])]
        random.shuffle(self.order)

    def evaluate(self):
        error = 0
        for i, x in enumerate(self.test):
            if np.sign(np.dot(x, self.W)) != self.Yt[i]:
                error += 1
        return error

    def error(self, W):
        error = 0
        for i, x in enumerate(self.X):
            if np.sign(np.dot(x, W)) != self.Y[i]:
                error += 1
        return error

    def random_check_error(self):
        for i in self.order:
            if np.sign(np.dot(self.X[i], self.W)) != self.Y[i]:
                Wt = self.W + (self.lr * self.Y[i] * self.X[i])
                if self.error(Wt) < self.error(self.W):
                    self.W = Wt
                    print(i)
                break

    # def random_check_error(self):
    #     for i in self.order:
    #         if np.sign(np.dot(self.X[i], self.W)) != self.Y[i]:
    #             Wt = self.W + (self.lr * self.Y[i] * self.X[i])
    #             self.W = Wt
    #             break

    def run(self, iter):
        for _ in range(iter):
            self.random_check_error()
        return self.evaluate()

if __name__ == "__main__":
    data = np.loadtxt('train')
    test = np.loadtxt('test')
    order = [i for i in range(data.shape[0])]
    errors = 0
    for i in range(2000):
        print('========={}========='.format(i))
        p = Pocket(data, test)
        e = p.run(100)
        print(e)
        errors += e
    print('average number of updates: {}'.format(errors/2000/test.shape[0]))