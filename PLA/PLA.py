import numpy as np
import random

class PLA():
    def __init__(self, data, s=0, lr=1):
        self.X = np.concatenate((np.ones((data.shape[0],1),dtype=np.float), data[:,:4]), axis = 1)
        self.Y = data[:,4]
        self.W = np.zeros(5)
        self.size = data.shape[0]
        self.lr = lr
        self.order = [i for i in range(data.shape[0])]
        random.seed(s)
        random.shuffle(self.order)


    def check_error(self):
        error = 0
        for i, x in enumerate(self.X):
            if np.sign(np.dot(x, self.W)) != self.Y[i]:
                error += 1
                self.W += (self.lr * self.Y[i] * x)
                break
        # print('Error rate: {}'.format(error/self.size))
        return (error > 0)

    def random_check_error(self):
        error = 0
        for i in self.order:
            # print(self.X[i], self.Y[i])
            if np.sign(np.dot(self.X[i], self.W)) != self.Y[i]:
                error += 1
                self.W += (self.lr * self.Y[i] * self.X[i])
                # print('update {}'.format(i))
                break
        # print('Error rate: {}'.format(error/self.size))
        return (error > 0)

    def run(self):
        updates = 0
        while(1):
            updates += 1
            if not(self.check_error()):
                break

    def run2(self):
        updates = 0
        while(1):
            updates += 1
            if not(self.random_check_error()):
                return updates
       

if __name__ == "__main__":
    data = np.loadtxt('train')
    # p = PLA(data)
    # p.run()
    s = 0
    updates = 0
    for i in range(2000):
        p = PLA(data, s, 0.5)
        print(s)
        updates += p.run2()
        s += 1
    print('average number of updates: {}'.format(updates/2000))