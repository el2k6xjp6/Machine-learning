import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.matlib as ml
import os

class Eigenface():
    def __init__(self,data):
        self.data=data
        if not os.path.exists('image'):
            os.mkdir('image')
        if not os.path.exists('image/eigenface'):
            os.mkdir('image/eigenface')
        if not os.path.exists('image/reconstruction'):
            os.mkdir('image/reconstruction')

if __name__ == "__main__":
    if not os.path.exists('image'):
        os.mkdir('image')
    if not os.path.exists('image/eigenface'):
        os.mkdir('image/eigenface')
        
    if not os.path.exists('image/reconstruction'):
        os.mkdir('image/reconstruction')
    for t in range(1,41):
        img=[Image.open('data/att_faces/s{}/{}.pgm'.format(t,i)).convert('L').resize((92, 112)) for i in range(1,11)]
        face=np.asarray([np.array(i).flatten() for i in img])
        mean=ml.repmat(np.mean(face,axis=0),face.shape[0],1)
        data=face-mean
        U,sig,_=np.linalg.svd(data.T)
        gap=np.array([sig[i+1]-sig[i] for i in range(sig.shape[0]-1)]).argmin()
        eigenface=U[:,0]
        for i in range(1,gap+1):
            eigenface+=U[:,i]
        plt.imshow((eigenface).reshape(112,92),cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig('image/eigenface/{}.png'.format(t))
        plt.close('all')
        reconstruct=np.array([ x*eigenface*eigenface.T for x in face])
        if not os.path.exists('image/reconstruction/{}'.format(t)):
            os.mkdir('image/reconstruction/{}'.format(t))
        for i in range(10):
            plt.imshow(reconstruct[i].reshape(112,92),cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig('image/reconstruction/{}/{}.png'.format(t,i+1))
            plt.close('all')