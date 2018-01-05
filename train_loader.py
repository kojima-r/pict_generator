import glob
from PIL import Image
import numpy as np
from matplotlib import pylab as plt


def load_image_dir(path,size):
	l=glob.glob(path)
	imgs=[]
	for fn in l:
		img=Image.open(fn) 
		img=img.resize(size)
		imgs.append(np.array(img))
	return np.array(imgs)
#path="images/american_bulldog_*.jpg"
path="train/cat.*.jpg"
# 500 x 375 -> 100 x 75 -> 20 x 15
size=(128,128)
#size=(20,15)
imgs=load_image_dir(path,size)

num=imgs.shape[0]
#imgs=imgs.reshape((num,-1))
#imgs=np.mean(imgs,axis=3)
imgs=np.reshape(imgs,[-1,128,128,3])
print(imgs.shape)
print(imgs[0].shape)
#plt.imshow(imgs[2])
#plt.show()
np.save("data.npy",imgs)


