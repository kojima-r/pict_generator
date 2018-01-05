import glob
from PIL import Image
import numpy as np
from matplotlib import pylab as plt


def load_image_dir(path,size):
	l=glob.glob(path)
	imgs=[]
	for fn in l:
		#print(fn)
		img=Image.open(fn) 
		img=img.resize(size)
		a=np.array(img)
		if a.shape[2]!=3:
			print(a.shape)
			print(fn)
		else:
			imgs.append(a)

	print(len(imgs))
	res=np.array(imgs)
	print(res.shape)
	return res
#path="images/american_bulldog_*.jpg"
path="./animeface-character-dataset/thumb/**/*.png"
# 500 x 375 -> 100 x 75 -> 20 x 15
size=(128,128)
#size=(20,15)
img_set=load_image_dir(path,size)

num=img_set.shape[0]
#imgs=imgs.reshape((num,-1))
#imgs=np.mean(imgs,axis=3)
#imgs=np.reshape(imgs,[-1,128,128,1])
print(img_set.shape)
print(img_set[0].shape)
#plt.imshow(imgs[2])
#plt.show()
np.save("anime.npy",img_set)


