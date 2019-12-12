import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re



def split_images(getpath,savepath,num_split):
    subw = 4288//num_split
    subh = 2842//num_split + 1
    img = Image.open(getpath)
    pattern1 = re.compile(r'IDRiD_(\d){2}')
    pattern2 = re.compile(r'[.][a-zA-Z]+')
    name = re.search(pattern1,getpath).group()
    suffix = re.search(pattern2,getpath).group()
    for i in range(num_split):
        for j in range(num_split):
            left = j*subw
            upper = i*subh
            right = left + subw
            lower = upper + subh
            subimg = img.crop((left,upper,right,lower))
            subimg.save(savepath+name+'_'+str(i)+str(j)+suffix)

# just to merge subimages to the a one big image
# in this case, the final image is 4288*2842
def merge_slice(getpath,savepath,num_slice):
    column = num_slice
    row = num_slice
    IMAGE_H = 356
    IMAGE_W = 536
    image = Image.new('RGB',(column*IMAGE_W,row*IMAGE_H))
    for i in range(row):
        for j in range(column):
            subimg = Image.open(getpath+'_'+str(i)+str(j)+'.jpg')
            image.paste(subimg,j*IMAGE_W,i*IMAGE_H)
    image.save(savepath)

def merge_npy(img,seq):
    temp = []
    for n in range(8):
        temp.append(np.hstack((img[n] for n in range(n*8,(n+1)*8))))
    img = np.vstack((m for m in temp))
    cop = img.copy()
    cv2.normalize(img,cop,0,255,cv2.NORM_MINMAX) 
    cop = cop.astype(np.int)
    cv2.imwrite('testcombine/'+str(seq)+'.tif',cop)
    img = 0

if __name__ == '__main__':
    # split the original images
#    paths = glob.glob(r'train/*.jpg')
#    for n in paths:
#        split_images(n,'cut_train/',8)
#    paths = glob.glob(r'train/*.tif')
#    for m in paths:
#        split_images(m,'cut_train/',8)
    images = np.load(r'5/test_ex.npy')
    size = len(images)//64
    for i in range(size):
        img = images[i*64:(i+1)*64]
        merge_npy(img,i)
        