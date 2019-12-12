import numpy as np
import cv2
import glob
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
import PIL.Image as Image
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def	forward(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        
    def	forward(self, input, target):
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss

def make_dataset_test(root):
    imgs=[]
    n=len(os.listdir(root))//2
    for i in range(55,82):
        if i <10:
            ind='0'+str(i)
        else:
            ind = str(i)
        #for n in range(8):
        #    for m in range(8):
        #        mask=os.path.join(root,"IDRiD_"+ind+"_"+str(n)+str(m)+".tif")
        #        img=os.path.join(root,"IDRiD_"+ind+"_"+str(n)+str(m)+".jpg")
        #        imgs.append((img,mask))
        mask=os.path.join(root,"IDRiD_"+ind+"_MA"+".tif")
        img=os.path.join(root,"IDRiD_"+ind+".jpg")
        imgs.append((img,mask))
    return imgs

def make_dataset_train(root):
    imgs=[]
    n=len(os.listdir(root))//2
    for i in range(1,55):
        if i <10:
            ind='0'+str(i)
        elif i == 43:
            continue
        else:
            ind = str(i)
        
        #for n in range(8):
        #    for m in range(8):
        #        mask=os.path.join(root,"IDRiD_"+ind+"_"+str(n)+str(m)+".tif")
        #        img=os.path.join(root,"IDRiD_"+ind+"_"+str(n)+str(m)+".jpg")
        #        imgs.append((img,mask))
        mask=os.path.join(root,"IDRiD_"+ind+"_MA"+".tif")
        img=os.path.join(root,"IDRiD_"+ind+".jpg")
        imgs.append((img,mask))
        
    return imgs

class myDataset(Dataset):
    def __init__(self, root, path, transform=None, target_transform=None):
        if path == 'train':
            imgs = make_dataset_train(root)
        else:
            imgs = make_dataset_test(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB').resize((536, 356),Image.ANTIALIAS)
        img_y = Image.open(y_path).convert('L').resize((536, 356),Image.ANTIALIAS)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=41):
    epochloss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        epochloss.append(epoch_loss/step)
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    epochloss = np.array(epochloss)
    np.save('train_loss.npy',epochloss)
    return model

def train():
    model = UNet(3, 1).to(device)
    batch_size = 1
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.99)
    mydataset = myDataset("/kaggle/input/mamode/trainma",'train', transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(mydataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

def test():
    model = UNet(3, 1)
    model.load_state_dict(torch.load('/kaggle/working/weights_40.pth',map_location='cpu'))
    mydataset = myDataset("/kaggle/input/mamode/testma", 'test', transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(mydataset, batch_size=1)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    plt.ion()
    imgs = []
    losses = []
    with torch.no_grad():
        for x, mask in dataloaders:
            y=model(x)
            loss = criterion(y, mask)
            losses.append(loss.item())
            print(loss.item())
            img_y=torch.squeeze(y).numpy()
            imgs.append(img_y)
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()
        imgs = np.array(imgs)
        losses = np.array(losses)
        np.save('test_ex.npy',imgs)
        np.save('test_loss.npy',losses)

if __name__ == '__main__':
    #参数解析
    print('ok')