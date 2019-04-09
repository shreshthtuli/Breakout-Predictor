# MIT License

# Copyright (c) 2019 Shreshth Tuli

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision as tv
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform

np.set_printoptions(threshold=np.inf)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True

def showData(sample, idx):
    print(sample['image'].shape, sample['reward'])
    # print(sample['image'])
    plt.figure()
    plt.tight_layout()
    plt.axis('off')
    img = sample['image'].numpy().reshape(144, 144) #transpose((1, 2, 0))
    print(img)
    print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.savefig("img"+str(idx)+".png")

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return img[self.x1:self.x2,self.y1:self.y2,:]

class ToTensor(object):
    def __call__(self, img):
        image = img
        # swap color axis because numpy image: H x W x C, torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class Binary(object):
    def __call__(self, img):
        img[img>0] = 1
        return img

class BreakoutDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, numepi=500):
        self.root_dir = root_dir
        self.transform = transform
        self.episodes = []
        self.numEpisodes = numepi
        self.parseAllData()
    
    def parseAllData(self):
        for episode in range(1,self.numEpisodes+1):
            x = []
            episodePath = self.root_dir+"/"+str(episode).zfill(8)+"/"
            rewards = pd.read_csv(episodePath+"rew.csv", header=None).values
            frames = os.listdir(episodePath)
            for frame in frames:
                if "rew" in frame: continue
                image = io.imread(episodePath+frame)
                image = self.transform(image)
                x.append(image)
            print(len(x)); x = torch.cat(x, 0)
            self.episodes.append([x,torch.from_numpy(rewards).float()])    

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "00000001/"+str(idx).zfill(5)+".png")
        image = io.imread(img_name)
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'reward': 0}
        return sample

class ANN(torch.nn.Module):
    
    #Our input shape is (5, 144, 144)
    def __init__(self, size):
        super(ANN, self).__init__()
        self.fc = torch.nn.Linear(5 * 144 * 144, size)
        self.sig1 = torch.nn.Sigmoid()
        self.out = torch.nn.Linear(size, 1)
        self.out_act = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(-1, 5 * 144 * 144)

        a1 = self.fc(x)
        h1 = self.sig1(a1)
        a2 = self.out(h1)
        # print(a2)
        y = self.out_act(a2)
        return y


def trainNet(net, data, batch_size, n_epochs, learning_rate):
    
    train_loader = data.episodes 
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss.to(device)

    correct = 0.0; wrong = 0.0
        
    for epoch in range(n_epochs):
        print("Epoch "+str(epoch+1))
        running_loss = 0.0
        total_train_loss = 0
        for episode in train_loader:
            x = Variable(episode[0]); y = Variable(episode[1])
            x = x.to(device); y = y.to(device)
            for i in range(len(episode[0])-9):
                inputs, labels = x[i:i+5,:,:], y[i+8]
                optimizer.zero_grad()
                
                outputs = net(inputs)
                # print("Output = "+str(outputs))
                # print("Label = "+str(labels))
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()
                
                running_loss += loss_size.data.item()
                
                # print("running_loss: "+str(running_loss))
                if epoch == n_epochs - 1:
                    pred = 1 if outputs >= 0.5 else 0
                    act = labels[0]
                    if pred == act: correct += 1
                    else: wrong += 1
                running_loss = 0.0
    
    print(correct, wrong, correct+wrong, correct/(correct+wrong))

data_transform = transforms.Compose([
        Crop(32, 176, 8, 152),
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        Binary(),
    ])
    
print("Parsing Data")
Data = BreakoutDataset("", "./train_dataset", data_transform, 1)

print(len(Data.episodes), len(Data.episodes[0][0]), Data.episodes[0][0][0].shape)
print(type(Data.episodes), type(Data.episodes[0]), type(Data.episodes[0][0]), type(Data.episodes[0][0][0]))

# dataPoint = Data.__getitem__(1283)
# showData(dataPoint, 1283)

print("Learning NN")

nn = ANN(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
nn.cuda()

trainNet(nn, Data, batch_size=32, n_epochs=5, learning_rate=0.01)

torch.save(nn.state_dict(), "./ann.pt")