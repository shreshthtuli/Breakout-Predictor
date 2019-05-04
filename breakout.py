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
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision as tv
import pandas as pd
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
from itertools import combinations
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import pickle
from torchsummary  import summary
from sys import argv
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from svmutil import *

np.set_printoptions(threshold=np.inf)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True

Global_loss = []

def applyPCA(lst):
    res = []
    pca = PCA(n_components=50)
    for x in lst:
        x = x.numpy().reshape(144, 144)
        pca.fit(x)
        y = pca.transform(x)
        res.append(y)
    return res

def sample_svm(x, prob, rewards):
    data = []; ys = []; low = 0; high = 0;
    print("Frames:",len(x),)
    for i in range(len(x)-9):
        if rewards[i+7] == 1:
            ys.append([1]); high += 1
            seq = *random.choice(list(combinations(x[i:i+6], 4))),x[i+6]
            data.append(np.concatenate(applyPCA(seq), 0).reshape(720*50))
        elif random.randint(1,100) <= 100*prob:
            ys.append([0]); low += 1
            seq = *random.choice(list(combinations(x[i:i+6], 4))),x[i+6]
            data.append(np.concatenate(applyPCA(seq), 0).reshape(720*50))
    print("Positives:",high,"Negatives:",low)
    return np.asarray(data), np.asarray(ys)

def genSVMtrain():
    numEpisodes = 50
    root_dir = "./train_dataset"
    inputs = []; outputs = []
    for episode in range(1,numEpisodes+1):
        x = []
        episodePath = root_dir+"/"+str(episode).zfill(8)+"/"
        rewards = pd.read_csv(episodePath+"rew.csv", header=None).values
        rewards = np.insert(rewards, 0, 0)
        frames = os.listdir(episodePath)
        for frame in frames:
            if "rew" in frame: continue
            image = io.imread(episodePath+frame)
            image = data_transform_grayscale(image)
            x.append(image)
        print("Episode num:", episode,)
        x, y = sample_svm(x, 0.03, rewards)
        print(x.shape,y.shape)
        inputs.extend(x)
        outputs.extend(y)
    inputs = np.asarray(inputs); outputs = np.asarray(outputs)
    return inputs, outputs

def genSVMtest():
    root_dir = "./validation_dataset"
    rewards = pd.read_csv(root_dir+"/rewards.csv", header=None).values
    rewards = rewards[0:1000,1]
    inputs = []
    for episode in range(0, 1000): # 11600
        x = []
        episodePath = root_dir+"/"+str(episode).zfill(8)+"/"
        frames = os.listdir(episodePath)
        for frame in frames:
            image = io.imread(episodePath+frame)
            image = data_transform_grayscale(image)
            x.append(image)
        if episode%100 == 0: print("Episode num:", episode)
        inputs.append(np.concatenate(applyPCA(x), 0).reshape(720*50))
    inputs = np.asarray(inputs);
    return inputs, rewards

def listify(X, Y):
    retx = []
    rety = []
    for i in range(Y.shape[0]):
        rety.append(int(Y.item(i)))
    for i in range(X.shape[0]):
        param = []
        for j in range(X.shape[1]):
            param.append(X.item(i,j))
        retx.append(param)
    return retx, rety

def showData(input, output, idx, folder):
    print(input.shape, output.shape)

    plt.figure(); plt.tight_layout(); plt.axis('off')
    for i in range(5):
        img = input[i]
        if argv[2] != "Binary":
            img = input[3*i:3*i+3]
            img = img.numpy()
            img = img.transpose((1, 2, 0))
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.savefig(folder+"img"+str(idx)+str(i)+".png")
    print("Output:", output)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
def load_object(filename):
    with open(filename, 'rb') as input:
        dataset = pickle.load(input)
        return dataset

def cleardir(filename):
    files = glob.glob(filename)
    for f in files:
        os.remove(f)

def sample(x, prob, rewards):
    data = []; ys = []; low = 0; high = 0;
    print("Frames:",len(x),)
    for i in range(len(x)-9):
        if rewards[i+7] == 1:
            ys.append([1]); high += 1
            seq = *random.choice(list(combinations(x[i:i+6], 4))),x[i+6]
            data.append(torch.cat(seq, 0))
        elif random.randint(1,100) <= 100*prob:
            ys.append([0]); low += 1
            seq = *random.choice(list(combinations(x[i:i+6], 4))),x[i+6]
            data.append(torch.cat(seq, 0))
    print("Positives:",high,"Negatives:",low)
    return data, torch.from_numpy(np.asarray(ys)).long()

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
    def __init__(self, csv_file, root_dir, transform=None, numepi=500, load=False, start=1):
        self.root_dir = root_dir
        self.transform = transform
        self.inputs = []
        self.rewards = []
        self.numEpisodes = numepi
        self.datasize = 0
        self.parseAllData(start)
    
    def parseAllData(self, start):
        for episode in range(start,self.numEpisodes+1):
            x = []
            episodePath = self.root_dir+"/"+str(episode).zfill(8)+"/"
            rewards = pd.read_csv(episodePath+"rew.csv", header=None).values
            rewards = np.insert(rewards, 0, 0)
            frames = os.listdir(episodePath)
            for frame in frames:
                if "rew" in frame: continue
                image = io.imread(episodePath+frame)
                image = self.transform(image)
                x.append(image)
            print("Episode num:", episode,)
            x, y = sample(x, 0.03, rewards)
            self.datasize += len(x)
            self.inputs.extend(x)
            self.rewards.extend(y)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "00000001/"+str(idx).zfill(5)+".png")
        image = io.imread(img_name)
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'reward': 0}
        return sample
    

class BreakoutTestset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, numepi=11600, load=True):
        self.root_dir = root_dir
        self.transform = transform
        self.inputs = []
        self.rewards = []
        self.numEpisodes = numepi
        self.datasize = 0
        self.parseAllData(load)
    
    def parseAllData(self, load):
        if load:
            rewards = pd.read_csv(self.root_dir+"/rewards.csv", header=None).values
            self.rewards = torch.from_numpy(rewards[:,1])
        for episode in range(0,self.numEpisodes):
            x = []
            episodePath = self.root_dir+"/"+str(episode).zfill(8)+"/"
            frames = os.listdir(episodePath)
            for frame in frames:
                image = io.imread(episodePath+frame)
                image = self.transform(image)
                x.append(image)
            if episode%100 == 0: print("Episode num:", episode)
            self.datasize += 1
            self.inputs.append(torch.cat(x, 0))
        

class CNN(torch.nn.Module):
    #Our input shape is (15, 144, 144)
    def __init__(self, size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(size, 32, kernel_size=3, stride=2, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 9 * 9, 2048)
        self.out = torch.nn.Linear(2048, 2)
        self.out_act = torch.nn.LogSoftmax()
        
    def forward(self, x):
        x = F.relu(self.conv1(x)) # 15x144x144 -> 128x144x144
        x = self.pool1(x) # 128x144x144 -> 128x72x72
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 9 * 9) 
        x = F.relu(self.fc1(x))
        y = self.out_act(self.out(x))
        return y

class CompNN(torch.nn.Module):  
    #Our input shape is (15, 144, 144)
    def __init__(self, size):
        super(CompNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(size, 48, kernel_size=3, stride=1, padding=1) # Earlier it was 32 for first two
        self.conv2 = torch.nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 36 * 36, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.out = torch.nn.Linear(2048, 2)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.out_act = torch.nn.LogSoftmax()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 36 * 36) 
        x = self.dropout(F.relu(self.fc1(x))) # Dropout
        x = self.dropout(F.relu(self.fc2(x)))
        # x = F.relu(self.fc1(x)) # Earlier with dropout
        # x = F.relu(self.fc2(x))
        y = self.out_act(self.out(x))
        return y


def trainNet(net, data, batch_size, n_epochs, learning_rate, filename, size):
    global Global_loss
    # loss = torch.nn.NLLLoss() # Use this for CompNN
    loss = torch.nn.CrossEntropyLoss() # Use this for resnet
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss.to(device)
    epoch_avg = 0.0; counts = 0

    for epoch in range(n_epochs):
        print("Epoch "+str(epoch+1))
        running_loss = 0.0
        for i in range(data.datasize):
            x = torch.reshape(data.inputs[i], (1,size,144,144))
            x = Variable(x); y = torch.autograd.Variable(data.rewards[i].reshape(1).long()) # Use this for validation training
            # x = Variable(x); y = torch.autograd.Variable(data.rewards[i])
            inputs, labels = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            # print(outputs, labels, loss_size.item())
            loss_size.backward()
            optimizer.step() 
            running_loss += loss_size.item()
            if i % 500 == 499:    # print every 500 mini-batches
                # print(outputs, labels, loss_size.item())
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss))
                Global_loss.append(running_loss); epoch_avg += running_loss; counts += 1
                running_loss = 0.0
                plt.clf(); plt.plot(Global_loss); plt.savefig("Loss_with_time.png")
        print("Epoch average =", epoch_avg/counts)
        epoch_avg = 0.0; counts = 0
        torch.save(net.state_dict(), filename) 

def testNet(net, data, size):
    pred = []; act = []
    print(data.inputs[0].shape, data.rewards.shape)
    print(data.rewards)
    for i in range(data.datasize):
        x = torch.reshape(data.inputs[i], (1,size,144,144))
        x = Variable(x); y = Variable(data.rewards[i])
        inputs, labels = x.to(device), y.to(device)
                
        outputs = net(inputs)
        p = 1 if outputs[0,1] >= outputs[0,0] else 0
        print(i,outputs,p,labels.item())
        pred.append(float(p))
        act.append(labels.item())

    print("Accuracy : ", accuracy_score(act, pred))
    print("Macro F1 Score : ", f1_score(act, pred, average='macro'))
    cm = confusion_matrix(act, pred)
    print(cm)


def predict(net, data, size):
    print(data.inputs[0].shape)
    for i in range(data.datasize):
        x = torch.reshape(data.inputs[i], (1,size,144,144))
        outputs = net(Variable(x).to(device))
        # print(i, outputs[0,1], outputs[0,0])
        p = 1 if outputs[0,1] >= outputs[0,0] else 0
        print(str(i)+","+str(p))

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    model_ft = models.resnet18(pretrained=use_pretrained)
    # set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    model_ft.conv1 = torch.nn.Conv2d(15, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
    input_size = 15
    return model_ft, input_size

data_transform = transforms.Compose([
        Crop(32, 176, 8, 152),
        transforms.ToTensor(),
    ])

data_transform_grayscale = transforms.Compose([
        Crop(32, 176, 8, 152),
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

data_transform_binary = transforms.Compose([
        Crop(32, 176, 8, 152),
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        Binary(),
    ])
    
print("Argument:", argv[1])
if argv[1] == '0':
    print("Parsing Data for training")
    # Data = BreakoutDataset("", "./train_dataset", data_transform, 200)
    # save_object(Data, "datasets/dataset.pkl")  
    # Data2 = BreakoutDataset("", "./train_dataset", data_transform, 400, start=201)
    # save_object(Data2, "datasets/dataset2.pkl")  
    Data3 = BreakoutDataset("", "./train_dataset", data_transform, 500, start=401)
    save_object(Data3, "datasets/dataset3.pkl")  

if argv[1] == '1':
    print("Parsing Data for validation")
    Data = BreakoutTestset("", "./validation_dataset", data_transform, 11600)
    save_object(Data, "datasets/validation.pkl")   

if argv[1] == '2':
    print("Parsing Data for test")
    Data = BreakoutTestset("", "./test_dataset", data_transform, 30910, False)
    save_object(Data, "datasets/test.pkl")   

elif argv[1] == '3' or argv[1] == '4':
    print("Loading training dataset")  
    Data = load_object("datasets/dataset3.pkl")
    # print("Loading test dataset")  
    # Test = load_object("datasets/test.pkl")

    size = 15
    if argv[1] == '3':
        print("Learning CNN")
        nn = CNN(size)
        summary(nn, (size, 144, 144))
        filename = "models/cnn.pt"
        if os.path.isfile(filename):
            nn.load_state_dict(torch.load(filename))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        nn.cuda()
        trainNet(nn, Data, batch_size=32, n_epochs=10, learning_rate=0.000001, filename=filename, size=size)
        # testNet(nn, Test, size)

    else:
        print("Learning Competition NN")
        nn = CompNN(size)
        summary(nn, (size, 144, 144))
        filename = "models/compnn.pt"
        if os.path.isfile(filename):
            nn.load_state_dict(torch.load(filename))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        nn.cuda()
        trainNet(nn, Data, batch_size=32, n_epochs=4, learning_rate=0.000001, filename=filename, size=size)
        # testNet(nn, Test, size)

elif argv[1] == '5':
    print("Loading input dataset")  
    Val = load_object("datasets/dataset.pkl")
    idx = 0
    cleardir("negatives/*"); cleardir("positives/*")
    for i in range(1000):
        if Val.rewards[i] == 1:
            showData(Val.inputs[i], Val.rewards[i], idx, "positives/")
            idx+=1
        else:
            showData(Val.inputs[i], Val.rewards[i], idx, "negatives/")
            idx+=1

elif argv[1] == '6':
    print("SVM and PCA - generating datasets")

    inputs, outputs = genSVMtrain()
    print(inputs.shape, outputs.shape)
    np.save("datasets/trainx.npy", inputs)
    np.save("datasets/trainy.npy", outputs)

    inputs, outputs = genSVMtest()
    print(inputs.shape, outputs.shape)
    np.save("datasets/testx.npy", inputs)
    np.save("datasets/testy.npy", outputs)

elif argv[1] == '7':
    print("SVM and PCA")

    trainx = np.load("datasets/trainx.npy")
    trainy = np.load("datasets/trainy.npy")
    testx = np.load("datasets/testx.npy")
    testy = np.load("datasets/testy.npy")

    trainy = trainy.ravel()
    print(trainx.shape)
    print(trainy.shape)
    print(testx.shape)
    print(testy.shape)

    svm = SVC(kernel="rbf",gamma=0.05,C=1)

    filename = "models/svm-g-0.05-c-1.pkl"
    if not os.path.isfile(filename):
        svm.fit(trainx, trainy)
        print("Model trained with g 0.05 c 1")
        pickle.dump(svm, open(filename, 'wb'))
    else:
        svm = pickle.load(open(filename, 'rb'))

    print("Training Accuracy: ", svm.score(trainx, trainy))
    print("Test Accuracy = ", svm.score(testx, testy))

elif argv[1] == '10':
    print("Loading test dataset")  
    Test = load_object("datasets/test.pkl")

    size = 15
    print("Predicting CNN")
    nn = CompNN(size)
    summary(nn, (size, 144, 144))
    filename = "models/compnn.pt"
    nn.load_state_dict(torch.load(filename))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    nn.cuda()
    predict(nn, Test, size)


elif argv[1] == '11':
    nn, input_size = initialize_model("resnet", 2, True, True)
    print(nn)
    print("Loading training dataset")  
    Data = load_object("datasets/dataset3.pkl")
    print("Learning Resnet")
    summary(nn, (15, 144, 144))
    filename = "models/resnet18.pt"
    if os.path.isfile(filename):
        nn.load_state_dict(torch.load(filename))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    nn.cuda()
    trainNet(nn, Data, batch_size=32, n_epochs=5, learning_rate=0.00001, filename=filename, size=15)

elif argv[1] == '12':
    print("Predicting CNN")
    nn, size = initialize_model("resnet", 2, True, True)
    summary(nn, (size, 144, 144))
    filename = "models/resnet18.pt"
    nn.load_state_dict(torch.load(filename))
    print("Loading test dataset")  
    Test = load_object("datasets/test.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    nn.cuda()
    predict(nn, Test, size)