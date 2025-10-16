import numpy as np
import torch
import math
from PIL import Image
import os
from natsort import natsorted
from torchsummary import summary

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = torch.nn.Conv2d(1, 8, 3)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(8, 8, 3)
        self.pool2 = torch.nn.MaxPool2d(5)
        self.conv3 = torch.nn.Conv2d(8, 8, 5)
        self.pool3 = torch.nn.MaxPool2d(5)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(200, 200)
        self.lstm = torch.nn.LSTM(200,200,1)
        self.unflatten = torch.nn.Unflatten(1,(8,5,5))
        self.upsample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4 = torch.nn.Conv2d(8,16,3)
        self.conv5 = torch.nn.Conv2d(16,16,3,padding = 1)
        self.conv6 = torch.nn.Conv2d(16,16,3, padding = 1)
        self.conv7 = torch.nn.Conv2d(16,8,3, padding = 1)
        self.conv8 = torch.nn.Conv2d(8,8,3, padding = 1)
        self.conv9 = torch.nn.Conv2d(8,4,3, padding = 1)
        self.conv10 = torch.nn.Conv2d(4,1,13)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation(x)
        x,(h,c) = self.lstm(x, (torch.zeros(1,200),torch.zeros(1,200))) 
        x = self.unflatten(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.upsample1(x)
        x = self.conv6(x)
        x = self.upsample1(x)
        x = self.conv7(x)
        x = self.upsample1(x)
        x = self.conv8(x)
        x = self.upsample1(x)
        x = self.conv9(x)
        x = self.upsample1(x)
        x = self.conv10(x)
        return x




#define model
model1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 8, 3),
    torch.nn.MaxPool2d(3),
    torch.nn.Conv2d(8, 8, 3),
    torch.nn.MaxPool2d(5),
    torch.nn.Conv2d(8, 8, 5),
    torch.nn.MaxPool2d(5),
    torch.nn.Flatten(),
    torch.nn.Linear(200, 200),
    torch.nn.LSTM(200,300,1),
    torch.nn.Flatten(),
    torch.nn.Unflatten(1,(8,5,5)),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(8,16,3),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(16,16,3,padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(16,16,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(16,8,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(8,8,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(8,4,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(4,1,13)
    #torch.nn.Conv2d(4,1,71),
    #torch.nn.Conv2d(1,1,71),
    )

model = Model()
if os.path.exists("model_weights.pth"):
    model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
#print(summary(model, input_size = (1,500,500), batch_size = -1))
# encoder: convo, downsample (maxpool), convo, downsample..., flatten 
# rnn: lstm
# decoder: reshape to 2d img, upsample, convo, upsample, 
loss_fn = torch.nn.MSELoss(reduction='mean')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#go through a random selection of images and run model on each in order, reset directory and lstm hidden state vectors after
os.chdir("embryo_dataset")

embryo_vids = os.listdir()
np.random.shuffle(embryo_vids)
select_amount = 0.01
batch_size = 50
embryo_vids = embryo_vids[:int(len(embryo_vids)*select_amount)]
for i in embryo_vids:
    PATH = "./../model_weights.pth"
    torch.save(model.state_dict(), PATH)
    print(os.getcwd())
    os.chdir("./"+i)

    print(os.getcwd())
    images = os.listdir()
    try:
        np.array([Image.open(img) for img in images])
    except OSError:
        os.chdir("./..")
        continue
    images = natsorted(images)
    images = [img for img in images if not os.path.isdir(img)]
    for k in range(len(images)//batch_size):
        x = torch.tensor(np.array([Image.open(img) for img in images[k*batch_size:min(len(images)-1,(k+1)*batch_size)]]), dtype=torch.float32).reshape((-1,1,500,500))
        y = x.clone()
        print(x.shape)
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        print(loss)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    os.chdir("./..")
    print(os.getcwd())

os.chdir("./..")

