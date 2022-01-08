import torch
from torch import  nn

image_size = 784

class D(nn.Module):
    def __init__(self, hidden):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden,hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x.reshape(x.size(0),28*28)
        return self.main(x)

class G(nn.Module):
    def __init__(self, door, hidden):
        super(G, self).__init__()
        self.door = door
        self.main = nn.Sequential(
            nn.ConvTranspose2d(door, 16, 10, 1, 0, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,4 , 10, 1, 0, bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, 10, 1, 0, bias=False),
            nn.Tanh()
        )
    def forward(self,x):
        x = x.reshape(x.size(0),self.door,1,1)
        x = self.main(x)
        # print(x.shape)
        x = x.reshape(x.size(0),28,28)
        return x