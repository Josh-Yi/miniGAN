import math
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torchvision.utils import save_image

sample_dir = 'samples'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

door = 64
hidden_size = 256
image_size = 784  # 28*28
num_epoch = 200
batch_size = 100

import torchvision

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)  # both (0.5,0.5,0.5) 3 Channel
    ]
)

mnist = torchvision.datasets.MNIST(root='MNIST_DATA', train=False, download=True, transform=transform)


def show_me_sth():
    x = 1
    for i in range(20, 40):
        plt.subplot(4, 5, x)
        x += 1
        plt.imshow(mnist[i][0].reshape(28, 28))
        plt.title(mnist[i][1])
        plt.axis('off')
    plt.show()


# show_me_sth()

loader = DataLoader(dataset=mnist,
                    batch_size=batch_size,
                    shuffle=True)

from model import D, G

D = D(hidden=hidden_size)
G = G(door=door, hidden=hidden_size)

G = G.to(device)
D = D.to(device)

criterion = nn.BCELoss()

D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


def denorm(x):
    out = (x + 1) / 2
    return torch.clamp(out, min=0, max=1)


def reset_grad():
    D_optimizer.zero_grad()
    G_optimizer.zero_grad()


# Train
print(len(loader), 'batch')
for epoch in range(num_epoch):
    for i, item in enumerate(tqdm(loader)):
        img, _ = item  # img: 28*28
        img = img.reshape(batch_size, -1).to(device)  # batch_size*784
        real_label = torch.ones(batch_size, 1).to(device)
        fake_label = torch.zeros((batch_size, 1)).to(device)

        # Train D
        # See real
        outputs = D(img)
        D_Loss_Real = criterion(outputs, real_label)
        real_score = outputs
        # See fake
        z = torch.randn((batch_size, door)).to(device)
        fake_img = G(z)
        outputs = D(fake_img)
        D_Loss_Fake = criterion(outputs, fake_label)
        fake_score = outputs
        # BackProp
        D_Loss_Total = D_Loss_Fake + D_Loss_Real
        reset_grad()
        D_Loss_Total.backward()
        D_optimizer.step()

        # Train G
        # z = torch.randn(batch_size, door).to(device)
        fake_img = G(z)
        outputs = D(fake_img)

        G_Loss = criterion(outputs, real_label)

        reset_grad()
        G_Loss.backward()
        G_optimizer.step()

        if (i) % 100 == 0:
            print('Epoch[{}/{}], Step[{}/{}], D_Loss: {:.4f}, G_Loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                epoch, num_epoch, i + 1, len(loader), D_Loss_Total.item(), G_Loss.item(), real_score.mean().item(),
                fake_score.mean().item()
            ))

    if (epoch + 1) == 1:
        images = img.reshape(img.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    fake_images = fake_img.reshape(fake_img.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images_{}.png'.format(epoch + 1)))

torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
