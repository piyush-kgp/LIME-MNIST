

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
import numpy as np
import seaborn as sns

NUM_EPOCHS_TO_TRAIN = 5
BATCH_SIZE = 32
LR = 1e-3
TAU = 0.5

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=5408, out_features=10, bias=False)
    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        y = self.flatten(y)
        y = self.fc(y)
        return y

mnist_train = datasets.mnist.MNIST(root='mnist/', download=True, train=True, \
                transform=transforms.ToTensor())
mnist_test = datasets.mnist.MNIST(root='mnist/', download=True, train=False, \
                transform=transforms.ToTensor())
train_dl = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

model = MNIST_CNN()
model.load_state_dict(torch.load('ckpt/model_E0.ckpt'))

test_data = torch.empty(test_dl.dataset.__len__(), 1, 28, 28)
test_labels = torch.empty(test_dl.dataset.__len__(), dtype=torch.int)
prob_mat = torch.empty(test_dl.dataset.__len__(),10)
for batch_num, (batch_x, batch_y) in enumerate(test_dl):
    test_data[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE] = batch_x
    test_labels[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE] = batch_y
    prob_mat[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE] = model(batch_x)

test_data = test_data.squeeze(1).view(-1,28*28).numpy()
mu, sigma = test_data.mean(), test_data.std()
test_data = (test_data-mu)/sigma
test_labels = test_labels.numpy()
prob_mat = prob_mat.detach().numpy()

im_ids_to_explain = np.random.randint(0, test_dl.dataset.__len__(), 10)

for im_id in im_ids_to_explain:
    break

X_h = test_data[im_id]
im1 = (255*(X_h*sigma+mu).reshape((28,28))).astype(np.uint8)

w = np.exp(-np.linalg.norm(test_data-X_h,axis=1)**2/(2*TAU**2))
X_dash = test_data * np.sqrt(w)[:,np.newaxis]
Y = prob_mat[:,test_labels[im_id]]
Y_dash = np.sqrt(w)*Y

theta = ((X_dash.T.dot(X_dash))**-1).dot(X_dash.T).dot(Y_dash)
theta = (theta-theta.mean())/theta.std()
theta = (theta-theta.min())/(theta.max()-theta.min())
im2 = (255*theta.reshape((28,28))).astype(np.uint8)

Image.fromarray(im1).save('images/E_{}_{}_orig.jpg'.format(epoch, im_id))
Image.fromarray(im2).save('images/E_{}_{}_lime.jpg'.format(epoch, im_id))
