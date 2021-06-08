import os
import pathlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.utils
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np




# TODO: fixe train/test splits
from config import BASE_DIR, LEARNING_RATE, BATCH_SIZE, EPOCHS

try:
    print(os.environ['COLAB_TPU_ADDR'])
    tpu = True
except:
    tpu = False

# if tpu:
#     !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl
#     import torch_xla
#     import torch_xla.core.xla_model as xm
#     device = xm.xla_device()
# elif torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")
print(f"running on {device}")


def load_data():
    """
    Load training data and split it into training and validation set
    """
    # reads CSV file into a single dataframe variable
    train_df = pd.read_csv(os.path.join(os.getcwd(), 'driving_dataset/train.txt'), names=['frame', 'steering'], sep=' ')
    test_df = pd.read_csv(os.path.join(os.getcwd(), 'driving_dataset/test.txt'), names=['frame', 'steering'], sep=' ')

    return train_df['frame'].values, train_df['steering'].values, test_df['frame'].values, test_df['steering'].values


"""
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
"""
"""
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
"""


# sources
# pytorch
# - https://github.com/ManajitPal/DeepLearningForSelfDrivingCars
# - https://github.com/milsun/AI-Driver-CNN-DeepLearning-PyTorch
# keras
# - https://github.com/llSourcell/How_to_simulate_a_self_driving_car
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)

        self.dropout = nn.Dropout(0.5)

        """ Calculate first parameter of first Linear()
            Initially image size 1, W1 = 256
            Initially image size 2, W2 = 455
            Kernel Size, k
            Stride , s
            Padding, P = 0
            For me the following formula generated the right result: O=floor((w-k+2*P)/s)
            Other source (https://www.youtube.com/watch?v=1gQR24B3ISE&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=6) generated a wrong result: O = { (W - k + 2*P)/s } + 1
        """
        self.fc1 = nn.Linear(25 * 50 * 64, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x


def augment(image, steering):
    # if np.random.rand() < 0.5:
    #     image = cv2.flip(image, 1)
    #     steering = steering * -1.0
    return image, steering


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([transforms.Lambda(lambda x: x / 127.5 - 1.0),
                                transforms.ToTensor()])
        # transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)]) # other source

    def __getitem__(self, index):
        image = self.samples[0][index]
        steering = self.samples[1][index]

        image_path = os.path.join(os.getcwd(), 'driving_dataset', image)
        # image = Image.open(image_path)
        # image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
        image = cv2.imread(image_path)

        image, steering = augment(image, steering)
        image = self.transform(image)

        image = image.reshape(3, 256, 455).float()

        # image = torch.from_numpy(image)  # dont understand why .float() is necessary
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
        return image, torch.tensor([steering])

    def __len__(self):
        return len(self.samples[0])


loss_function = nn.MSELoss()


def fwd_pass(net, X, y, train=False):
    # X = X.view(-1,3,455,256)
    X = X.to(device)
    y = y.to(device)

    if train:
        net.zero_grad()
    outputs = net(X)

    # y = y.cpu()
    # X = X.cpu()
    # outputs = outputs.cpu()
    # for i in range(len(X)):
    #     print(y[i])
    #     print(outputs[i])
    #     plt.imshow(X[i].permute(1, 2, 0))
    #     plt.show()
    # exit(0)
    # matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    loss = loss_function(outputs, y.float())

    if train:
        loss.backward()
        optimizer.step()

    return loss





def train(net, batch_size=32, epochs=25):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    log_filename = f"log-{MODEL_NAME}.txt"

    net.train()

    print(f"logfile: {log_filename}")
    with open(os.path.join(BASE_DIR, log_filename), "a") as f:
        for epoch in range(epochs):
            for i, batch in enumerate(tqdm(train_loader)):
                X = batch[0]
                y = batch[1]

                loss = fwd_pass(net, X, y, train=True)

                if i % 10 == 0:
                    batch_X, batch_y = next(iter(test_loader))
                    test_loss = fwd_pass(net, batch_X, batch_y)
                    f.write(f"{epoch},{i},{round(float(loss), 4)},{round(float(test_loss), 4)}, {optimizer.param_groups[0]['lr']}\n")
            print(f"{epoch},{i},{round(float(loss), 4)},{round(float(test_loss), 4)}, {optimizer.param_groups[0]['lr']}")
            f.flush()


data = load_data()
train_set = Dataset((data[0], data[1]))
test_set = Dataset((data[2], data[3]))
net = Net()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
MODEL_NAME = f"model-b{BATCH_SIZE}-{LEARNING_RATE}-{int(time.time())}"


def main():
    pathlib.Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

    train(net, epochs=EPOCHS, batch_size=BATCH_SIZE)

    torch.save(net.state_dict(), os.path.join(BASE_DIR, f"{MODEL_NAME}.pth"))



if __name__ == '__main__':
    main()
