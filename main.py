import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.utils
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

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
print(f"running on {device}")


def load_data():
    """
    Load training data and split it into training and validation set
    """
    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), 'driving_dataset', 'data.txt'), names=['frame', 'steering'],
                          sep=' ')

    # yay dataframes, we can select rows and columns by their names
    # we'll store the camera images as our input data
    X = data_df[['frame']].values
    # and our steering commands as our output data
    y = data_df['steering'].values

    # now we can split the data into a training (80), testing(20), and validation set
    # thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

    return X_train, X_valid, y_train, y_valid


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
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = steering * -1.0
    return image, steering


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Lambda(lambda x: x / 127.5 - 1.0)
        # transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)]) # other source

    def __getitem__(self, index):
        image = self.samples[0][index][0]
        steering = self.samples[1][index]

        image_path = os.path.join(os.getcwd(), 'driving_dataset', image)
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        image, steering = augment(image, steering)
        # image = torch.from_numpy(np.array(image, dtype=np.int32)).long()
        image = self.transform(image)

        image = image.reshape(3, 256, 455)

        image = torch.from_numpy(image).float()  # dont understand why .float() is necessary

        return image, torch.tensor([steering])

    def __len__(self):
        return len(self.samples[0])


loss_function = nn.MSELoss()


def fwd_pass(net, X, y, train=False):
    #   X = X.view(X.size(0), 3, 70, 320)
    X = X.to(device)
    y = y.to(device)

    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)


    loss = loss_function(outputs, y.float())

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss


def train(net, batch_size=32, epochs=25):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    net.train()

    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(train_loader)):
            X = batch[0]
            y = batch[1]

            acc, loss = fwd_pass(net, X, y, train=True)

            if i % 10 == 0:
                batch_X, batch_y = next(iter(test_loader))
                test_acc, test_loss = fwd_pass(net, batch_X, batch_y)
                # f.write(f"{MODEL_NAME},{round(time.time(),4)},{round(float(acc),2)},{round(float(loss),4)},{round(float(test_acc),2)},{round(float(test_loss),4)}\n")


data = load_data()
train_set = Dataset((data[0], data[2]))

test_set = Dataset((data[1], data[3]))

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

train(net)


def main():
    print('Hi')


if __name__ == '__main__':
    main()
