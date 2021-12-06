import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

path_train = './train'
path_test = './test'
learning_rate = 0.001
training_epoch = 15
batch_size = 128

time.perf_counter()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(1),
                                transforms.Normalize((0.5,), (0.5,))])
train_data = dsets.ImageFolder(root=path_train, transform=transform)
test_data = dsets.ImageFolder(root=path_test, transform=transform)

# change num_workers to 0 if windows

train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                shuffle=True, drop_last=True, num_workers=0)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.6
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(7 * 7 * 64, 128, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer5 = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob))
        self.fc2 = nn.Linear(128, 26, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc2(out)

        return out


model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_data_loader)
torch.backends.cudnn.benchmark = True

for epoch in range(training_epoch):
    avg_cost = 0

    for batch_idx, samples in enumerate(train_data_loader):

        image, labels = samples
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        hypothesis = model(image)
        cost = criterion(hypothesis, labels)
        cost.backward()
        optimizer.step()

        if batch_idx % batch_size == 0:
            print(' -Batch {0}: cost is {1}'.format(batch_idx, cost))

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
print('Learning finished')

model.eval()
result = 0
number = 0
with torch.no_grad():
    for _, data in enumerate(test_data_loader):
        X_test, Y_test = data
        X_test = X_test.view(len(X_test), 1, 28, 28).float().to(device)
        Y_test = Y_test.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        number += correct_prediction.size(0)
        result += correct_prediction.int().sum()

    print('Accuracy:', 100.0*result.item()/number)
print('Elapsed time:', time.perf_counter())
