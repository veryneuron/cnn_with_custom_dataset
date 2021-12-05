import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5,0.5),(0.5,0.5,0.5)])
image_train = dsets.ImageFolder(root=path_train, transform=transform)
image_test = dsets.ImageFolder(root=path_test, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=image_train, batch_size=batch_size,
                                          shuffle=True, drop_last=True)
data_loader_test = torch.utils.data.DataLoader(dataset=image_test, batch_size=batch_size,
                                          shuffle=True, drop_last=True)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)

for epoch in range(training_epoch):
    avg_cost = 0

    for num, data in enumerate(data_loader):
        inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

        optimizer.zero_grad()
        hypothesis = model(inputs)
        cost = criterion(hypothesis, labels)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
print('Learning finished')

with torch.no_grad():
    for num, data in enumerate(data_loader_test):
        inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

        prediction = model(inputs)
        correct_prediction = torch.argmax(prediction, 1) == labels
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
        print(time.perf_counter())