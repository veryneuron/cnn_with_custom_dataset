import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn

from cnn_model import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

path_train = './train'
path_model = './model'
learning_rate = 0.001
training_epoch = 20
batch_size = 128

start = time.perf_counter()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(1),
                                transforms.Normalize((0.5,), (0.5,))])
train_data = dsets.ImageFolder(root=path_train, transform=transform)

# change num_workers to 0 if windows

train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                shuffle=True, drop_last=True, num_workers=0)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_data_loader)

# train model

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
print('Learning finished, save parameters file at', path_model)
print('Elapsed time:', time.perf_counter() - start)

torch.save(model.state_dict(), path_model)