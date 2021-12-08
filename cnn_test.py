import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from cnn_model import CNN


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)

path_test = './test'
path_model = './model'
batch_size = 128

start = time.perf_counter()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Grayscale(1),
                                transforms.Normalize((0.5,), (0.5,))])
test_data = dsets.ImageFolder(root=path_test, transform=transform)

# change num_workers to 0 if windows

test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)
model = CNN()
model.load_state_dict(torch.load(path_model))
model.to(device)
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
print('Elapsed time:', time.perf_counter() - start)
model.train()
