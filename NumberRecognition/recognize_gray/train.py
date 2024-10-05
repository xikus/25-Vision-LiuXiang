import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import model

batch_size = 256
lr = 0.04
momentum = 0.5
epoch = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([transforms.Resize((28, 28)),transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.0294,), (0.0171,))])
train_dataset = datasets.ImageFolder(
    root='/home/yoda/25-Vision-LiuXiang/NumberRecognition/number_recognition_trainset/gray/trainDataSet',
    transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dataset = datasets.ImageFolder(
    root='/home/yoda/25-Vision-LiuXiang/NumberRecognition/number_recognition_trainset/gray/valDataSet',
    transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = model.Net()
model.to(device)

# Construct loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

losses_per_epoch = []

def train(i):
    model.train()
    total_loss = 0
    losses = []
    for images, target in train_loader:
        images, target = images.to(device), target.to(device)
        outputs = model(images)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        losses.append(loss.item())
    plt.plot(losses, label=f'Epoch {i + 1}')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.title('Loss Change During an Epoch')
    plt.legend()
    plt.show()
    average_loss = total_loss / len(train_loader)
    losses_per_epoch.append(average_loss)
    print(f'Epoch: [{i + 1}], Loss: {average_loss:.4f}')
    return losses


def test(i):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, target in test_loader:
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('[%d / %d]: %.2f %% ' % (i + 1, epoch, 100 * correct / total))
    print(total)


# Start train and Test
print('Training started...')
train_times = []
for i in range(epoch):
    start_time = time.time()
    train_losses = train(i)
    test(i)
    end_time = time.time()
    # 绘制单个epoch的损失变化图

    train_times.append(end_time - start_time)
    print(f'Epoch {i+1} finished in {train_times[-1]:.2f} seconds')


plt.plot(losses_per_epoch, label='Average Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.show()
# Print average training time
print(f'Average training time per epoch: {sum(train_times) / len(train_times):.2f} seconds')

# Save model
# torch.save(model.state_dict(), 'number_recognition_model.pth')
#打出参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')