import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = ''
transform = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
     transforms.Normalize((0.0294,), (0.0171,))])
predict_dataset = datasets.ImageFolder(root, transform=transform)
predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model = model.Net()
model.load_state_dict(torch.load('number_recognition_model.pth', map_location=device))

model.eval()

total = 0
correct = 0
with torch.no_grad():
    for data in predict_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total}%')
