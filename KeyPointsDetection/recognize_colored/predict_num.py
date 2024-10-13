import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import model_num
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# root = ''
transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor(),
     transforms.Normalize((0.3000, 0.3020, 0.4224), (0.2261, 0.2384, 0.2214))])
# predict_dataset = datasets.ImageFolder(root, transform=transform)
# predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model_num = model_num.Net()
model_num.load_state_dict(torch.load('number_recognition_model.pth', map_location=device, weights_only=True))
model_num.eval()


# total = 0
# correct = 0
# with torch.no_grad():
#     for data in predict_loader:
#         inputs, labels = data
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f'Accuracy of the model on the test images: {100 * correct / total}%')

def predict_number(image):
    image = transform(image)
    image = image.unsqueeze(0)
    outputs = model_num(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()
