import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理，将图像转换为张量
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 将图像调整为28x28
    transforms.Grayscale(num_output_channels=1),  # 确保图像是单通道
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 加载数据集
dataset = datasets.ImageFolder(root='number_recognition_trainset/gray/trainDataSet', transform=transform)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)


# 定义一个函数来计算均值和标准差
def compute_mean_std(loader):
    # 初始化求和变量和计数器
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        # batch_size（当前批次的图像数量）
        # print(images.shape)
        batch_samples = images.size(0)

        # 计算批次图像的均值和标准差
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    # 计算总体的均值和标准差
    mean /= total_images_count
    std /= total_images_count

    return mean, std


# 计算均值和标准差
mean, std = compute_mean_std(dataloader)

print(f"Mean of the dataset: {mean}")
print(f"Std dev of the dataset: {std}")
