
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加一個批次維度
    return image



import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 定義AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            for j in range(5):
                print(f"Layer {i + 1}: {x[0][1][j][4]}")  # 打印每层的输出尺寸

        x = self.avgpool(x)
        for j in range(5):
            print(f"avg: {x[0][1][j][4]}")  # 打印每层的输出尺寸

        x = torch.flatten(x, 1)
        for j in range(50):
            print(f"flatten: {x[0][j * 21 + 13]}")  # 打印每层的输出尺寸

        for i, layer in enumerate(self.classifier):
            x = layer(x)
            # 分别打印全连接层后的输出尺寸，由于Dropout不改变尺寸，所以可能只想打印Linear和ReLU层
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.ReLU):
                for j in range(30):
                    print(f"Classifer layer {i}_{j:-3d} : {x[0][j * 21 + 13]}")  # 打印每层的输出尺寸
        
        return x

alexnet_pretrained = models.alexnet(pretrained=True)  # 預訓練的模型
print(alexnet_pretrained)
my_alexnet = AlexNet(num_classes=1000)  # 自定義模型
my_alexnet.load_state_dict(alexnet_pretrained.state_dict())  # 複製權重
def readTXTfile(file_path):
    # Step 1: Read the file contents
    with open(file_path, 'r') as file:
        file_contents = file.read()
    
    # Step 2: Convert the contents to a list of floats
    pixel_values = list(map(float, file_contents.split()))
    
    # Step 3: Convert the list to a NumPy array and reshape it
    # Note: You'll need to adjust the reshape parameters based on your data's specifics and requirements
    image_array = np.array(pixel_values).reshape(3, 224, 224)  # Example shape, adjust as necessary
    
    # Step 4: Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float)
    
    # Step 5: Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor
def infer(image_path, model):
    # image = preprocess_image(image_path)
    image = readTXTfile(image_path)
    for i in range(3):
        print(image[0][1][i][14])
    model.eval()  # 確保模型處於推論模式
    with torch.no_grad():  # 不計算梯度
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# 示範如何使用
image_path = r"C:\Users\User\Desktop\MLChip\HW1\HW1\data\dog.txt"  # 替換為您的圖像路徑
prediction = infer(image_path, my_alexnet)
print("Predicted class:", prediction)