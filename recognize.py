import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10  # 图片的类别数
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取网络
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)  # 第一层卷积,卷积核大小为3*3
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 设置池化层，池化核大小为2*2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)  # 第二层卷积,卷积核大小为3*3
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 第二层卷积,卷积核大小为3*3
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 分类网络
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
    # 前向传播
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# 将模型转移到GPU中（我们模型运行均在GPU中进行）
model = Model().to(device)
loss_fn    = nn.CrossEntropyLoss() # 创建损失函数
learn_rate = 1e-2 # 学习率
opt        = torch.optim.SGD(model.parameters(),lr=learn_rate)
# 训练循环
def model_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 训练集的大小，一共60000张图片
    num_batches = len(dataloader)  # 批次数目，1875（60000/32）
    train_loss, train_acc = 0, 0  # 初始化训练损失和正确率
    for X, y in dataloader:  # 获取图片及其标签
        X, y = X.to(device), y.to(device)
        # 计算预测误差
        pred = model(X)  # 网络输出
        loss = loss_fn(pred, y)  # 计算网络输出和真实值之间的差距，targets为真实值，计算二者差值即为损失
        # 反向传播
        optimizer.zero_grad()  # grad属性归零
        loss.backward()  # 反向传播
        optimizer.step()  # 每一步自动更新
        # 记录acc与loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
    train_acc /= size
    train_loss /= num_batches
    return train_acc, train_loss
def model_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 测试集的大小，一共10000张图片
    num_batches = len(dataloader)  # 批次数目，313（10000/32=312.5，向上取整）
    test_loss, test_acc = 0, 0
    # 当不进行训练时，停止梯度更新，节省计算内存消耗
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            # 计算loss
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)
            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()
    test_acc /= size
    test_loss /= num_batches
    return test_acc, test_loss
# 加载模型的状态字典
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
# 加载并分割图片
def load_and_split_image(image_path, grid_size=(4, 4)):
    image = Image.open(image_path)
    image = image.convert('RGB')  # 将图像转换为RGB格式
    image = np.array(image)
    h, w, _ = image.shape
    grid_h, grid_w = grid_size
    split_images = []
    h_step = h // grid_h
    w_step = w // grid_w
    for i in range(grid_h):
        for j in range(grid_w):
            split_images.append(image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step])
    return split_images, h_step, w_step
def binary_threshold_and_extract_blocks(image, min_size=20, max_size=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_size < w < max_size and min_size < h < max_size:
            blocks.append([x, y, x + w, y + h])
    merged_blocks = merge_overlapping_blocks(blocks)
    merged_images = []
    for mb in merged_blocks:
        x1, y1, x2, y2 = mb
        block = image[y1:y2, x1:x2]
        merged_images.append(block)
    return merged_images
def merge_overlapping_blocks(blocks, overlap_threshold = 0):
    if not blocks:
        return []
    blocks = np.array(blocks)
    merged = []
    while len(blocks) > 0:
        x1, y1, x2, y2 = blocks[0]
        others = blocks[1:]
        overlapped = [blocks[0]]
        blocks = []
        for other in others:
            ox1, oy1, ox2, oy2 = other
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            if iw * ih >= overlap_threshold * ((x2 - x1) * (y2 - y1) + (ox2 - ox1) * (oy2 - oy1)):
                overlapped.append(other)
            else:
                blocks.append(other)
        x1 = min([box[0] for box in overlapped])
        y1 = min([box[1] for box in overlapped])
        x2 = max([box[2] for box in overlapped])
        y2 = max([box[3] for box in overlapped])
        merged.append([x1, y1, x2, y2])
    return merged
# 加载并分割图片
image_path = r"C:\Users\ASUS\Desktop\tran.png"
split_images, h_step, w_step  = load_and_split_image(image_path)
# 模型预测函数
def predict_image(model, image, block_size=(32, 32)):
    image = cv2.resize(image, block_size)
    image = image / 255.0  # Normalize the image
    image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item(), torch.nn.functional.softmax(output, dim=1).max().item()
# 定义识别阈值
similarity_threshold = 0.5
# 初始化统计结果
object_counts = {i: 0 for i in range(num_classes)}
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10类别名称
# 用于存储原图尺寸的分块
labeled_images = np.zeros((4*h_step, 4*w_step, 3), dtype=np.uint8)
for i, split_image in enumerate(split_images):
    row = i // 4
    col = i % 4
    blocks = binary_threshold_and_extract_blocks(split_image)
    for block in blocks:
        x, y, w, h = cv2.boundingRect(cv2.findContours(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
        predicted_class, confidence = predict_image(model, block)
        if confidence > similarity_threshold:
            object_counts[predicted_class] += 1
            # 在分割块上绘制边界框和标签
            cv2.rectangle(split_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(split_image, class_names[predicted_class], (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    labeled_images[row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step] = split_image
# 汇总统计结果
print("识别结果统计：")
for class_id, count in object_counts.items():
    if count > 0:
        print(f"{class_names[class_id]}: {count}")
# 展示结果
plt.figure(figsize=(10, 10))
plt.imshow(labeled_images)
plt.axis('off')
plt.show()