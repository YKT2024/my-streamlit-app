# 必要なライブラリをインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # tqdm をインポート

# 1. デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. データの準備と分割
transform = transforms.Compose([
    transforms.Resize((630, 630)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = "C:/Users/E.ykt/Desktop/1009_自走課題/Tryna_use_Mickey/train"
dataset = datasets.ImageFolder(data_path, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. モデルの定義
class ImageAuthClassifier(nn.Module):
    def __init__(self):
        super(ImageAuthClassifier, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)  # 2クラス分類に変更

    def forward(self, x):
        return self.resnet(x)

model = ImageAuthClassifier().to(device)

# 4. 損失関数と最適化の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 学習と検証の関数
def train(model, train_loader, val_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm を使った進捗バー付きのループ
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

    print("学習が完了しました。")
    return model

# 6. 学習の実行
trained_model = train(model, train_loader, val_loader, num_epochs=20)

# 7. 学習済みモデルの保存
torch.save(trained_model.state_dict(), "model.pth")
print("モデルが 'model.pth' に保存されました。")
