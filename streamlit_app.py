import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# モデルの定義（ResNet18を使用）
class CatDogClassifier(torch.nn.Module):
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 2)  # 2クラス分類

    def forward(self, x):
        return self.resnet(x)

# モデルのロード関数
def load_model(model_path="model.pth"):
    model = CatDogClassifier()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((630, 630)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlitのタイトル
st.title("“あの”キャラクター、使える？画像判定")
st.write("以下に画像をアップロードして、分類を実行します。")

# モデルのロード
model = load_model("model.pth")

# 画像アップロードUI
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"], key="unique_file_uploader")

if uploaded_file is not None:
    # アップロードされた画像を開いて表示
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 「分類を実行」ボタン
    if st.button("分類を実行"):
        # 前処理の適用
        input_tensor = transform(image).unsqueeze(0)

        # 推論の実行
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        # クラスラベルの定
