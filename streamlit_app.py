import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import base64
from fpdf import FPDF
from datetime import datetime  # 日時の取得に使用
from io import BytesIO  # 一時データ保存に使用

# ===== Streamlit ページ設定 =====
st.set_page_config(page_title="Tryna use Mxxxxy", layout="wide")

# ===== 背景画像をBase64エンコード =====
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_image_path = "C:/Users/E.ykt/Desktop/1009_自走課題/background.png"

if os.path.exists(background_image_path):
    print(f"背景画像が見つかりました: {background_image_path}")
    img_base64 = get_base64_of_bin_file(background_image_path)
else:
    print(f"背景画像が見つかりませんでした: {background_image_path}")
    img_base64 = ""

# ===== カスタムCSSで背景画像を設定 =====
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
    }}
    h1 {{
        color: #FF4B4B;
        text-align: center;
    }}
    .stButton>button {{
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===== タイトルの表示 =====
st.markdown(
    """
    <h1>
    “あの”キャラクター、使える？<br>画像判定
    </h1>
    """,
    unsafe_allow_html=True
)

# ===== モデルの定義 =====
class ImageAuthClassifier(torch.nn.Module):
    def __init__(self):
        super(ImageAuthClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet(x)

# ===== モデルのロード関数 =====
def load_model(model_path="model.pth"):
    model = ImageAuthClassifier()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ===== 画像の前処理 =====
transform = transforms.Compose([
    transforms.Resize((630, 630)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===== PDF生成関数 =====
def generate_pdf(image_data, image_name, result, timestamp):
    pdf = FPDF()
    pdf.add_page()

    font_path = "C:/Windows/Fonts/msgothic.ttc"
    pdf.add_font("Gothic", "", font_path, uni=True)
    pdf.set_font("Gothic", size=20)

    pdf.cell(200, 10, txt="画像判定レポート", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font("Gothic", size=16)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"画像ファイル名: {image_name}", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"判定日時: {timestamp}", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"判定結果: {result}", new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.ln(10)

    # 一時的な画像データをPDFに追加
    temp_image_path = BytesIO(image_data)
    pdf.image(temp_image_path, x=60, y=None, w=100)

    # PDFを一時バッファに保存
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)  # バッファの先頭に移動
    return pdf_buffer

# ===== Streamlitアプリ =====
with st.sidebar:
    st.header("画像をアップロード")
    uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "png", "jpeg"])

model = load_model("model.pth")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", width=300)

    width, height = image.size
    st.write(f"画像サイズ: **{width} x {height}** ピクセル")

    if st.button("画像を判定"):
        with st.spinner('判定中...少々お待ちください'):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)

            class_names = ["使用可能", "使用不可"]
            result = class_names[predicted.item()]

        if result == "使用可能":
            st.success(f"予測結果: **{result}** 🟢")
        else:
            st.error(f"予測結果: **{result}** 🔴")

        image_name = uploaded_file.name.split('.')[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 画像をバイナリデータに変換
        image_buffer = BytesIO()
        image.save(image_buffer, format='PNG')
        image_data = image_buffer.getvalue()

        # PDFの生成とダウンロード
        pdf_buffer = generate_pdf(image_data, image_name, result, timestamp)

        st.download_button(
            label="レポートをダウンロード",
            data=pdf_buffer,
            file_name=f"{image_name}_判定レポート.pdf",
            mime="application/pdf"
        )
