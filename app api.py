import streamlit as st
     import tensorflow as tf
     import numpy as np
     from PIL import Image
     import gdown
     import os

     # Hàm tải model từ Google Drive
     @st.cache_resource
     def load_model():
         model_path = "animal_classifier_advanced.h5"
         if not os.path.exists(model_path):
             # ID file từ Google Drive
             gdown.download("https://drive.google.com/uc?id=1nBdEoBfxGHgRyITgFlLRDZn9SdXrBTgS", model_path, quiet=False)
         return tf.keras.models.load_model(model_path)

     model = load_model()

     # Hàm xử lý ảnh
     def preprocess_image(image):
         image = image.resize((224, 224))  # Resize theo kích thước model
         image = np.array(image) / 255.0  # Chuẩn hóa
         image = np.expand_dims(image, axis=0)  # Thêm batch dimension
         return image

     # Giao diện Streamlit
     st.title("Phân loại Chó, Mèo, Chim")
     st.write("Tải ảnh lên để dự đoán (jpg, png, jpeg)!")

     # Widget tải file
     uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

     if uploaded_file is not None:
         try:
             # Hiển thị ảnh
             image = Image.open(uploaded_file).convert('RGB')
             st.image(image, caption="Ảnh đã tải", width=300)

             # Xử lý và dự đoán
             processed_image = preprocess_image(image)
             predictions = model.predict(processed_image)
             classes = ['Chó', 'Mèo', 'Chim']  # Cập nhật thứ tự lớp nếu cần
             predicted_class = classes[np.argmax(predictions[0])]
             confidence = np.max(predictions[0]) * 100

             # Hiển thị kết quả
             st.write(f"Dự đoán: **{predicted_class}** ({confidence:.2f}%)")
         except Exception as e:
             st.error(f"Lỗi: {str(e)}")
