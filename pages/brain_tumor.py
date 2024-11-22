import streamlit as st
# brain_tumor.py
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
import requests
from io import BytesIO

st.title("YOLOv5 Brain tumor detection")

# ---------------------------------------------------------------------------------------------#
### Mode for Axial detection ###
# File input #
st.sidebar.header("Axial")
uploaded_file = st.sidebar.file_uploader("Select image from your folder...", type=["jpg", "jpeg", "png"], key="file_uploader_1")
# ---------------------------------------------------------------------------------------------#
# Def model #
@st.cache_resource
def load_model_axial():
    model_axial = torch.hub.load(
        # будем работать с локальной моделью в текущей папке
        repo_or_dir = '/Users/vladislavkravchenko/Desktop/WORK_IT/1.Elbrus_Bootcamp/3.Phase2/9W/9.4./3.Streamlit/pythonProject/yolov5',
        # непредобученная – будем подставлять свои веса
        model = 'custom',
        # путь к нашим весам
        path='/Users/vladislavkravchenko/Desktop/WORK_IT/1.Elbrus_Bootcamp/3.Phase2/9W/9.4./3.Streamlit/pythonProject/yolov5/1.Top_yolo/1.axial_t1wce_2_class/exp2_b20_ep200/weights/best.pt',
        # откуда берем модель – наша локальная
        source='local'
        )
    return model_axial
# Variable for model #
model_axial = load_model_axial()
# ---------------------------------------------------------------------------------------------#
# Confidence interval #
# Определение диапазона значений для уверенности
min_confidence = 0.0
max_confidence = 1.0
step = 0.01  # Шаг изменения

# Использование select_slider для выбора уровня уверенности
model_conf_axial = st.sidebar.select_slider(
    "Model Confidence Selection:",
    options=[round(i, 2) for i in list(np.arange(min_confidence, max_confidence + step, step))],
    value=0.5  # Значение по умолчанию
)
# ---------------------------------------------------------------------------------------------#
# Predict function #
def detect_axial(img):
    model_axial.conf = model_conf_axial
    with torch.inference_mode():
        results = model_axial(img)
    result_img = results.render()[0]  # render() возвращает список изображений с аннотациями
    result_pil = Image.fromarray(result_img)  # Преобразуем numpy array в PIL.Image
    st.image(result_pil, caption='Detected Axial image', use_container_width=True)

# Predict plot #
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption='Non-detected Axial image.', use_container_width=True)
if st.sidebar.button("Predict"):
    img = Image.open(uploaded_file).convert("RGB")
    detect_axial(img)
st.sidebar.header("--------------------------------------------")
st.write('---')
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
### Mode for Coronal detection ###
# File input #
st.sidebar.header("Coronal")
uploaded_file = st.sidebar.file_uploader("Select image from your folder...", type=["jpg", "jpeg", "png"], key = 'coronal')
# ---------------------------------------------------------------------------------------------#
# Def model #
@st.cache_resource
def load_model_coronal():
    model_coronal = torch.hub.load(
        # будем работать с локальной моделью в текущей папке
        repo_or_dir = '/Users/vladislavkravchenko/Desktop/WORK_IT/1.Elbrus_Bootcamp/3.Phase2/9W/9.4./3.Streamlit/pythonProject/yolov5',
        # непредобученная – будем подставлять свои веса
        model = 'custom',
        # путь к нашим весам
        path='/Users/vladislavkravchenko/Desktop/WORK_IT/1.Elbrus_Bootcamp/3.Phase2/9W/9.4./3.Streamlit/pythonProject/yolov5/1.Top_yolo/2.coronal_t1wce_2_class/exp2_b20_ep200/weights/best.pt',
        # откуда берем модель – наша локальная
        source='local'
        )
    return model_coronal
# Variable for model #
model_coronal = load_model_coronal()
# ---------------------------------------------------------------------------------------------#
# Confidence interval #
# Определение диапазона значений для уверенности
min_confidence = 0.0
max_confidence = 1.0
step = 0.01  # Шаг изменения

# Использование select_slider для выбора уровня уверенности
model_conf_coronal = st.sidebar.select_slider(
    "Model Confidence Selection:",
    options=[round(i, 2) for i in list(np.arange(min_confidence, max_confidence + step, step))],
    value=0.5,
    key = 'coronal_2'# Значение по умолчанию
)
# ---------------------------------------------------------------------------------------------#
# Predict function #
def detect_coronal(img):
    with torch.inference_mode():
        model_coronal.conf = model_conf_coronal
        results = model_coronal(img)
    result_img = results.render()[0]  # render() возвращает список изображений с аннотациями
    result_pil = Image.fromarray(result_img)  # Преобразуем numpy array в PIL.Image
    st.image(result_pil, caption='Detected Coronal image', use_container_width=True)

# Predict plot #
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption='Non-detected Coronal image.', use_container_width=True)
if st.sidebar.button("Predict", key = 'coronal_3'):
    img = Image.open(uploaded_file).convert("RGB")
    detect_coronal(img)
st.sidebar.header("--------------------------------------------")
st.write('---')
# ---------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------#
### Mode for Sagittal detection ###
# File input #
st.sidebar.header("Sagittal")
uploaded_file = st.sidebar.file_uploader("Select image from your folder...", type=["jpg", "jpeg", "png"], key = 'sagittal')
# ---------------------------------------------------------------------------------------------#
# Def model #
@st.cache_resource
def load_model_sagittal():
    model_sagittal = torch.hub.load(
        # будем работать с локальной моделью в текущей папке
        repo_or_dir = '/Users/vladislavkravchenko/Desktop/WORK_IT/1.Elbrus_Bootcamp/3.Phase2/9W/9.4./3.Streamlit/pythonProject/yolov5',
        # непредобученная – будем подставлять свои веса
        model = 'custom',
        # путь к нашим весам
        path='/Users/vladislavkravchenko/Desktop/WORK_IT/1.Elbrus_Bootcamp/3.Phase2/9W/9.4./3.Streamlit/pythonProject/yolov5/1.Top_yolo/3.sagittal_t1wce_2_class/exp2_b20_ep200/weights/best.pt',
        # откуда берем модель – наша локальная
        source='local'
        )
    return model_sagittal
# Variable for model #
model_sagittal = load_model_sagittal()
# ---------------------------------------------------------------------------------------------#
# Confidence interval #
# Определение диапазона значений для уверенности
min_confidence = 0.0
max_confidence = 1.0
step = 0.01  # Шаг изменения

# Использование select_slider для выбора уровня уверенности
model_conf_sagittal = st.sidebar.select_slider(
    "Model Confidence Selection:",
    options=[round(i, 2) for i in list(np.arange(min_confidence, max_confidence + step, step))],
    value=0.5,
    key = 'sagittal_2'# Значение по умолчанию
)
# ---------------------------------------------------------------------------------------------#
# Predict function #
def detect_sagittal(img):
    with torch.inference_mode():
        model_sagittal.conf = model_conf_sagittal
        results = model_sagittal(img)
    result_img = results.render()[0]  # render() возвращает список изображений с аннотациями
    result_pil = Image.fromarray(result_img)  # Преобразуем numpy array в PIL.Image
    st.image(result_pil, caption='Detected Sagittal image', use_container_width=True)

# Predict plot #
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption='Non-detected Sagittal image.', use_container_width=True)
if st.sidebar.button("Predict", key = 'sagittal_3'):
    img = Image.open(uploaded_file).convert("RGB")
    detect_sagittal(img)
st.sidebar.header("--------------------------------------------")
st.write('---')
# ---------------------------------------------------------------------------------------------#
