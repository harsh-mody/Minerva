import streamlit as st
from streamlit.legacy_caching import caching
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from img_utils import *
from style_utils import *
from style_content_model import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

st.image("icon.png", width=100)
st.title("Minerva: Neural Style Transfer")
st.title("")
st.sidebar.title('Features')

learning_rate = st.sidebar.slider('Learning rate', 0.0, 0.1, .02)
epochs = st.sidebar.slider('Number of Epochs', 1, 100, 10)
steps_per_epoch = st.sidebar.slider('Steps Per Epoch', 1, 500, 100)

st.markdown(f'<br>', unsafe_allow_html=True)

content_img_buffer = st.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
style_img_buffer = st.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])


if content_img_buffer:
  content_image = load_img(content_img_buffer)


if style_img_buffer:
  style_image = load_img(style_img_buffer)

if content_img_buffer and style_img_buffer:
  st.sidebar.image([content_img_buffer, style_img_buffer], caption=['Content Image', 'Style Image'], use_column_width = True)

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = total_loss(outputs, targets, weights)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

@st.cache(allow_output_mutation=True, suppress_st_warning = True)
def run_style_transfer(image, epochs, steps_per_epoch):
  generated_image.image(tensor_to_image(image), caption = 'Starting...', use_column_width = True)
  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image)
      generated_image.image(tensor_to_image(image), caption = f'Training Step {step}', use_column_width = True)
  generated_image.image(tensor_to_image(image), caption = 'Generated Image', use_column_width = True)


if content_img_buffer and style_img_buffer:
  caching.clear_cache()
  clicked = st.sidebar.button('Generate')
  if clicked:
    caching.clear_cache()
    image = tf.Variable(content_image)
    extractor = StyleContentModel(style_layers, content_layers)
    opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

    targets = {
      "style": extractor(style_image)['style'],
      "content": extractor(content_image)['content']
    }

    weights = {
      "style": 1e-2,
      "content": 1e4
    }

    generated_image = st.empty()
    run_style_transfer(image, epochs, steps_per_epoch)

st.title("")
st.title("Image Catalog: ")
st.title("")
input_file = open ('images/images.json')
json_array = json.load(input_file)
for item in json_array:
  st.header(item['name'])
  col1, col2, col3 = st.columns(3)

  content = Image.open(item['content'])
  col1.text("Content")
  col1.image(content, use_column_width=True)

  style = Image.open(item['style'])
  col2.text("Style")
  col2.image(style, use_column_width=True)

  result = Image.open(item['result'])
  col3.text("Result")
  col3.image(result, use_column_width=True)