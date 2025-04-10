import streamlit as st
import cv2 
import numpy as np    
import tensorflow as tf
import time
import os
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from PIL import Image
import keras
#from streamlit_background_image import st_background_image

def main():
    #t.sidebar.markdown("**Image Noise Removal(Denoising) using Deep Learning**")
    st.sidebar.image("Title.png")
    st.sidebar.image("DeNoise.gif")
    st.sidebar.subheader("Choose an option here:arrow_down:",divider='grey')
    selected_box = st.sidebar.selectbox(
            '',('About','Denoise Image')
        )
    if selected_box == 'About':
        #readme=Image.open('Hear.png')
        #st.image(readme)
        st.write("# Given an image, this web-app will try to remove the noise present in it using Deep Learning.")
        st.header('',divider='rainbow')
        st.subheader(":orange[How to use it?]")
        st.text("On the left sidebar, click on the option 'Denoise Image'.\nA web page will open where you can get a denoised output for a given noisy image.")
        st.subheader(":orange[There are two ways to upload a noisy image:]")
        st.text("1.You can choose any one of the sample images provided.\n2.You can provide any image to denoise from your PC.")
    if selected_box == 'Denoise Image':
        models()
    
def models():
    #st.title("Image Noise Removal(Denoising) using Deep Learning")
    #top=Image.open('Top.png')
    #st.image(top)
    st.write('# You can predict on sample images or you can upload a noisy image and get its denoised output.',divider='rainbow')
    
    selection=st.selectbox("Choose how to load image",["Select","Upload an Image","Predict on sample Images"])
    
    if selection=="Upload an Image":
        image = st.file_uploader('Upload the image below')
        predict_button = st.button('Predict on uploaded image')
        if predict_button:
            if image is not None:
                file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
                nsy_img = cv2.imdecode(file_bytes, 1)
                prediction(nsy_img)
            else:
                st.text('Please upload the image')    
    
    if selection=='Predict on sample Images':
        option = st.selectbox('Select a sample image',('Select','Toy car','Vegetables','Gadget desk','Scrabble board','Shoes','Door','Chess board','A note'))
        if option=='Select':
            pass
        else:
            path = os.path.join(os.getcwd(),'NoisyImage/')
            nsy_img = cv2.imread(path+option+'.jpg')
            prediction(nsy_img)
            
def patches(img,patch_size):
  patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
  return patches

def get_model():
    RIN=tf.keras.models.load_model('RIDNet.h5')
    return RIN

def prediction(img):
    
    with st.columns(3)[1]:
        state = st.text('\n Wait the Model is Running')
        im = st.image("Load.gif")
    progress_bar = st.progress(0)
    start = time.time()
    model = get_model()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nsy_img = cv2.resize(img,(1024,1024))
    nsy_img = nsy_img.astype("float32") / 255.0

    img_patches = patches(nsy_img,256)
    progress_bar.progress(30)
    nsy=[]
    for i in range(4):
        for j in range(4):
            nsy.append(img_patches[i][j][0])
    nsy = np.array(nsy)
    
    pred_img = model.predict(nsy)
    progress_bar.progress(70)
    pred_img = np.reshape(pred_img,(4,4,1,256,256,3))
    pred_img = unpatchify(pred_img, nsy_img.shape)
    progress_bar.progress(95)
    im.empty()

    end = time.time()

    img = cv2.resize(img,(512,512))
    pred_img = cv2.resize(pred_img,(512,512))
    fig,ax = plt.subplots(1,2,figsize=(10,10))

    ax[1].imshow(pred_img)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].title.set_text("Predicted Image")

    ax[0].imshow(img) 
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].title.set_text("Noisy Image")
    

    st.pyplot(fig)
    progress_bar.progress(100)
    st.write('Time taken for prediction :', str(round(end-start,3))+' seconds')
    progress_bar.empty()
    state.empty()
    st.text('\n Completed!')
    
if __name__ == "__main__":
    main()

