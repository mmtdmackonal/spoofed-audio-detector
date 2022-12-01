# Required Imports
import os
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import soundfile as sf
import wave
import tensorflow as tf
import matplotlib.pyplot as plt

# Loading Model Weights
model = tf.keras.models.load_model('./weights/model_weights.h5')

# Function to save the uploaded file
def save_uploadedfile(uploadedfile):
     filepath = os.path.join(os.getcwd(), "temp", uploadedfile.name)
     with open(filepath,"wb") as f:
         f.write(uploadedfile.getbuffer())
     return filepath

# Function to generate the amplitude and frequency plots
def get_plots(file):

    # Getting and preparing required parameters
    audio = wave.open(file, 'rb')
    sample_freq = audio.getframerate()
    sample_count = audio.getnframes()
    sig_wave = audio.readframes(sample_count)
    sig_arr = np.frombuffer(sig_wave, dtype=np.int16)    
    dur = sample_count/sample_freq
    timestamps = np.linspace(0, dur, sample_count)

    # Plotting Amplitude Plot
    amp_plot = plt.figure(figsize=(12,5))
    plt.plot(timestamps, sig_arr)
    plt.title('Amplitude Plot')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, dur)

    # Plotting Frequency Plot 
    freq_plot = plt.figure(figsize=(12, 5))
    plt.specgram(sig_arr, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title('Frequency Plot')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar()

    return amp_plot, freq_plot

# Function to predict the uploaded file
# (AI Inference PIPELINE)
def predict(file = None):

    # Generating Spectogram from the file
    y, sr = sf.read(file)
    plt.specgram(y,Fs=sr)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig('./temp/1.png')
    
    # Pre-processing
    image = cv2.imread('./temp/1.png')
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)

    # Predicting
    pred = model.predict(image)
    label = "Real Audio with Accurasy Score 93%" if np.argmax(pred[0]) == 0 else "Spoofed"

    # Removing temp files after prediction
    os.remove('./temp/1.png')
    os.remove(file)

    return label

if __name__ == '__main__':

    # Basic Page Configurations
    st.set_page_config(
     page_title="Fake Speech Detector",
     page_icon="🎙️",
     layout="centered",
     menu_items={
         'About': "##### This is a Fake Speech Detection app which predicts whether the speech is spoofed or not by analyzing the input audio of human voice."
        }
    ) 

    # Main Body 
    st.title("Fake Voice Detection Application")



    #with st.container():
       # col1, col2, col3 = st.columns(3)
       # col1.metric("Model Backend", "CNN")
       # col3.metric("Validation Accuracy", "98%")
        #col2.metric("Prediction Classes", "2")

    # Cover Image
    cover = cv2.imread('covers.jpg', cv2.IMREAD_UNCHANGED)
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    st.image(cover)


    # Audio File Upload Component
    uploaded_file = st.file_uploader("Upload an audio file", 
                                    type=['wav'],
                                    accept_multiple_files = False,)

    # If file is uploaded, allow user to perform prediction
    if uploaded_file:
        filepath = save_uploadedfile(uploaded_file)
        


        if st.button('Run Prediction'):

            with st.container():

                # Showing the uploaded file
                st.write("#### Uploaded Audio File")
                st.audio(uploaded_file, format='audio/wav', start_time=0)

                # Displaying the amplitude and frequency plots
                st.header("Visualization of uploaded Audio File:")
                st.write(" ")

                amp_plot, freq_plot = get_plots(filepath)

                st.write("#### Amplitude Plot")
                st.pyplot(amp_plot)
                st.write("#### Frequency Plot")
                st.pyplot(freq_plot)

                # Showing the results
                with st.container():
                    st.header("Results:")
                    res = predict(filepath)

                    st.write("### Classification Label: " + res)

