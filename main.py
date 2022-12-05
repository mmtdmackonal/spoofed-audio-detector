import streamlit as st
import os
import numpy as np
import soundfile as sf
import wave
import tensorflow as tf
import matplotlib.pyplot as plt
import sqlite3
import cv2
import hashlib

# To load the model
model = tf.keras.models.load_model('./weights/model_weights.h5')

# Function to save uploaded audio file


def save_uploaded_file(uploaded_file):

	filepath = os.path.join(os.getcwd(), "temp", uploaded_file.name)
	with open(filepath, "wb") as f:

		f.write(uploaded_file.getbuffer())
	return filepath

# Function to generate frequency plot and the amplitude plot


def get_plots(file):

	audio = wave.open(file, 'rb')
	sample_freq = audio.getframerate()
	sample_count = audio.getnframes()
	sig_wave = audio.readframes(sample_count)
	sig_arr = np.frombuffer(sig_wave, dtype=np.int16)
	dur = sample_count / sample_freq
	timestamps = np.linspace(0, dur, sample_count)

	# Amplitude plot
	amp_plot = plt.figure(figsize=(12, 5))
	plt.plot(timestamps, sig_arr)
	plt.title('Amplitude Plot')
	plt.ylabel('Signal Value')
	plt.xlabel('Time (s)')
	plt.xlim(0, dur)

	# Frequency plot
	freq_plot = plt.figure(figsize=(12, 5))
	plt.specgram(sig_arr, Fs=sample_freq, vmin=-20, vmax=50)
	plt.title('Frequency Plot')
	plt.ylabel('Frequency (Hz)')
	plt.xlabel('Time (s)')
	plt.colorbar()

	return amp_plot, freq_plot

# Function to predictions


def predict(file = None):
	# Generating Spectrogram from the file
	y, sr = sf.read(file)
	plt.specgram(y, Fs=sr)
	plt.axis('off')
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	plt.savefig('./temp/1.png')

	# Image preprocessing-Set width and height
	image = cv2.imread('./temp/1.png')
	image = cv2.resize(image, (128, 128))
	image = np.expand_dims(image, axis=0)

	# Prediction
	pred = model.predict(image)
	label = "This is a Fake Audio Fie" if np.argmax(pred[0]) == 0 else "This is a Real Audio Fie"

	# Removing temp files after prediction
	os.remove('./temp/1.png')
	os.remove(file)
	# Return Output
	return label


# For the Security of the system

# DB Management
# establishing connection
conn = sqlite3.connect('data.db')
c = conn.cursor()

# passlib,hashlib,bcrypt,scrypt
# hashing the password


def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# Put data to table
# creating a table


def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

# Adding data to Columns


def add_userdata(username, password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?, ?)', (username, password))
	conn.commit()
# Retreiwing data


def login_user(username, password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


# main body of the application
def main():
	"""Fake Voice Detector"""

	st.title("Fake Voice Detector")

	menu = ["Home", "Login", "SignUp"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Home":
		st.subheader("Welcome to Fake Voice Detection")
		st.write("When You have suspicious Audio File And you dont have original sample of the file Fake Voice Detector is help you to check whether the file is fake or not.Please goto Login menu To start the Action")
		cover = cv2.imread('covers.jpg', cv2.IMREAD_UNCHANGED)
		cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
		st.image(cover)
# For the clicks on Login Button
	elif choice == "Login":
		st.subheader("Please enter your user name and password to start detection")
		cover1 = cv2.imread('covers.jpg', cv2.IMREAD_UNCHANGED)
		cover1 = cv2.cvtColor(cover1, cv2.COLOR_BGR2RGB)
		st.image(cover1)

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password", type='password')
		if st.sidebar.checkbox("Login"):
			# if password == '12345':
			create_usertable()
			hashed_password = make_hashes(password)

			result = login_user(username, check_hashes(password, hashed_password))
			if result:

				st.sidebar.success("Logged In as {}".format(username))

				uploaded_file = st.file_uploader("Upload an audio file", type=['wav'], accept_multiple_files = False,)
				if uploaded_file:
					filepath = save_uploaded_file(uploaded_file)
					if st.button('Run Prediction'):
						with st.container():
							st.write("#### Uploaded Audio File")
							st.audio(uploaded_file, format='audio/wav', start_time=0)
							st.header("Visualization of uploaded Audio File:")
							st.write(" ")
							amp_plot, freq_plot = get_plots(filepath)

							st.write("#### Amplitude Plot")
							st.pyplot(amp_plot)
							st.write("#### Frequency Plot")
							st.pyplot(freq_plot)
							with st.container():
								st.header("Results:")
								res = predict(filepath)
								st.write("### Classification Label: " + res)
			else:
				st.warning("Incorrect Username/Password")
# For the clicks on Sign up menu
	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password", type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user, make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")
# End of the sign-up


if __name__ == '__main__':
	main()
