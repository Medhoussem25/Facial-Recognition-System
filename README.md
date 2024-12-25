Facial Recognition System
Project Overview
This project implements a facial recognition system using Tkinter for the graphical interface and TensorFlow Lite for feature extraction. The application enables facial identification through image testing or live camera feed, leveraging pre-trained models and a feature database.

Features
Login System:

Secure access using predefined admin credentials.
Displays the main application interface upon successful login.
Facial Recognition:

Supports image testing for facial identification.
Real-time recognition using a live camera feed.
Feature Extraction:

Utilizes a TFLite model to extract facial features from images.
Compares features with a preloaded database for identification.
Graphical Comparison:

Displays a comparison graph of input features and matched features.
Team Information:

Includes an option to display developer details.
User-Friendly Interface:

Designed with Tkinter, featuring intuitive navigation and responsive buttons.
Error Handling:

Provides clear error messages for incorrect credentials or camera issues.
Technologies Used
Python: Primary programming language.
Tkinter: GUI framework.
TensorFlow Lite: Model interpreter for feature extraction.
OpenCV: Camera handling and face detection.
NumPy: Array operations and feature database management.
Matplotlib: Visualization of feature comparison.
Pillow: Image processing and rendering.
How It Works
Login:

Users log in with a username and password.
Admin access allows entry to the main interface.
Image Testing:

Users select an image file for testing.
The system preprocesses the image, extracts features, and identifies the person by comparing features with the database.
Live Camera Feed:

The camera captures real-time video.
Detected faces are processed, and identities are displayed on the feed.
Graphical Comparison:

A graph shows the similarity between input features and matched features.
Developer Details:

An option displays the development team information.
Prerequisites
Python 3.8+
Required libraries: tkinter, cv2, numpy, tensorflow, Pillow, matplotlib, scipy
Installation and Setup
Clone the project repository.
Install dependencies using pip install -r requirements.txt.
Ensure the following files are in the project directory:
TFLite model: feature_extractor.tflite
Feature database: feature_db.npy
Icon image: reconnaissance-faciale.png
Run the script using python main.py.
Usage
Start the application.
Log in with the admin credentials:
Username: admin
Password: 1234
Choose between:
Testing an image.
Starting the live camera feed for real-time recognition.
View results and graphical comparisons.
Developer Information
Developed by Houssem Eddine Bouagal.
This system demonstrates a combination of machine learning and GUI programming for practical facial recognition applications.
