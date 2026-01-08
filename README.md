ASL Hand Sign Detection (Custom Dataset)
📌 Project Overview
This project is an American Sign Language (ASL) hand sign detection system built using Machine Learning / Deep Learning techniques.
The main goal of this project is to recognize hand signs (alphabets) using a camera feed by training a model on custom hand sign images captured by me.
This project is part of my learning journey in Machine Learning and Computer Vision, where I focused more on understanding the workflow rather than building a perfect production-ready system.
🎯 Objective
To understand how hand sign recognition works using ML/CNN
To create and train a model using my own hand sign dataset
To learn dataset preparation, model training, and real-time prediction
To experiment with different approaches and improve accuracy over time
🗂 Dataset
The dataset is custom-built using images of my own hands
Images were captured using a webcam and stored class-wise (A–Z, blank, etc.)
Both left and right hand images were experimented with
Dataset quality and variation are still limited, which affects accuracy
⚠️ Note:
The dataset is one of the main reasons the model is not 100% accurate. More diverse images (lighting, background, hand size, angles) are needed.
🧠 Model & Approach
Used Convolutional Neural Network (CNN) for image classification
Image preprocessing like resizing and normalization was applied
Model trained on the custom dataset
Real-time prediction done using OpenCV
The basic structure and workflow were created using:
Official documentation
Online references
Assistance from ChatGPT (for understanding, debugging, and improvements)
This project was built while learning, so the focus was on experimenting and understanding, not blindly copying code.
🛠 Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Machine Learning / Deep Learning concepts
✅ Current Status
✔ Model is trained and runs successfully
✔ Real-time hand sign detection works
✔ Some alphabets are detected correctly
❌ Accuracy is inconsistent
❌ Some signs are misclassified
❌ Model is sensitive to lighting and hand position
🔧 Limitations & Issues
Limited dataset (single person’s hand)
Similar hand signs confuse the model
Background and lighting affect predictions
Left and right hand variations need better handling