# Facial Emotion Detection using Transfer Learning and Webcam


This repository contains code for real-time facial emotion detection using MobileNet pretrained model and webcam feed. The model is trained on the facial expression dataset which consists of 224x224 grayscale images of faces labeled with seven emotions - anger, disgust, fear, happiness, sadness, surprise, and neutral. The model achieved an accuracy of 76.61%.

<img align="right" alt="happy" width="400" src="https://i.ibb.co/4JXgcYd/happy.png">

<img alt="neutral" width="400" src="https://i.ibb.co/JKQR9WR/neutral.png">



# Requirements:

* Python 3

* OpenCV

* TensorFlow

* Keras

* NumPy


# Usage:

* Clone the repository.

* Download the saved model and place it in the models directory.

* Run python webcam.py to start the webcam and detect facial emotions in real-time.

* Press q to quit.

# Results: 

* The system achieved an accuracy of 76.61% in recognizing 7 different emotions

* The emotions detected are displayed on the screen in real-time.

* In real-world scenarios, the accuracy may vary depending on various factors such as lighting conditions, facial expressions, and camera angles.

# Webapplication

This web application of emotion detection involves capturing video of a person's face using a webcam or other camera, and then analyzing that video using machine learning algorithms to identify the person's emotional state. The results then used to adjust the content or user experience of the web application in real-time, based on the person's emotional response.

# Conclusion:

In conclusion, this repository provides a solution for real-time facial emotion detection using MobileNet pretrained model and webcam feed. The model achieved an accuracy of 76.61% and the script can be easily run on any machine with the required dependencies. This project can be further improved by exploring different pre-trained models or custom model architectures, increasing the size of the dataset or using a larger batch size for training. Overall, this project showcases the power and versatility of deep learning in computer vision applications and can be used as a starting point for further research and development.
