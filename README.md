# Facial Emotion Detection using Transfer Learning and Webcam


This repository contains code for real-time facial emotion detection using MobileNet pretrained model and webcam feed. The model is trained on the facial expression dataset which consists of 48x48 grayscale images of faces labeled with seven emotions - anger, disgust, fear, happiness, sadness, surprise, and neutral. The model achieved an accuracy of 76.61%.


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

# Conclusion:

In conclusion, this repository provides a solution for real-time facial emotion detection using MobileNet pretrained model and webcam feed. The model achieved an accuracy of 76.61% and the script can be easily run on any machine with the required dependencies. This project can be further improved by exploring different pre-trained models or custom model architectures, increasing the size of the dataset or using a larger batch size for training. Overall, this project showcases the power and versatility of deep learning in computer vision applications and can be used as a starting point for further research and development.
