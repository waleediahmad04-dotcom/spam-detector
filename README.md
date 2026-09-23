# Spam Message Detector

A Python machine learning application that detects whether a text message is spam or not spam.

The application provides a simple graphical user interface where users can enter a message and receive a prediction along with a confidence percentage.

## Features

- Detects spam and non-spam messages
- Displays prediction confidence
- Simple desktop GUI
- Machine learning based text classification
- Standalone Windows application support

## Technologies Used

- Python
- Scikit-learn
- Tkinter
- Joblib
- TF-IDF text vectorization
- Machine Learning Classification
- PyInstaller

## How It Works

1. A dataset containing spam and non-spam messages is used to train the machine learning model.
2. Text messages are converted into numerical features using TF-IDF vectorization.
3. The trained model predicts whether a new message is spam or not spam.
4. The application displays the prediction and confidence percentage through the Tkinter GUI.

## Project Files

- spam_detector.py - Model training code
- spam_app.py - Desktop application GUI
- spam.csv - Dataset
- spam_model.pkl - Trained machine learning model
- vectorizer.pkl - Saved text vectorizer

## Example

Input:

Congrats! You have won £1000.

Output:

SPAM MESSAGE - 61.9% confidence

## Purpose

This project was developed as a practical introduction to Python, machine learning, text classification, and building desktop applications.
