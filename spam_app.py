import tkinter as tk
from tkinter import messagebox
import joblib
import os
import sys


# Find model files when running normally or as an EXE
def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)


# Load trained model and vectorizer
model = joblib.load(resource_path("spam_model.pkl"))
vectorizer = joblib.load(resource_path("vectorizer.pkl"))


# Check message
def check_message():
    message = message_box.get("1.0", tk.END).strip()

    if not message:
        messagebox.showwarning(
            "Empty Message",
            "Please enter a message to check."
        )
        return

    # Convert message into numbers
    message_vector = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0]
    confidence = probabilities[prediction] * 100

    if prediction == 1:
        result_label.config(
            text=f"🚨 SPAM MESSAGE\n{confidence:.1f}% confidence",
            fg="red"
        )
    else:
        result_label.config(
            text=f"✅ NOT SPAM\n{confidence:.1f}% confidence",
            fg="green"
        )


# Clear message
def clear_message():
    message_box.delete("1.0", tk.END)
    result_label.config(
        text="Enter a message and click CHECK MESSAGE",
        fg="gray"
    )


# Create main window
root = tk.Tk()
root.title("Spam Detector")
root.geometry("650x500")
root.resizable(False, False)
root.configure(bg="#f4f6f8")


# Title
title_label = tk.Label(
    root,
    text="SPAM MESSAGE DETECTOR",
    font=("Arial", 22, "bold"),
    bg="#f4f6f8",
    fg="#1f2937"
)
title_label.pack(pady=(30, 5))


# Subtitle
subtitle_label = tk.Label(
    root,
    text="Enter a message below to check whether it is spam.",
    font=("Arial", 11),
    bg="#f4f6f8",
    fg="#555555"
)
subtitle_label.pack(pady=(0, 20))


# Message box
message_box = tk.Text(
    root,
    height=9,
    width=60,
    font=("Arial", 12),
    wrap=tk.WORD,
    relief=tk.SOLID,
    borderwidth=1
)
message_box.pack(pady=10)


# Check button
check_button = tk.Button(
    root,
    text="CHECK MESSAGE",
    command=check_message,
    font=("Arial", 12, "bold"),
    width=20,
    height=2,
    cursor="hand2"
)
check_button.pack(pady=(15, 5))


# Clear button
clear_button = tk.Button(
    root,
    text="CLEAR",
    command=clear_message,
    font=("Arial", 10),
    width=12,
    cursor="hand2"
)
clear_button.pack(pady=5)


# Result
result_label = tk.Label(
    root,
    text="Enter a message and click CHECK MESSAGE",
    font=("Arial", 14, "bold"),
    bg="#f4f6f8",
    fg="gray"
)
result_label.pack(pady=20)


# Start application
root.mainloop()
