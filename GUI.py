import tkinter as tk
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import preprocessing as kprocessing

model = tf.keras.models.load_model('Text_Classification_Model.h5')

with open('AG_News_Str2Bin.pickle', 'rb') as s:
    Str2Bin = pickle.load(s)

with open('AG_News_Tokenizer.pkl', 'rb') as t:
    tokenizer = pickle.load(t)


def submit_text():
    user_input = text_area.get("1.0", tk.END)
    seq = kprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=200)
    prediction = model.predict(seq)
    confidence_score = np.max(prediction)
    result = Str2Bin.inverse_transform(prediction)
    
    result_area1.configure(text = "Topic: " + str(result))
    result_area2.configure(text = " Confidence score: " + str(confidence_score * 100) + "%")


root = tk.Tk()
root.title("Text Classification")
root.state('zoomed')
root.configure(bg = "gray")

text_area = tk.Text(root, height=20, width = 100, font=("Helvetica", 16), background = "white", foreground="black")
text_area.pack(pady=20)

submit_button = tk.Button(root, text="Submit", command=submit_text, font=("Helvetica", 16), background = "red4", foreground="white")
submit_button.pack()

result_area1 = tk.Label(root, text="", height=2, width=100, font=("Helvetica", 16), background = "gray", foreground="red4")
result_area1.pack(pady=20)

result_area2 = tk.Label(root, text="", height=2, width=100, font=("Helvetica", 16), background = "white", foreground="red4")
result_area2.pack(pady=5)

root.mainloop()
