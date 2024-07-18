import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import os
# import nltk
# from individual_python_files.contractions import contractions_dict
import contractions
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model

df = pd.read_csv("train.csv")

X = df['comment_text']
y = df[df.columns[2:]].values
MAX_FEATURES = 200000

vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(X.values)

model = load_model("model.h5")

st.title("Comment Toxicity Detection")
input_comment = st.text_area("Enter Comments", height=150)

if st.button("Check Toxicity"):
    if not input_comment.strip():
        st.warning("Please fill in the comments.")
    else:
        # Replace this with the actual toxicity detection logic
        input_comment1=contractions.fix(input_comment)
        # st.title(input_comment1)
        input_str = vectorizer(input_comment1)
        res = model.predict(np.expand_dims(input_str, 0))
        # b = vectorizer(tf.constant([[input_comment]]))
        # input_text = vectorizer(b)
        prediction_array = res.astype(float)>0.5
#
        # prediction_array = np.array([[0.99934345, 0.11597692, 0.9765822, 0.00283745, 0.95932055, 0.04028583]],
        #                             dtype=np.float32)
#
#         # Define thresholds
        threshold = 0.5
#         #
#         # # Map prediction to boolean based on the threshold
        detection_result = {
            'toxic': prediction_array[0, 0] > threshold,
            'severe_toxic': prediction_array[0, 1] > threshold,
            'obscene': prediction_array[0, 2] > threshold,
            'threat': prediction_array[0, 3] > threshold,
            'insult': prediction_array[0, 4] > threshold,
            'identity_hate': prediction_array[0, 5] > threshold
        }
#         #
        output = "\n".join([f"{key}: {value}" for key, value in detection_result.items()])
        st.text_area(label="Output:", value=output, height=180)
#
#
