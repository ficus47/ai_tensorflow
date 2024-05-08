import streamlit as st

import train

st.title("AI on Replit")

file1 = uploaded_file = st.file_uploader("Charger un fichier x .txt", type=["txt"])
 #"maman\ngentil\nimbecile\ndebile"
file2 = uploaded_file = st.file_uploader("Charger un fichier y .txt", type=["txt"])
#"neutre\nneutre\nvulgaire\nvulgaire"

epochs = st.slider(label="Epochs", min_value=1, max_value=100, value=10)
lr = st.number_input(label="Learning rate", min_value=0.0001, max_value=1.0, value=0.0001)


st.write(f"learning rate = {lr}")

if file1 is not None and file2 is not None and st.button("train"):
  train.train(file1.read().decode("utf-8"), file2.read().decode("utf-8"), epochs, lr)
