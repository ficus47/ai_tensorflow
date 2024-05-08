import streamlit as st

import train

st.title("AI on Replit")

file1 = "maman\ngentil\nimbecile\ndebile"
file2 = "neutre\nneutre\nvulgaire\nvulgaire"

epochs = st.slider(label="Epochs", min_value=1, max_value=100, value=10)
lr = st.number_input(label="Learning rate va etre divisé par 10", min_value=0.0001, max_value=1.0, value=0.0001)


st.write(f"learning rate = {lr}")

if file1 is not None and file2 is not None and st.button("train"):
  train.train(file1, file2, epochs, lr)
