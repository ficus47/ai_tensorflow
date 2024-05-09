import streamlit as st


def unique(a):
  result = []
  for i in a:
    if i not in result:
      result.append(i)
  return result

def train(file1, file2, epochs:int, learning_rate:float):
  # Imports
  import tensorflow.keras.layers as layers
  import tensorflow.keras.models as models
  import tensorflow.keras.optimizers as optimizers
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  from tensorflow.keras.preprocessing.text import Tokenizer

  file1, file2 = file1.split('\n'), file2.split('\n')
  
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(file1 + file2)
  x = tokenizer.texts_to_sequences(file1)
  y = tokenizer.texts_to_sequences(file2)
  maxlen = len(unique(file2))
  x, y = pad_sequences(x, maxlen=maxlen), pad_sequences(y, maxlen=maxlen)
  
  model = models.Sequential()
  model.add(layers.Embedding(len(tokenizer.word_index)+1, 64))
  model.add(layers.Bidirectional(layers.LSTM(64)))
  model.add(layers.Dense(maxlen, activation='softmax'))

  model.compile(loss="categorical_crossentropy",     
  optimizer=optimizers.Adam(learning_rate=learning_rate), 
                metrics=["accuracy"])
  
  model.fit(x, y, epochs=epochs)
  
  model.save("model.h5")
  
  z = model.evaluate(x, y)
  
  st.write(f"la perte du model est de {z[0]}% et son accuracy (precision) et de {z[1]}%")
  st.download_button(label="Télécharger le model", data=model.to_json(), file_name="model.json", mime="application/json")
