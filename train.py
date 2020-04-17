#Training using lstm model

#load data
import pandas as pd

df = pd.read_csv('preprocess.csv')
df.headline = df.headline.astype(str) #convert to string

#set the parameters
from keras.preprocessing.text import Tokenizer


MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

#tokenize
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(df['headline'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#create word sequences
from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

#convert categorical labels to numbers
Y = pd.get_dummies(df['category']).values
print('Shape of label tensor:', Y.shape)

#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)

#create word embeddings and train the model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(41, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
print(X_test.shape,Y_test.shape)

#save model
model.save("model.h5")
print("Saved model to disk")

