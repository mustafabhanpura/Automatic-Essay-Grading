import tensorflow as tf
#from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
#from keras.models import Sequential, load_model, model_from_config
import keras.backend as K

def get_model():
    """Define the model."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, recurrent_dropout=0.4))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model