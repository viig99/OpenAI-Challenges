from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Masking, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def get_lstm(dimensions):
    model = Sequential()
    model.add(Masking(mask_value=-1., input_shape=(dimensions, 1)))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def parity_generator(dimensions, batch_size=1000):
    while True:
        x_batch = np.round(np.random.random_sample(
            (batch_size, dimensions, 1)))
        y_batch = np.logical_xor.reduce(x_batch, 1, keepdims=False)
        yield [x_batch, y_batch]

def parity_generator_random_length(dimensions, batch_size=1000):
	while True:
		x_batch = []
		y_batch = []
		for i in range(batch_size):
			length = np.random.choice(np.arange(1, dimensions+1))
			x_mini_batch = np.round(np.random.random_sample((length, 1))).flatten()
			y_mini_batch = np.logical_xor.reduce(x_mini_batch, keepdims=False)
			x_batch.append(x_mini_batch)
			y_batch.append(y_mini_batch)
		x_batch = pad_sequences(x_batch, maxlen=dimensions, padding='pre', value=-1).reshape(batch_size, dimensions, 1)
		y_batch = np.array(y_batch).reshape(-1,1)
		yield [x_batch, y_batch]


if __name__ == '__main__':
    dimensions = 50
    model = get_lstm(dimensions)
    training_gen = parity_generator(dimensions)
    validity_data = next(parity_generator(dimensions, 1000))
    model.fit_generator(
        generator=training_gen,
        steps_per_epoch=100,
        epochs=20,
        verbose=1,
        validation_data=validity_data,
        max_q_size=10,
        workers=3,
        callbacks=[
            EarlyStopping(monitor='val_loss', min_delta=0.0001,
                          patience=20, verbose=1, mode='auto'),
            ReduceLROnPlateau(monitor='val_loss', patience=10,
                              verbose=1, min_lr=0.0001, epsilon=0.0001, factor=0.1),
            TensorBoard(log_dir='./logs/parity_generator', histogram_freq=1, write_grads=True)
        ])

    x_batch = np.round(np.random.random_sample((10, dimensions, 1)))
    y_actual = np.logical_xor.reduce(x_batch, 1, keepdims=False)
    y_batch = model.predict(x_batch)
    print(y_actual.flatten())
    print(y_batch)

    # Accuracy for parity_generator ~ 50%
    # Accuracy for parity_generator_random_length ~ 100%
