from PPdata import aapl_training_y, aapl_training_x, aapl_test_y, aapl_test_x
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU



model = Sequential()
model.add(GRU(units=100, return_sequences=True, input_shape=(aapl_training_x.shape[1], 1)))
model.add(Dropout(0.6))
model.add(GRU(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(aapl_training_x, aapl_training_y, batch_size=125, epochs=30)


prediction = model.predict(aapl_test_x)
plt.figure()
plt.plot(aapl_test_y, label='actual')
plt.plot(prediction, label='predicted')
plt.legend()
