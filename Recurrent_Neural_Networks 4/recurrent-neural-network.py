#Part 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_trained = sc.fit_transform(training_set)

#creating data structure with 60 timestamps and 1 output

X_train = []
y_train = []
 
for i in range(60, 1258):
     X_train.append(training_set_trained[i-60:i, 0])
     y_train.append(training_set_trained[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# part 2

#building the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialising the rnn

regressor = Sequential()

#adding the first LSTM layer and dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))

#2nd layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

#3rd layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

#4th layer

regressor.add(LSTM(units=50))

regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))


#compiling the rnn

regressor.compile(optimizer='adam', loss = 'mean_squared_error')

#fitting the rnn to the training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3 predictions and visualisations

#getting real stock price

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#getting predicted stock price

dataset_total =pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 : ].values

inputs = inputs.reshape(-1, 1)

inputs = sc.transform(inputs)

X_test = []

for i in range(60, 80):
     X_test.append(inputs[i-60:i])
X_test = np.array(X_test)

#reshaping

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualising the results

plt.plot(real_stock_price, color='red', label='real google stock price')

plt.plot(predicted_stock_price, color='blue', label='predicted google stock price')

plt.title('google stock price prediction')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()




