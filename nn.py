import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pandas.read_csv('reformed_assessors_data.csv', index_col='id')
features, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16], 11
data, target = df.values[:, features], df.values[:, labels]

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.3)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
features = [0, 9, 11, 13]

x_train = x_train[:, features]
x_test = x_test[:, features]

model = Sequential()

es = EarlyStopping()

model.add(Dense(1, activation='sigmoid', input_dim=len(features)))

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, nb_epoch=10000, validation_data=(x_test, y_test), callbacks=[es])
model.save('nn.h5')
