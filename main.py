from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

#Load data with input variables (signals)
signals = np.loadtxt('data_output.dat')

#Load data with output classes (geometrical structure)
classes = np.load('Matrix_of_geometries.npy')

#Reshape data
signals = np.reshape(signals, (100000,100))
classes = np.reshape(classes, (100000,3))

#Create model
model = Sequential()

#First layer
model.add(Dense(100))

#Second layer
model.add(Dense(40))

#Second layer
model.add(Dense(100))

#Third layer
model.add(Dense(40))

#Forth layer
model.add(Dense(3))

#Compile model
model.compile(loss='mean_absolute_error',optimizer='RMSprop',metrics=['accuracy'])

#Fit the model
model.fit(signals,classes,epochs=50,batch_size=20,verbose=2)

#Evaluate model
scores = model.evaluate(signals,classes)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

new_sign = np.loadtxt('try.dat')
new_sign = np.reshape(new_sign,(1,100))
new_class = model.predict(new_sign)

print(new_class)
print('[ 1.14860215  1.29455538  2.441472 ]')