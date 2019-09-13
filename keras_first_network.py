from keras.models import Sequential
from keras.layers import Dense
import numpy
import math

#for result reproducability, aka for setting the initial weights stochastically?
numpy.random.seed(7)

#each entry in the dataset has 9 tokens, with the last one being 0 if not
#onset of diabetes and 1 if onset of diabetes
dataset= numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#split the dataset into training and testing data, 90:10
index = math.ceil(len(dataset) * .90)
#print(index)
train_data = dataset[:index]
test_data = dataset[index:]
#print(test_data)

#X contains all of neurons in the input layer (8 input neurons) for each item
X = train_data[:,0:8]

#Y contains the output values for the items in the dataset
Y = train_data[:,8]

model = Sequential()
#adding layers to the base sequential model

'''Dense means the layers are fully connected;
the first parameter is the number of neurons in the layer;
input_dim is the number of expected input parameters for the first layer;
the weights in the network are set between 0 and 0.05, which is the default uniform
weight initilaization in keras;
the activation functions used are relu rectifiers because it achieves better performance
than a sigmoid;
sigmoid is used at the end to achieve a binary value of 0 or 1 '''
model.add(Dense(12, input_dim=8, activation='relu')) #input layer
model.add(Dense(8, activation='relu')) #a hidden layer
model.add(Dense(1, activation='sigmoid')) #output layer

'''Compiling the model:
   The loss parameter evaluates the set of weights (in this case, logarithmic loss,
   which in a binary example like this is binary_crossentropy).
   The optimizer param searches through the weights in the network, 'adam' is
   a default efficient gradient descent algorithm.
   The metrics param is an additional param for collecting data about training.'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''Fitting the model:
   Epochs (250) are the number of iterations the model trains through the dataset/
   The batch_size (10) is the number of instances that are evaluated before a weight update
   to the weights in the network is performed.'''
model.fit(X, Y, epochs=150, batch_size=10) #use training data for fitting

testX = test_data[:,0:8]
testY = test_data[:,8]

#evaluate the model on the testing data
scores = model.evaluate(testX, testY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
