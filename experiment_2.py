'''For this next test, I will remove the fourth number (the sum) entirely, although the targets will remain the same.

My hypothesis is that the model accuraccy score will go down, but the model will not be fooled as easily. I would
expect an even distribution of weights across the three input nodes. It could also be possilbe that this model fails
and is unable to predict with confidence whether the sum of the values will be positive or negative.'''

from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np


min_range = -10
max_range = 10
data_size = 10000
x_data = []
y_data = []

for i in range(data_size):
    a = random.randint(min_range, max_range)/10.0
    b = random.randint(min_range, max_range)/10.0
    c = random.randint(min_range, max_range)/10.0
    num_sum = a+b+c
    x_data.append([a,b,c])
    if num_sum > 0:
        y_data.append(1)
    else:
        y_data.append(0)

index = int(.9 * data_size)

train_x = np.array(x_data[:index])
test_x = np.array(x_data[index:])

train_y = np.array(y_data[:index])
test_y = np.array(y_data[index:])

print(train_x.shape)

#built this model in multiples of 3, as in layers have 3/6 nodes
model = Sequential()

model.add(Dense(3, input_dim=3, activation='relu'))

model.add(Dense(6, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(train_x, train_y, epochs=10, batch_size=10)

val_loss, val_acc = model.evaluate(test_x, test_y)
print('validation loss: ' , val_loss)
print('validation accuraccy: ', val_acc)

#test 1: numbers within the trained range add to positive sum, should predict 1
test1 = [3, 7, 8]
prediction = model.predict(np.array([test1]))
print(prediction)

#test 2: numbers outside the trained range add to positive sum, should predict 1
test2 = [13, 100, 999]
prediction = model.predict(np.array([test2]))
print(prediction)

#test 3: numbers within the trained range add to negative sum, should predict 0
test3 = [-1, -5, -9]
prediction = model.predict(np.array([test3]))
print(prediction)

#test 4: numbers outside the trained range add to negative sum, should predict 0
test4 = [-26, -24, -420]
prediction = model.predict(np.array([test4]))
print(prediction)

#test 5: numbers add up to a positive sum should predict 1
test5 = [6, 9, 13]
prediction = model.predict(np.array([test5]))
print(prediction)

#test 6: numbers add up to a negative sum should predict 0
test6 = [-27, -5, -789]
prediction = model.predict(np.array([test6]))
print(prediction)

#test 7: numbers of different signage with two negatives ,add up to a positive sum, should predict 1
test7 = [-13, -6, 20]
prediction = model.predict(np.array([test7]))
print(prediction)

#test 8: numbers of different signage with two positives ,add up to a negative sum, should predict 0
test8 = [-40, 12, 13]
prediction = model.predict(np.array([test8]))
print(prediction)

#test 9: numbers of different signage with two positives ,add up to a positive sum, should predict 1
test9 = [10, -12, 13]
prediction = model.predict(np.array([test9]))
print(prediction)

#test 10: numbers of different signage with two negatives ,add up to a negative sum, should predict 0
test= [-1, 3, -5]
prediction = model.predict(np.array([test]))
print(prediction)

#this is the first test with a different result, and although it is extremely close to 0, it is not quite there
#I wonder if it's the values being so close together?

for a in range(-3, 3):
    for b in range(-3, 3):
        for c in range(-3, 3):
            test = [a,b,c]
            prediction = model.predict(np.array([test]))
            print('For array ', test, ', prediction is: ', prediction[0][0])

'''This was an interesting experiment because it proved that I may have been underestimating the computer!
My hypothesis sorta went both ways, since I was unsure of whether the computer would arbitrarily put more weight into
one of the three nodes. However, the computer more accurately predicted the nature of the sum, since it did not have the fourth
sum value to trick it or throw it off. This leads me to believe that the weights are more evenly distributed across the model
and not so biased towards the fourth value. Another interesting thing to notice is that when the input numbers got smaller
and the sum became closer to zero, the model didn't predict a binary 1 or 0, rather it predicted a numbers approaching
0. When the sum was zero, the model predicted a value of around .5, since 0 is neither positive or negative.

This could mean that one way of countering biases is to remove superfluous data from the training set, which basically
means more preprocessing.'''

model.get_weights()
#the three input neurons have similarly distributed weights

'''[array([[-1.2705514 , -0.8968036 ,  1.3700829 ],
        [-1.256617  , -0.9582146 ,  1.3618464 ],
        [-1.2969731 , -0.90007716,  1.3827934 ]], dtype=float32),
 array([0.6878937 , 0.25054237, 0.61361134], dtype=float32),
 array([[-1.0456553 , -1.2399656 , -0.45384285, -0.77578616, -0.4916236 ,
          2.3122048 ],
        [-0.20156188, -1.2574689 , -0.46764502, -1.4913299 , -0.0040701 ,
          2.2130272 ],
        [ 1.8996539 ,  2.0216684 , -0.28537136,  1.7966127 , -0.662626  ,
         -1.141871  ]], dtype=float32),
 array([0.15761513, 0.51752365, 0.        , 0.3123651 , 0.        ,
        0.7036136 ], dtype=float32),
 array([[ 2.2458177 ],
        [ 2.168303  ],
        [-0.61742496],
        [ 1.9504585 ],
        [ 0.36813092],
        [-1.6358504 ]], dtype=float32),
 array([-0.7169651], dtype=float32)]'''
