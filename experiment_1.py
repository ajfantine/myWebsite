'''This experiement is designed to test how a simple deep neural net learns and figures out weights.
The data is a four-number array, with the fourth number being the sum of the first three, and the target
being a 1 if the sum is positive and a 0 if the sum is negative.

My hypothesis is that the model will place the heaviest weights on the fourth number, since that determines the
target value.

I can test this by providing it an input where the fourth number is not the sum but rather the opposite, and if it makes
its prediction based on that alone, I know it is puting more weight on that feature.'''

from keras.models import Sequential
from keras.layers import Dense
import random


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
    x_data.append([a,b,c,num_sum])
    if num_sum > 0:
        y_data.append(1)
    else:
        y_data.append(0)

index = int(.9 * data_size)
train_x = x_data[:index]
test_x = x_data[index:]
train_y = y_data[:index]
test_y = y_data[index:]

print(train_x[0])
print(train_y[0])

import numpy as np

model= Sequential()

model.add(Dense(12, input_dim=4, activation='relu')) #input layer

model.add(Dense(8, activation='relu')) # one hidden layer

model.add(Dense(1, activation='sigmoid')) #output layer

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

model.fit(np.array(train_x), np.array(train_y), epochs=10, batch_size=12)

val_loss, val_acc = model.evaluate(np.array(test_x), np.array(test_y))
print(val_loss, val_acc) #loss = .02, acc = .99
#overall pretty great scores, lets test and see if my hypothesis is correct

#test 1: numbers within the trained range add to positive sum, should predict 1
test1 = [3, 7, 8, 18]
prediction = model.predict(np.array([test1]))
print(prediction)

#test 2: numbers outside the trained range add to positive sum, should predict 1
test2 = [13, 100, 999, 1112]
prediction = model.predict(np.array([test2]))
print(prediction)

#test 3: numbers within the trained range add to negative sum, should predict 0
test3 = [-1, -5, -9, -15]
prediction = model.predict(np.array([test3]))
print(prediction)

#test 4: numbers outside the trained range add to negative sum, should predict 0
test4 = [-26, -24, -420, -470]
prediction = model.predict(np.array([test4]))
print(prediction)

#test 5: numbers add up to a positive sum, but the fourth number is a negative sum, should predict 1
test5 = [6, 9, 13, -10]
prediction = model.predict(np.array([test5]))
print(prediction)

#test 6: numbers add up to a negative sum, but the fourth number is a positive sum, should predict 0
test6 = [-27, -5, -78, 50]
prediction = model.predict(np.array([test6]))
print(prediction)

'''Just as I expected, the model has put most of the weight on the fourth number in the sequence and thus
can be easily fooled if that number is not an accurate sum. '''

new_model.get_weights()
#each layer is comprised of two arrays, one containing the neurons and their weights,
#the other containing the biases of each neuron
#notice in this model, the fourth input feature has heavier weights compared to the other three

'''output = [array([[-0.3376998 , -0.07662546,  0.25845477, -0.21493459,  0.21099603,
         -0.7086218 ,  0.44575873, -0.07553332,  0.2604803 , -0.29137275,
         -0.06426664, -0.00394071],
        [-0.38501784,  0.34631118, -0.2267133 , -0.44255874,  0.3364517 ,
         -0.33121136, -0.08362699,  0.21198475,  0.6367184 ,  0.0464736 ,
          0.64913964,  0.42474064],
        [-0.5502918 ,  0.2989374 ,  0.5152765 , -0.22142376,  0.2501408 ,
          0.47990787,  0.22030617, -0.6500378 ,  0.31368575, -0.22235024,
          0.62364787, -0.0263606 ],
        [-1.9926615 ,  1.0205095 ,  0.7482599 , -1.6924745 ,  2.047673  ,
         -2.0208368 ,  1.5888737 , -1.0587057 ,  1.6864334 , -1.7889286 ,
          0.41409034,  1.7860037 ]], dtype=float32),
 array([0.4002214 , 0.0478525 , 0.32690117, 0.37918535, 0.37721   ,
        0.2156248 , 0.39161336, 0.00185018, 0.39764878, 0.4447414 ,
        0.41260657, 0.14769828], dtype=float32),
 array([[-1.0936117 ,  1.0559564 ,  0.37589446,  0.6713635 ,  0.7457316 ,
         -0.16003546, -0.20950831, -1.0450882 ],
        [ 0.7061817 , -0.6310257 ,  0.11495946, -0.3644773 , -0.6496878 ,
          0.08021338, -0.42498326,  0.02620237],
        [ 0.45119262,  0.22282949,  0.26961547, -0.3862202 ,  0.23766893,
         -0.33474413, -0.5810658 ,  0.7824437 ],
        [-0.3885166 ,  0.2545973 ,  1.0437952 ,  1.051436  ,  0.7178736 ,
         -0.51340437, -0.3013528 , -0.08393191],
        [ 1.1756164 , -0.3371364 , -0.6529685 , -0.69988215, -0.7593062 ,
         -0.18009672, -0.1398301 ,  0.8076441 ],
        [-0.3458773 ,  0.52123314,  0.05243704,  0.22519134,  0.34349716,
         -0.33185038, -0.34091824, -0.36536083],
        [ 0.8686688 , -0.44599876, -0.10676776, -0.1268699 , -0.5276506 ,
         -0.21176599,  0.33935508,  1.2665352 ],
        [-0.3641122 ,  0.28205267,  0.34353217,  0.06952313,  0.19194414,
          0.4853603 , -0.3610013 , -0.69559366],
        [ 0.6128468 , -0.31454214,  0.03950627, -0.39685282,  0.12898912,
         -0.42557168, -0.5446137 ,  0.93390346],
        [-0.7683159 ,  0.89827967,  0.79998773,  0.44946197,  0.813586  ,
         -0.03856986,  0.15033096, -0.67496765],
        [ 0.7680406 , -0.23308666, -0.15516356,  0.4990753 ,  0.24087486,
          0.24015786,  0.3639546 ,  0.6899885 ],
        [ 0.9033077 , -0.3698882 , -0.12580255, -0.01303204, -0.5326952 ,
         -0.15892881,  0.13305716,  0.7218688 ]], dtype=float32),
 array([ 0.53476644,  0.250458  , -0.03201656,  0.2344991 ,  0.2515367 ,
        -0.05105212, -0.12003176,  0.3805325 ], dtype=float32),
 array([[ 0.9364545 ],
        [-1.395778  ],
        [-0.99005544],
        [-1.497177  ],
        [-1.0508037 ],
        [ 0.15132934],
        [-0.27846524],
        [ 1.1606947 ]], dtype=float32),
 array([0.21866016], dtype=float32)]'''
