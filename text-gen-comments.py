from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
import keras.utils as ku

import pandas as pd
import numpy as np
import string,os

comments = []
cur_dir = './nyt-comments/'

for filename in os.listdir(cur_dir):
    if 'Comments' in filename:
        comment_df = pd.read_csv(cur_dir + filename, dtype={"userURL": object, "userTitle": object})
        comments.extend(list(comment_df.commentBody.values))
        break #with the break, use only April 2017 comments, there are 243,832

#there are 2,176,364 comments total in this dataset across all files

for i in range(len(comments)):
    comments[i] = "".join(char for char in comments[i] if char not in string.punctuation).lower()
    comments[i] = comments[i].encode("utf8").decode("ascii", 'ignore')

smaller_data = []

for comment in comments:
    if len(comment) <= 50:
        smaller_data.append(comment)
print(len(smaller_data))

comments = smaller_data


tokenizer = Tokenizer()

tokenizer.fit_on_texts(comments)
total_words = len(tokenizer.word_index) + 1 #there are 332,226 words

input_sequences = []
for comment in comments:
    token_list = tokenizer.texts_to_sequences([comment])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

#print(input_sequences[0])

'''
small_input = []

for seq in input_sequences:
    if len(seq) <= 50:
        small_input.append(seq)

print(len(small_input)) #this generates over 9 million sequences, when comments is left to its full length!
'''


max_seq_len = max([len(seq) for seq in input_sequences])
#print("largest sequence length: ", max_seq_len)


'''
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_seq_len))

predictors, label = input_sequences[:,:-1], input_sequences[:,-1]

label = ku.to_categorical(label, num_classes = total_words)

#print(predictors[:10], label[:10])

input_len = max_seq_len -1

model = Sequential()

model.add(Embedding(total_words, 10, input_length = input_len))
model.add(LSTM(128))
model.add(Dropout(0.1))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam')

model.fit(predictors, label, epochs=100, verbose=1)
model.save("comment-generator-v1.model")
'''


model = load_model('comment-generator-v1.model')

def generate_comment(num_words, input_text):
    for i in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1)
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word
    return input_text

print(generate_comment(5, "I"))
print(generate_comment(13, "I"))
print(generate_comment(13, "You"))
print(generate_comment(13, "Trump"))
print(generate_comment(13, "The"))
print(generate_comment(13, "I want"))
print(generate_comment(15, "Do you think"))
print(generate_comment(10, "In"))
print(generate_comment(10, "Republicans"))
