from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping #not sure what this one does
from keras.models import Sequential, load_model
import keras.utils as ku #not sure what this is for

from random import randint
import pandas as pd #for use with text preprocessing
import numpy as np
import string, os

def preprocess():
    headlines = []
    cur_dir = './nyt-comments/'

    for filename in os.listdir(cur_dir):
        if 'Articles' in filename:
            #if '2017' in filename:
            #print(filename)
            article_df = pd.read_csv(cur_dir + filename)
            headlines.extend(list(article_df.headline.values))
    headlines = [h for h in headlines if h != "Unknown"]

    print('how many headlines: ', len(headlines))

    #arrays are not inherently mutable so use index to changes values
    for i in range(len(headlines)):
        #you can lowercase an entire string, it doesn't have to be word by word
        headlines[i] = "".join(v for v in headlines[i] if v not in string.punctuation).lower()
        headlines[i] = headlines[i].encode("utf8").decode("ascii", 'ignore')


    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(headlines) #creates an integer:word dictionary
    total_words = len(tokenizer.word_index) + 1
    print('vocab size: ', total_words)


    input_sequences = []
    for line in headlines:
        #turns each headline into its corresponding integer sequence
        token_list = tokenizer.texts_to_sequences([line])[0]

        #splits each headline into a sequence, so [1,2,3] would become [1,2][1,2,3]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    #data must be padded to be accepted by keras, AKA all sequences must have same length

    max_seq_len = max([len(seq) for seq in input_sequences])
    
    #pad_sequences(input, maxlen, padding) padding defaults to pre
    #why does it need to be pre and not post?
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len))

    print('num seq: ', len(input_sequences))

    #split into data and labels, where the label is the last word in the sequence
    predictors, label = input_sequences[:,:-1], input_sequences[:,-1]

    #transforms only the label data into one hot vectors
    label = ku.to_categorical(label, num_classes=total_words)

    #input_len = max_seq_len - 1
    
    return tokenizer, max_seq_len, headlines


def model():
    model = Sequential()

    #number of distinct words in training set, size of embedding vectors,
    #size of each input sequence
    #second param was 10 before
    model.add(Embedding(total_words, 32, input_length = input_len))

    model.add(LSTM(128))

    model.add(Dropout(0.2))

    #softmax squashes the values between 0 and 1 but makes sure they add to 1
    #the values are proprtional to the number of values
    model.add(Dense(total_words, activation='softmax'))

    #calculates with an ouput between 0 and 1
    model.compile(loss='categorical_crossentropy', optimizer = 'adam')

    model.summary()

    model.fit(predictors, label, epochs=50, verbose=1)

    model.save("headline-generator-v3.model")


def sample_c(preds, temperature=1.0):
    
    #helper function to sample an index from a probability array?
    print("the arguments, preds and temperature: ", preds, temperature)
    
    #turns preds into a numpy array continaing 64 bit floats
    preds = np.asarray(preds).astype('float64')
    
    #np.log takes the natural log of each element in peds
    #preds1 = np.log(preds) / temperature
    #print("log divided by temp: ", preds)
    #finds the exponential of each elements, which is just x^e where e~2.7
    #i think this is to make sure all the data is positive?
    #exp_preds = np.exp(preds1)
    #print("exponentials of preds: ", exp_preds)
    #print("")
    exp_preds = preds ** (1.0/temperature)
    #print("other exp of preds: ", preds2)
    
    #normalizes each element based on sum of exp_preds
    preds = exp_preds / np.sum(exp_preds)
    print("normalized preds: ", preds)
    #takes experiments with one of p possible outcomes, in this case, preds is the 
    #probabilites of each of the p different outcomes, p is equal to size of vocab
    probas1 = np.random.multinomial(5, preds, 1)
    print("five experiments: ", np.argmax(probas1))
    #the more experiments performed, the more likely the probabilities get closer to the actual probabilities
    probas = np.random.multinomial(1, preds, 1)
    #these are arrays of mostly zeroes and a one
    #print("preds after multinomial occurs: ", preds)
    #print("multinomial probabilities: ", probas)
    chosen_index = 0
    for index in range(len(probas[0])):
        if probas[0][index] == 0:
            pass
        else:
            #chosen_index = index
            print("probas at index ", index, ": ", probas[0][index])
    print("prediction val at index before: ", preds[chosen_index-1])
    print("prediction value at chosen prob: ", preds[chosen_index])
    print("prediction val at index after: ", preds[chosen_index+1])
    print("return index of max: ", np.argmax(probas))
    return np.argmax(probas1)


def sample(preds, diversity, experiments):
    preds = np.asarray(preds).astype('float64')
    exp_preds = preds ** (1.0/diversity)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(experiments, preds, 1)
    return np.argmax(probas)

'''
num_words: the length of the generation

seed_text: the input text

diversity: the larger this value, the more likely to get different generations

headlines: the training data
'''
def gen_text_version_three(num_words, seed_text, diversity, headlines, experiments):
    repeat = 0
    #this for loop iterates the length of the headline you want to generate
    for i in range(num_words):
        #counts the number of repeated phrases in the generation
        for line in headlines:
            if seed_text in line:
                #print("found seed text in headline: ", line)
                repeat += 1
                break
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        #print(token_list)
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1)
        #print(predicted)
        predicted = model.predict(token_list, verbose=0)[0]
        next_index = model.predict_classes(token_list, verbose=0)[0]
        #print("predicted class index before sampling using predict classes: ", next_index)
        #predict_classes will always return the same values when used alone
        #this list of predictions does not change each time when 
        #the sample method is not called
        #print('prediction from model: ', predicted)
        #print('argmax index from .predict: ', np.argmax(predicted))
        next_index = sample(predicted, diversity, experiments)
        #print("predicted index after sampling: ", next_index)
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                #print(word)
                seed_text += " " + word
        #print("")
        #print("")
    if repeat <=10:
        print(seed_text, ': ', repeat)
    #print("the text repeated ", repeat, " times")

def gen_text_version_four(num_words, seed_text, diversity, headlines, experiments):
    repeat = 0
    #this for loop iterates the length of the headline you want to generate
    for i in range(num_words):
        #counts the number of repeated phrases in the generation
        for line in headlines:
            if seed_text in line:
                #print("found seed text in headline: ", line)
                repeat += 1
                break

        
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1)
        
        predicted = model.predict(token_list, verbose=0)[0]
        
        #print("this is token ", i, " with diversity ", diversity[i])
        next_index = sample(predicted, diversity[i], experiments)
        
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                #print(word)
                seed_text += " " + word
    if repeat <=10:
        #print(seed_text, ': ', repeat)
        return seed_text
    #print("the text repeated ", repeat, " times")


def test_diff_diversity(seed_text, experiments, num_words, headlines):

    div_list = [[1.2, 1.3, 1.4, 1.5, 1.6], [2.2, 2.3, 2.4, 2.5, 2.6]]
    for diversity in div_list:
        #print("")
        #print('start diversity: ', diversity[0], ': ')
        for i in range(5):
            gen_headline = gen_text_version_four(num_words, seed_text, diversity, headlines, experiments)
            print(gen_headline)
            #output_file = open('gen_headlines.txt', 'a')
            #print(gen_headline, end='\n', file=output_file)


if __name__ == '__main__':
    #np.random.seed(2) #set the seed for deterministic generation
    tokenizer, max_seq_len, headlines = preprocess()
    
    model = load_model("headline-generator-v3.model")

    for i in range(0,60):
        print('iteration: ', i, '/100')
        rando = randint(0,len(tokenizer.word_index.items()))
        #print('random index: ', rando)
        for word, index in tokenizer.word_index.items():
            if index == rando:
                #print('random word: ', word)
                rando_word = word
                break


        #seed text, num experiments, num words, headline training data
        test_diff_diversity(word, 1, 5, headlines)

    #test_diff_diversity("trump", 1, 5, headlines)
    #if there are more tokens in a row, increase diversity
    #start with high diversity and decrease as the string builds
    #train a dataset with the newline character
    #make a classifier for headlines, if they are real headlines or fake





