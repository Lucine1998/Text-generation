from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint # will help us save model at best training epoch
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import tensorflow
import keras
import pandas as pd
import numpy as np

tokenizer = Tokenizer() # keras tonekizer class object

def text_to_sequence_ngram(corpus):

    '''
    get numerical value for each word of corpus and put them in list
    the numerical value represents the index of the word in the vocabulary of the corpus
    '''

    tokenizer.fit_on_texts(corpus) #fit on texts is mandatory before any pre-processing with keras
    total_words = len(tokenizer.word_index) + 1
    
    # create n_grams of number sequences
    sequences = []

    for headline in corpus:
        tokens = tokenizer.texts_to_sequences([headline])[0] #transforms each word of text to a sequence of integers => assigns number value to word based on their index in the corpus
        for i in range(1, len(tokens)):
            n_gram = tokens[:i+1]
            sequences.append(n_gram)
        #print(f"{headline} : {n_gram}")

    return sequences, total_words

def generate_padded_sequences(sequences):

    '''
    since all sequences aren't of the same length, we need to 'pad' the sequences (= make them equal in length)
    predictors = the word(s) before the word we want to predict, aka the words that will 'predict' the presence of another word
    label = the word we want to predict next, after the predictors
    say we want to predict a word after "i am"... "i am" are the predictors and the word we want to predict next is the label
    '''

    max_sequence_len = max([len(x) for x in sequences]) #define the max length of our sequence as the length of the longest sequence (here: 10)
    sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')) #fonction keras pad_sequences
    
    predictors, label = sequences[:,:-1],sequences[:,-1]
    label = to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    '''
    create our model for text generation
    '''
    model = Sequential() # keras model : "a plain stack of layers where each layer has exactly one input tensor and one output tensor"
    # so there HAS to be only one input and one output !
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=(max_sequence_len - 1)))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def generate_text(predictors, next_words, model, max_sequence_len):

    for __ in range(next_words):

        tokens = tokenizer.texts_to_sequences([predictors])[0] # sequence of the input predictors
        tokens = pad_sequences([tokens], maxlen=(max_sequence_len-1), padding='pre') # pad them as well
        predicted = model.predict(tokens, verbose=0) # verbose=1 will output a progression bar, verbose=0 will output nothing
        predicted = np.argmax(predicted, axis=1) # numpy function: "Returns the indices of the maximum values along an axis."
        
        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        predictors += " " + output_word

    return predictors.title()

if __name__ == '__main__':

    dataset = pd.read_csv('news-headlines.csv')
    corpus = dataset['headline_text'][:50000]

    sequences, total_words = text_to_sequence_ngram(corpus)

    predictors, label, max_sequence_len = generate_padded_sequences(sequences)

    model = create_model(max_sequence_len, total_words)

    filepath = 'lstm-text-generation-model.hdf5'

    '''determining optimal number of epochs'''
    es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

    checkpoint = ModelCheckpoint(filepath=filepath, 
                                monitor='val_loss',
                                verbose=1, 
                                save_best_only=True,
                                mode='min')

    history = model.fit(predictors, label, batch_size=10, epochs=100, callbacks=[es, checkpoint], validation_split=0.2) # epoch = number of time the model goes through training corpus

    # plot the training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.savefig('model_training_history') # save graph
    plt.show()

    model = keras.models.load_model(filepath) #load model back

    print(generate_text("Donald Trump", 10, model, max_sequence_len))