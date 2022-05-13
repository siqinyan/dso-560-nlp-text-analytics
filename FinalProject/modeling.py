#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


import argparse
import numpy as np
import pandas as pd
from typing import Dict
from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Masking
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import SimpleRNN, LSTM

# from clr import LRFinder, OneCycleLR



def integer_encode_documents(docs, tokenizer):
    return tokenizer.texts_to_sequences(docs)

def plot_fit_history(log):
    fig, ax = plt.subplots()
    plt.plot(log.history['acc'])
    plt.plot(log.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
#     plt.show()
    plt.tight_layout()

    return fig

def load_glove_vectors() -> Dict:
    embeddings_index = {}
    with open('../datasets/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def make_mlp_classification_model(docs_padded, vocab_size, EMBEDDING_SIZE):
    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=docs_padded.shape[1]))
    model.add(Flatten())

    # model.add(Dense(256, activation='sigmoid')) 
    # model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    model.summary()
    
    return model


def make_lstm_classification_model(embedding_matrix, vocab_size, MAX_SEQUENCE_LENGTH, plot=False):
        model = Sequential()
        model.add(Embedding(vocab_size, 100, 
                            weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH, 
                            trainable=False))
        model.add(Masking(mask_value=0.0)) # masking layer, masks any words that don't have an embedding as 0s.
        model.add(LSTM(units=32, input_shape=(1, MAX_SEQUENCE_LENGTH)))
        model.add(Dense(16))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
                optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # summarize the model
        model.summary()
        
        if plot:
            plot_model(model, to_file='model.png', show_shapes=True)
        return model

# define model
def make_binary_classification_rnn_model(embedding_matrix, vocab_size, MAX_SEQUENCE_LENGTH, plot=False):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, 
                        weights=[embedding_matrix], 
                        input_length=MAX_SEQUENCE_LENGTH, 
                        trainable=False))
    model.add(Masking(mask_value=0.0)) # masking layer, masks any words that don't have an embedding as 0s.
    model.add(SimpleRNN(units=64, input_shape=(1, MAX_SEQUENCE_LENGTH)))
    model.add(Dense(16))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    model.summary()
    
    if plot:
        plot_model(model, to_file='model.png', show_shapes=True)
    return model



def main(args): 

    ############# set hyperparameters #############
    path = args.path
    MAX_SEQUENCE_LENGTH = args.max_sequence_length
    NUM_WORDS = args.num_words
    BATCH_SIZE = args.batch_size
    model_type = args.model
    des = args.description
    EMBEDDING_SIZE = args.embedding_size
    if model_type=='mlp':
        model_name = f'{model_type}_{des}_es{EMBEDDING_SIZE}'
    else:
        model_name = f'{model_type}_{des}_msl{MAX_SEQUENCE_LENGTH}'

    es_patience = 7
    num_epochs = 50


    ########################## DATA ##########################
    data = pd.read_csv(path+'steam_cleaned_v2.csv', index_col=0, na_filter=False)
    data['label'] = pd.to_numeric(data['label'])
    docs_cleaned = data['review'].tolist()
    labels = data['label']

    ############# tokenize #############
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="UNKNOWN_TOKEN")
    tokenizer.fit_on_texts(docs_cleaned)


    # integer encode the documents
    docs_encoded = integer_encode_documents(docs_cleaned, tokenizer)

    docs_padded = pad_sequences(docs_encoded, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    vocab_size = int(len(tokenizer.word_index) * 1.3)
    # print(f"Vocab size is {vocab_size} unique tokens.")
    ######## TODO:save print into log files ########


    data_docs_padded = pd.DataFrame(docs_padded)
    data_docs_padded = pd.concat([data_docs_padded.reset_index(drop=True),
                                data.reset_index(drop=True)], axis=1, ignore_index=True)

    mp = {data_docs_padded.columns[-1]: 'label',
        data_docs_padded.columns[-2]: 'text'}
    data_docs_padded.rename(columns=mp, errors="raise", inplace=True)



    X_train, X_test, y_train, y_test = train_test_split(docs_padded, labels, 
                                                        test_size=0.2, random_state=41)
    # X_train.shape, X_test.shape


    ########################## MODEL ##########################

    if model_type=='mlp':
        ##### self-trained embedding #####
        model = make_mlp_classification_model(docs_padded, vocab_size, EMBEDDING_SIZE)
    else:
        ##### pre-trained embedding #####
        embeddings_index = load_glove_vectors()
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: # check that it is an actual word that we have embeddings for
                embedding_matrix[i] = embedding_vector

        if model_type=='rnn':
            model = make_binary_classification_rnn_model(embedding_matrix, vocab_size, MAX_SEQUENCE_LENGTH, plot=False)
        elif model_type=='lstm':
            model = make_lstm_classification_model(embedding_matrix, vocab_size, MAX_SEQUENCE_LENGTH, plot=False)



    ############# training #############
    # fit the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=es_patience, verbose=1)
    # lr_manager = OneCycleLR(num_samples=len(X_train), batch_size=BATCH_SIZE, max_lr=1e-2,
    #                         end_percentage=0.1, scale_percentage=None,
    #                         maximum_momentum=0.95, minimum_momentum=0.85)
                            
    log = model.fit(X_train, y_train, batch_size=BATCH_SIZE, validation_split=0.1, 
    #                 epochs=50, verbose=1, callbacks=[es, lr_manager])
                    epochs=num_epochs, verbose=1, callbacks=[es])


    # evaluate the model
    loss_tr, acc_tr = model.evaluate(X_train, y_train, verbose=1)
    print('TRAIN Accuracy: %f' % (acc_tr*100))

    fig = plot_fit_history(log)
    fig.savefig(f'./Data/plots/log_{model_name}.png', bbox_inches='tight')

    ############# testing #############
    loss_tt, acc_tt = model.evaluate(X_test, y_test, verbose=1)
    print('TEST Accuracy: %f' % (acc_tt*100))



    ########################## RESULTS ##########################
    # pred = model.predict(docs_padded)
    # pred = pred.flatten()

    # threshold = .5
    # data[f'pred_{model_name}'] = np.where(pred>=threshold, 1, 0)
    # data.to_csv('./Data/steam_cleaned.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Model for Steam Reviews")
    

    parser.add_argument('-p', "--path", type=str, default='./Data/')

    parser.add_argument('-msl', "--max_sequence_length", type=int, default=150)
    parser.add_argument('-nw', "--num_words", type=int, default=5000)

    parser.add_argument("-m", '--model', type=str, default='mlp')
    parser.add_argument('-bs', "--batch_size", type=int, default=64)
    parser.add_argument('-emd_size', "--embedding_size", type=int, default=64)
    # parser.add_argument('-des', "--description", type=str, default='')
    parser.add_argument('-des', "--description", type=str)


    args = parser.parse_args()
    main(args)

