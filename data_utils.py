#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np
import tensorflow as tf
import sklearn
import scipy




def get_split_data(data, split_prop=0.8):
    split_idx = int(len(data) * split_prop)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def get_nth_difference(data, n=1):
    return np.diff(data, n=n)


def get_smoothed_gauss_1d(data, sigma):
    return scipy.ndimage.filters.gaussian_filter1d(data, sigma=sigma)


def get_moving_average(data, window_size):
    return np.convolve(data, np.ones((window_size)) / window_size, mode='valid')

def get_shifted_sequences(seq, offset):
    last_possible_idx = len(seq) - offset
    seq1 = seq[:last_possible_idx]
    seq2 = seq[offset:]
    return seq1, seq2


def get_random_shifted_seq_batch(data_sequence, batch_size, seq1_length, seq2_length):
    """ Function to retrieve randomly places minibatches of sequences from a sequence of datapoints

    Args:
        data_sequence: The total sequence of datapoints to sample from
        batch_size:    The number of sequences in the returned minibatch
        seq1_length:    The length of the sequence presented to the model, the x_sequence length
        seq2_length:     The number of time steps shifted from the x_sequence to get the y_sequence, the sequence the model will try to predict

    Returns:
        batch_seq1: batch_size number of sequences of length seq_length
        batch_seq2: batch_size number of sequences of length seq_shift following the  values of seqs_x in data_sequences with
                    seq_overlap number of overlapping values from the end of sequences in seq_x

    """

    data_length = len(data_sequence)

    # Get batch_size random numbers being start_idxs where the sequences start.
    last_possible_idx = data_length - seq1_length - seq2_length  # Make sure no index is to close to the end to get outofbounds error
    start_idxs = np.random.uniform(0, last_possible_idx, size=batch_size)  # Sample random positions in data_sequence
    start_idxs = list(map(lambda x: int(x), start_idxs))  # Make positions integer start_idxs

    batch_seq1 = []
    batch_seq2 = []
    for idx in start_idxs:
        # Store sequences of size seq1_length starting at the found start_idxs
        seq1 = data_sequence[idx:idx + seq1_length]
        # Store sequences of length seq2_length that follow the sequences in seq1
        seq2 = data_sequence[idx + seq1_length:idx + seq1_length + seq2_length]

        batch_seq1.append(seq1)
        batch_seq2.append(seq2)

    return batch_seq1, batch_seq2


'''
    Usefull pandas functions
    
    # Calculating the long-window simple moving average
    df.rolling(window=100).mean()
    
    # Using Pandas to calculate a 20-days span EMA (exponential moving average. adjust=False specifies that we are 
        interested in the recursive calculation mode.
    df.ewm(span=20, adjust=False).mean()
'''


def get_random_shifted_seq_batch_v2(data_sequence, batch_size,
                                    size_per_step, steps_per_sample, steps_per_prediction):


    data_len = len(data_sequence)
    
    points_per_sample     = size_per_step * steps_per_sample
    points_per_prediction = size_per_step * steps_per_prediction

    last_possible_idx = data_len - points_per_sample - points_per_prediction

    start_idxs = np.random.uniform(0, last_possible_idx, size=batch_size)  # Sample random positions in data_sequence
    start_idxs = list(map(lambda x: int(x), start_idxs))  # Make positions integers

    batch_seq1 = []
    batch_seq2 = []
    for idx in start_idxs:
        # Store sequences of size seq1_length starting at the found start_idxs
        seq1 = data_sequence[idx:idx + points_per_sample]
        # Store sequences of length seq2_length that follow the sequences in seq1
        seq2 = data_sequence[idx + points_per_sample - 5:idx + points_per_sample + points_per_prediction - 5]

        batch_seq1.append(seq1)
        batch_seq2.append(seq2)

    batch_seq1 = np.array(batch_seq1).reshape([batch_size, steps_per_sample, size_per_step]
    )
    batch_seq2 = np.array(batch_seq2).reshape([batch_size, steps_per_prediction, size_per_step])


    # ## DEBUG CHECK
    # print(batch_seq1.shape)
    # print(batch_seq2.shape)
    # for x, y in zip(batch_seq1, batch_seq2):
    #     x = x.flatten()
    #     y = y.flatten()
    #     z = np.concatenate((x, y))
    #     plt.plot(z)
    #
    # xcoord = [steps_per_sample * size_per_step]
    # plt.axvline(x=xcoord, color='k')
    # plt.show()



    return batch_seq1, batch_seq2





