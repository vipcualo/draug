from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
### We modified Pahikkala et al. (2014) source code for cross-val process ###

import os

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from datahelper import *
# import logging
from itertools import product
from arguments import argparser, logging

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, GRU
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, merge, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers

import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

TABSY = "\t"
figdir = "figures/"


def build_combined_onehot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len, FLAGS.charsmiset_size))
    XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))

    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)  # pool_size=pool_length[i]

    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein])
    # encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    predictions = Dense(1, kernel_initializer='normal')(FC2)

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    interactionModel.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[cindex_score])  # , metrics=['cindex_score']

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_onehot.png')

    return interactionModel


def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    print("hihi ", FLAGS.charsmiset_size + 1)
    print("haha ", FLAGS.max_smi_len)
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=FLAGS.charseqset_size + 1, output_dim=128, input_length=FLAGS.max_seq_len)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein],
                                                  axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(
        FC2)  # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[cindex_score])  # , metrics=['cindex_score']
    print(interactionModel.summary())
    # plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel


def build_single_drug(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()
    XTmodel = Sequential()
    XTmodel.add(Activation('linear', input_shape=(FLAGS.target_count,)))

    encode_smiles = Sequential()
    encode_smiles.add(Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len))
    encode_smiles.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                             strides=1))  # input_shape=(MAX_SMI_LEN, SMI_EMBEDDING_DIMS)
    encode_smiles.add(
        Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid', strides=1))
    encode_smiles.add(
        Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid', strides=1))
    encode_smiles.add(GlobalMaxPooling1D())

    interactionModel.add(Merge([encode_smiles, XTmodel], mode='concat', concat_axis=1))
    # interactionModel.add(layers.merge.Concatenate([XDmodel, XTmodel]))

    # Fully connected
    interactionModel.add(Dense(1024, activation='relu'))  # 1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))  # 1024
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_drug.png')

    return interactionModel


def build_single_prot(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()
    XDmodel = Sequential()
    XDmodel.add(Activation('linear', input_shape=(FLAGS.drugcount,)))

    XTmodel1 = Sequential()
    XTmodel1.add(Embedding(input_dim=FLAGS.charseqset_size + 1, output_dim=128, input_length=FLAGS.max_seq_len))
    XTmodel1.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                        strides=1))  # input_shape=(MAX_SEQ_LEN, SEQ_EMBEDDING_DIMS)
    XTmodel1.add(
        Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid', strides=1))
    XTmodel1.add(
        Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid', strides=1))
    XTmodel1.add(GlobalMaxPooling1D())

    interactionModel.add(Merge([XDmodel, XTmodel1], mode='concat', concat_axis=1))

    # Fully connected
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_single_protein.png')

    return interactionModel


def build_baseline(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    interactionModel = Sequential()

    XDmodel = Sequential()
    XDmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.drug_count,)))

    XTmodel = Sequential()
    XTmodel.add(Dense(1, activation='linear', input_shape=(FLAGS.target_count,)))

    interactionModel.add(Merge([XDmodel, XTmodel], mode='concat', concat_axis=1))

    # Fully connected
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(1024, activation='relu'))
    interactionModel.add(Dropout(0.1))
    interactionModel.add(Dense(512, activation='relu'))

    interactionModel.add(Dense(1, kernel_initializer='normal'))
    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score])

    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_baseline.png')

    return interactionModel


def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, runmethod, FLAGS, dataset):
    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS.dataset_path, FLAGS.problem_type)

    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    # logger.info('Start training')
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT, Y,
                                                                                                          label_row_inds,
                                                                                                          label_col_inds,
                                                                                                          measure,
                                                                                                          runmethod,
                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets)

    # print("Test Set len", str(len(test_set)))
    # print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                         label_col_inds,
                                                                                         measure, runmethod, FLAGS,
                                                                                         train_sets, test_sets)

    testperf = all_predictions[bestparamind]  ##pointer pos

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd



class Net(nn.Module):
    def __init__(self,FILTER_LENGTH1,NUM_FILTERS,FILTER_LENGTH12):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.embeddingXD=nn.Embedding(65,128)
        self.conv1XD = nn.Conv1d(128, 32, FILTER_LENGTH1, padding=0)
        torch.nn.init.xavier_uniform(self.conv1XD.weight)
        self.conv2XD = nn.Conv1d(NUM_FILTERS, NUM_FILTERS*2, FILTER_LENGTH1, padding=0)
        torch.nn.init.xavier_uniform(self.conv2XD.weight)
        self.conv3XD = nn.Conv1d(NUM_FILTERS*2, NUM_FILTERS*3, FILTER_LENGTH1, padding=0)
        torch.nn.init.xavier_uniform(self.conv3XD.weight)

        self.embeddingXT = nn.Embedding(26, 128)
        self.conv1XT = nn.Conv1d(128, NUM_FILTERS, FILTER_LENGTH12, padding=0)
        torch.nn.init.xavier_uniform(self.conv1XT.weight)
        self.conv2XT = nn.Conv1d(NUM_FILTERS, NUM_FILTERS * 2, FILTER_LENGTH12, padding=0)
        torch.nn.init.xavier_uniform(self.conv2XT.weight)
        self.conv3XT = nn.Conv1d(NUM_FILTERS * 2, NUM_FILTERS * 3, FILTER_LENGTH12, padding=0)
        torch.nn.init.xavier_uniform(self.conv3XT.weight)

        self.fc1 = nn.Linear(192, 1024)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(1024, 1024)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(1024, 512)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        self.fc4= nn.Linear(512,1,)
        torch.nn.init.normal(self.fc4.weight)
        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, XDinput,XTinput):
        #print(XDinput)
        Embedding1=self.embeddingXD(XDinput)
        Embedding3 = Embedding1
        Embedding1 = torch.transpose(Embedding1,2,1)
        #print(Embedding.shape)
        encode_smiles=F.relu(self.conv1XD(Embedding1))
        encode_smiles = F.relu(self.conv2XD(encode_smiles))
        encode_smiles = F.relu(self.conv3XD(encode_smiles))
        #print(encode_smiles.shape)
        encode_smiles = F.max_pool1d(encode_smiles, kernel_size=encode_smiles.size()[2:])
        encode_smiles = encode_smiles.view(encode_smiles.shape[0], encode_smiles.shape[1])
        #encode_smiles = torch.mean(encode_smiles.view(encode_smiles.size(0), encode_smiles.size(1), -1), dim=2)
        Embedding = self.embeddingXT(XTinput)
        Embedding = torch.transpose(Embedding, 2, 1)
        encode_protein = F.relu(self.conv1XT(Embedding))
        encode_protein = F.relu(self.conv2XT(encode_protein))
        encode_protein = F.relu(self.conv3XT(encode_protein))
        #encode_protein = torch.mean(encode_protein.view(encode_protein.size(0), encode_protein.size(1), -1), dim=2)
        encode_protein = F.max_pool1d(encode_protein, kernel_size=encode_protein.size()[2:])
        encode_protein = encode_protein.view(encode_protein.shape[0], encode_protein.shape[1])
        encode_interaction=torch.cat((encode_smiles, encode_protein), 1)
        #print("ha ",encode_interaction.shape)
        x=F.relu(self.fc1(encode_interaction))
        x=self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets,
                     val_sets):  ## BURAYA DA FLAGS LAZIM????

    paramset1 = FLAGS.num_windows  # [32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths  # [4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths  # [8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch  # 100
    batchsz = FLAGS.batch_size  # 256

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        # print("trrows", str(trrows), str(len(trrows)))
        # print("trcols", str(trcols), str(len(trcols)))

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        # print("terows", str(terows), str(len(terows)))
        # print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0

        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]
                    train_drugs=np.array(train_drugs)
                    train_prots=np.array(train_prots)
                    train_Y=np.array(train_Y)
                    model = Net(param2value, param1value,param3value)
                    print("param ",param2value," ",param1value," ",param3value)
                    model.cuda()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(),0.001)
                    for i in range(epoch):
                        loss_epoch=0
                        model.train()
                        st=randint(0,batchsz)
                        for j in range(st,len(train_drugs),batchsz):
                            optimizer.zero_grad()
                            end=min(j+batchsz,len(train_drugs))
                            data=train_drugs[j:end]
                            data2=train_prots[j:end]
                            target=train_Y[j:end]
                            target=torch.FloatTensor(target)
                            target=target.cuda()
                            data = torch.tensor(data, dtype=torch.long)
                            data = data.cuda()
                            data2 = torch.tensor(data2, dtype=torch.long)
                            data2 = data2.cuda()
                            output = model(data,data2)
                            loss = criterion(output,target)
                            loss.backward()
                            optimizer.step()
                            loss_epoch+=loss.item()*len(data)

                        print("epoch ",i," ,loss ",loss_epoch*1.0/len(train_drugs))
                        model.eval()
                        loss_eval=0
                        for j in range(0,len(val_drugs),batchsz):
                            end=min(j+batchsz,len(val_drugs))
                            data=val_drugs[j:end]
                            data2=val_prots[j:end]
                            data = torch.tensor(data, dtype=torch.long)
                            data = data.cuda()
                            data2 = torch.tensor(data2, dtype=torch.long)
                            data2 = data2.cuda()
                            target=val_Y[j:end]
                            target = torch.FloatTensor(target)
                            target = target.cuda()
                            output = model(data,data2)
                            loss = criterion(output, target)
                            loss_eval+=loss.item()*len(data)
                        print("val ",loss_eval*1.0/len(val_drugs))


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select


def plotLoss(history, batchind, epochind, param3ind, foldind):
    figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_" + str(foldind) + "_" + str(
        time.time())
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6):  # 5-fold cross validation + test

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet(fpath=FLAGS.dataset_path,  ### BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.problem_type,  ##BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(fpath=FLAGS.dataset_path)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    perfmeasure = get_cindex
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
    FLAGS.log_dir = "tmp1545016336.2369363/"
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS)
