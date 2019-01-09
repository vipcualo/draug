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
import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2
from datetime import datetime
import visdom
class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )

TABSY = "\t"
figdir = "figures/"




def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, FLAGS, dataset):
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

                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets)

    # print("Test Set len", str(len(test_set)))
    # print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                         label_col_inds,
                                                                                         measure, FLAGS,
                                                                                         train_sets, test_sets)

    testperf = all_predictions[bestparamind]  ##pointer pos

    print("---FINAL RESULTS-----")
    print("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam))

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

    print("Test Performance CI", FLAGS)
    print(testperfs, FLAGS)
    print("Test Performance MSE", FLAGS)
    print(testloss)

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
        XDinput=self.embeddingXD(XDinput)
        XDinput = torch.transpose(XDinput,2,1)
        #print(Embedding.shape)
        encode_smiles=F.relu(self.conv1XD(XDinput))
        encode_smiles = F.relu(self.conv2XD(encode_smiles))
        encode_smiles = F.relu(self.conv3XD(encode_smiles))
        #print(encode_smiles.shape)
        encode_smiles = F.max_pool1d(encode_smiles, kernel_size=encode_smiles.size()[2:])
        encode_smiles = encode_smiles.view(encode_smiles.shape[0], encode_smiles.shape[1])
        #encode_smiles = torch.mean(encode_smiles.view(encode_smiles.size(0), encode_smiles.size(1), -1), dim=2)
        XTinput = self.embeddingXT(XTinput)
        XTinput = torch.transpose(XTinput, 2, 1)
        encode_protein = F.relu(self.conv1XT(XTinput))
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



def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, prfmeasure, FLAGS, labeled_sets,
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
        train_drugs = np.array(train_drugs)
        train_prots = np.array(train_prots)
        train_Y = np.array(train_Y)
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        # print("terows", str(terows), str(len(terows)))
        # print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)
        val_drugs=np.array(val_drugs)
        val_prots=np.array(val_prots)
        val_Y=np.array(val_Y)
        val_drugs=val_drugs[:1000]
        val_prots=val_prots[:1000]
        val_Y=val_Y[:1000]
        pointer = 0
        print(paramset1)
        print(paramset2)
        print(paramset3)
        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]
                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]
                    model = Net(param2value, param1value,param3value)
                    print("param ",param2value," ",param1value," ",param3value)
                    model.cuda()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(),lr=0.001)
                    predicted_labels = []
                    for i in range(epoch):
                        loss_epoch=0
                        model.train()
                        for j in range(0,int(len(train_drugs)),batchsz):
                            optimizer.zero_grad()
                            end=min(j+batchsz,len(train_drugs))
                            train_drug_batch=train_drugs[j:end]
                            train_prot_batch=train_prots[j:end]
                            target=train_Y[j:end]
                            target=torch.FloatTensor(target)
                            target=target.cuda()
                            train_drug_batch = torch.tensor(train_drug_batch, dtype=torch.long)
                            train_drug_batch = train_drug_batch.cuda()
                            train_prot_batch = torch.tensor(train_prot_batch, dtype=torch.long)
                            train_prot_batch = train_prot_batch.cuda()
                            output = model(train_drug_batch,train_prot_batch)
                            loss = criterion(output,target)
                            loss.backward()
                            optimizer.step()
                            loss_epoch+=loss.item()*len(train_drug_batch)
                        model.eval()
                        loss_eval=0
                        for j in range(0,int(len(val_drugs)),batchsz):
                            end = min(j + batchsz, len(val_drugs))
                            train_drug_batch = val_drugs[j:end]
                            train_prot_batch = val_prots[j:end]
                            target = val_Y[j:end]
                            target = torch.FloatTensor(target)
                            target = target.cuda()
                            train_drug_batch = torch.tensor(train_drug_batch, dtype=torch.long)
                            train_drug_batch = train_drug_batch.cuda()
                            train_prot_batch = torch.tensor(train_prot_batch, dtype=torch.long)
                            train_prot_batch = train_prot_batch.cuda()
                            output = model(train_drug_batch, train_prot_batch)
                            loss = criterion(output, target)
                            loss_eval += loss.item() * len(train_drug_batch)
                            if i==(epoch-1):
                                if len(predicted_labels)==0:
                                    predicted_labels=output.cpu().detach().numpy()
                                else :
                                    predicted_labels = np.concatenate((predicted_labels, output.cpu().detach().numpy()), 0)
                        print("epoch ", i, " , train loss ", loss_epoch * 1.0 / len(train_drugs),"  , vali loss ",loss_eval/len(val_drugs))
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]
                    print("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, MSE = %f" %
                           (param1ind, param2ind, param3ind, foldind, rperf, loss_eval/len(val_drugs)))
                    all_predictions[pointer][foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind] = loss
                    pointer += 1
    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                # print(epoch, batchsz, avgperf)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind, param3ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses

def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select



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


def experiment(FLAGS, perfmeasure, foldcount=6):  # 5-fold cross validation + test

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
                                                                    perfmeasure, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    perfmeasure = get_cindex
    experiment(FLAGS, perfmeasure)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
    FLAGS.log_dir = "tmp1545016336.2369363/"
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS)
