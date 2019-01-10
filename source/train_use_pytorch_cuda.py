# -*- coding: utf8 -*-
from __future__ import print_function
# khai bao thu vien
import numpy as np
import random as rn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
from datahelper import *
from arguments import argparser, logging
import sys, pickle, os
import math, json, time
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2
from datetime import datetime
TABSY = "\t"
figdir = "figures/"




def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, measure, FLAGS, dataset): # tao du lieu training , validation, va testing chayj theo k fold
    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS.dataset_path, FLAGS.problem_type)  # du lieu training va testing

    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    # logger.info('Start training')
    # phan chia du lieu thanh training, testing v
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

    # chon hyperparameter tot nhat với tập dữ liệu training và validation
    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT, Y,
                                                                                                          label_row_inds,
                                                                                                          label_col_inds,
                                                                                                          measure,
                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets)

    # print("Test Set len", str(len(test_set)))
    # print("Outer Train Set len", str(len(outer_train_sets)))
    # chon hyperparameter tot nhat voi tap du lieu training va testing
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                         label_col_inds,
                                                                                         measure, FLAGS,
                                                                                         train_sets, test_sets)

    testperf = all_predictions[bestparamind]  ##pointer pos
    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %     # Kết quả tham số tốt nhất traning dùng tập training và validation
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind] # lấy ra kết quả CI loss của mỗi tổ hợp hyperparameter
        foldloss = all_losses[bestparamind][test_foldind] # lấy ra kết quả MSE loss của mỗi tố hợp hyperparameter
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf # tính CI loss của tất cả quá trình traninng của các model

    avgperf = avgperf / len(test_sets) # tính CI loss trung bình
    avgloss = np.mean(testloss) # tính loss mse
    teststd = np.std(testperfs) # tính độ lệch chuẩn

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd



class Net(nn.Module):
    def __init__(self,FILTER_LENGTH1,NUM_FILTERS,FILTER_LENGTH2):
        super(Net, self).__init__()
        self.embeddingXD=nn.Embedding(65,128) # layer embedding cho XDinput
        self.conv1XD = nn.Conv1d(128, NUM_FILTERS, FILTER_LENGTH1, padding=0) # sử dụng conv1d với 32 filter(NUM_FILTERS=32) , ko dùng padding zeros , kernel_size= FILTER_LENGTH1
        torch.nn.init.xavier_uniform(self.conv1XD.weight) # khởi tạo trọng số cho layer conv1XD sử dụng công thức xavier
        self.conv2XD = nn.Conv1d(NUM_FILTERS, NUM_FILTERS*2, FILTER_LENGTH1, padding=0) # sử dụng conv1d với 64  filter , ko dùng padding zeros , kernel_size= FILTER_LENGTH1
        torch.nn.init.xavier_uniform(self.conv2XD.weight) # khởi tạo trọng số cho layer conv2XD sử dụng công thức xavier
        self.conv3XD = nn.Conv1d(NUM_FILTERS*2, NUM_FILTERS*3, FILTER_LENGTH1, padding=0) # sử dụng conv1 với 96 filter , ko dùng padding zeros , kernel_size= FILTER_LENGTH1
        torch.nn.init.xavier_uniform(self.conv3XD.weight) # khởi tạo trọng số cho layer conv3XD sử dụng công thức xavier

        self.embeddingXT = nn.Embedding(26, 128) # layer embedding cho XTinput
        self.conv1XT = nn.Conv1d(128, NUM_FILTERS, FILTER_LENGTH2, padding=0) # sử dụng conv1d với 32 filter(NUM_FILTERS=32) , ko dùng padding zeros , kernel_size= FILTER_LENGTH2
        torch.nn.init.xavier_uniform(self.conv1XT.weight) # khởi tạo trọng số cho layer conv1XT sử dụng công thức xavier
        self.conv2XT = nn.Conv1d(NUM_FILTERS, NUM_FILTERS * 2, FILTER_LENGTH2, padding=0) # sử dụng conv1d với 64  filter , ko dùng padding zeros , kernel_size= FILTER_LENGTH2
        torch.nn.init.xavier_uniform(self.conv2XT.weight) # khởi tạo trọng số cho layer conv1XT sử dụng công thức xavier
        self.conv3XT = nn.Conv1d(NUM_FILTERS * 2, NUM_FILTERS * 3, FILTER_LENGTH2, padding=0) # sử dụng conv1 với 96 filter , ko dùng padding zeros , kernel_size= FILTER_LENGTH2
        torch.nn.init.xavier_uniform(self.conv3XT.weight) # khởi tạo trọng số cho layer conv1XT sử dụng công thức xavier

        self.fc1 = nn.Linear(192, 1024) # layer fully connected với 1024 unit
        torch.nn.init.xavier_uniform(self.fc1.weight)  # khởi tạo trọng số cho layer fc1 sử dụng công thức xavier
        self.fc2 = nn.Linear(1024, 1024) # layer fully connected với 1024 unit
        torch.nn.init.xavier_uniform(self.fc2.weight) # khởi tạo trọng số cho layer fc1 sử dụng công thức xavier
        self.fc3 = nn.Linear(1024, 512) # layer fully connected với 512 unit
        torch.nn.init.xavier_uniform(self.fc3.weight) # khởi tạo trọng số cho layer fc1 sử dụng công thức xavier
        self.fc4= nn.Linear(512,1,) #  output layer fully connected với 1 unit
        torch.nn.init.normal(self.fc4.weight) # khởi tạo trọng số cho layer fc1 sử dụng công thức normal
        self.dropout = nn.Dropout(0.1) # dropout sau layer fc1 để tránh overfitting
        self.dropout2 = nn.Dropout(0.1) # dropout sau layer fc2 để tránh overfitting

    def forward(self, XDinput,XTinput):
        XDinput=self.embeddingXD(XDinput) # tính layer embedding của XDinput
        XDinput = torch.transpose(XDinput,2,1) # chuyển từ batch_size,step,channel => batch,channel,step
        encode_smiles=F.relu(self.conv1XD(XDinput)) # tinh kết quả sau khi forward XDinput qua conv1XD và hàm kích hoạt relu
        encode_smiles = F.relu(self.conv2XD(encode_smiles)) # tinh kết quả sau khi forward encode_smiles qua conv2XD và hàm kích hoạt relu
        encode_smiles = F.relu(self.conv3XD(encode_smiles)) # tinh kết quả sau khi forward encode_smiles qua conv3XD và hàm kích hoạt relu
        encode_smiles = F.max_pool1d(encode_smiles, kernel_size=encode_smiles.size()[2:])  # tính global maxpooling1D cho từng channel của encode_smiles
        encode_smiles = encode_smiles.view(encode_smiles.shape[0], encode_smiles.shape[1]) # chuyển từ batch_size,channel,1 => batch_size, channel
        #encode_smiles = torch.mean(encode_smiles.view(encode_smiles.size(0), encode_smiles.size(1), -1), dim=2)

        XTinput = self.embeddingXT(XTinput)  # tính layer embedding của XTinput
        XTinput = torch.transpose(XTinput, 2, 1) # chuyển từ batch_size,step,channel => batch,channel,step
        encode_protein = F.relu(self.conv1XT(XTinput)) # tinh kết quả sau khi forward XTinput qua conv1XT và hàm kích hoạt relu
        encode_protein = F.relu(self.conv2XT(encode_protein)) # tinh kết quả sau khi forward encode_protein qua conv1XT và hàm kích hoạt relu
        encode_protein = F.relu(self.conv3XT(encode_protein)) # tinh kết quả sau khi forward encode_protein qua conv1XT và hàm kích hoạt relu
        encode_protein = F.max_pool1d(encode_protein, kernel_size=encode_protein.size()[2:])  # tính global maxpooling1D cho từng channel của encode_protein
        encode_protein = encode_protein.view(encode_protein.shape[0], encode_protein.shape[1]) # chuyển từ batch_size,channel,1 => batch_size, channel

        encode_interaction=torch.cat((encode_smiles, encode_protein), 1) # concatenate encode_smiles và encode_protein
        x=F.relu(self.fc1(encode_interaction)) # forward encode_interaction qua fully connected 1
        x=self.dropout(x) # sử dụng dropout để tránh onverfitting
        x = F.relu(self.fc2(x)) # forward encode_interaction qua fully connected 2
        x = self.dropout2(x) # sử dụng dropout để tránh onverfitting
        x = F.relu(self.fc3(x)) # forward encode_interaction qua fully connected 2
        x = self.fc4(x) # tính kết quả dự đoán
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
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        train_drugs = np.array(train_drugs)
        train_prots = np.array(train_prots)
        train_Y = np.array(train_Y)
        #train_drugs=train_drugs[:1000]
        #train_prots=train_prots[:1000]
        #train_Y=train_Y[:1000]
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)
        val_drugs=np.array(val_drugs)
        val_prots=np.array(val_prots)
        val_Y=np.array(val_Y)
        #val_drugs=val_drugs[:1000]
        #val_prots=val_prots[:1000]
        #val_Y=val_Y[:1000]
        pointer = 0
        print(paramset1)
        print(paramset2)
        print(paramset3)
        for param1ind in range(len(paramset1)):  # chọn số filter của các mạng convolution
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # chọn kernel_size cho các mạng convolution phần tính encode_smiles
                param2value = paramset2[param2ind]
                for param3ind in range(len(paramset3)): # chọn kernel_size cho các mạng convolution phần tính encode_protein
                    save_model_path="../model_weight/fold"+str(foldind)+"_param"+str(param1ind)+"_param2ind"+str(param2ind)+str(param3ind)+".pt" # đường dẫn lưu model
                    param3value = paramset3[param3ind]
                    model = Net(param2value, param1value,param3value) # khởi tạo model
                    print("param ",param2value," ",param1value," ",param3value) # in ra tham số
                    model.cuda() # chúng ta dùng cuda để training nên mình chọn cuda
                    criterion = nn.MSELoss() # khởi tạo function tính hàm loss , mean square error
                    optimizer = optim.Adam(model.parameters(),lr=0.001) # chọn giải thuật Adam là giải thuật tối ưu, dùng learning rate = 0.001
                    best_predicted_labels = [] # lưu kết quả của dự đoán tập validation tốt nhất
                    best_vali_loss=100000000 # lưu loss vali tốt nhất
                    for i in range(epoch): # vòng lặp epoch
                        loss_epoch=0 # loss train theo từng epoch
                        model.train() # bắt đầu training
                        for j in range(0,int(len(train_drugs)),batchsz): # vòng lặp step
                            optimizer.zero_grad() # restart lại gradisent
                            end=min(j+batchsz,len(train_drugs))
                            train_drug_batch=train_drugs[j:end] # lấy dữ liệu drug training theo step
                            train_prot_batch=train_prots[j:end] # lấy dữ liệu prots otstraining theo step
                            target=train_Y[j:end] # lấý label theo steps
                            target=torch.FloatTensor(target) # chuyển numpy array thanh tensor
                            target=target.cuda() # khai báo dùng cuda
                            train_drug_batch = torch.tensor(train_drug_batch, dtype=torch.long) # chuyển numpy array thanh tensor
                            train_drug_batch = train_drug_batch.cuda() # chuyển sang tensor cuda
                            train_prot_batch = torch.tensor(train_prot_batch, dtype=torch.long) # chuyển numpy array thanh tensor
                            train_prot_batch = train_prot_batch.cuda() # chuyển sang tensor cuda
                            output = model(train_drug_batch,train_prot_batch) # tính y dự đoán ( forward)
                            loss = criterion(output,target) # tính loss step
                            loss.backward() # tính gradisent trên từng parameter
                            optimizer.step() # back propagation
                            loss_epoch+=loss.item()*len(train_drug_batch) # cộng loss từng step cho loss theo epoch
                        model.eval() # khai báo để chuẩn bị đánh giá mô hình, tắt traning, dropout=0,...
                        loss_eval=0
                        predicted_labels=[]
                        for j in range(0,int(len(val_drugs)),batchsz):
                            end = min(j + batchsz, len(val_drugs))
                            vali_drug_batch = val_drugs[j:end] # lấy dữ liệu drug validation theo step
                            vali_prot_batch = val_prots[j:end] # lấy dữ liệu prots validation theo step
                            target = val_Y[j:end]  # lấý label theo steps
                            target = torch.FloatTensor(target) # chuyển numpy array thanh tensor
                            target = target.cuda() # khai báo dùng cuda
                            vali_drug_batch = torch.tensor(vali_drug_batch, dtype=torch.long) # chuyển numpy array thanh tensor
                            vali_drug_batch = vali_drug_batch.cuda() # chuyển sang tensor cuda
                            vali_prot_batch = torch.tensor(vali_prot_batch, dtype=torch.long) # chuyển numpy array thanh tensor
                            vali_prot_batch = vali_prot_batch.cuda() # chuyển sang tensor cuda
                            output = model(vali_drug_batch, vali_prot_batch) # tính y dự đoán ( forward)
                            loss = criterion(output, target) # tính loss step
                            loss_eval += loss.item() * len(vali_drug_batch) # cộng loss từng step cho loss theo epoch
                            if len(predicted_labels)==0: # lưu toạn bộ y dự đoán
                                    predicted_labels=output.cpu().detach().numpy()
                            else :
                                    predicted_labels = np.concatenate((predicted_labels, output.cpu().detach().numpy()), 0)
                        if best_vali_loss > (loss_eval/len(val_drugs)):  # nếu loss ở epoch hiện tại tốt hơn best loss thì lưu model và lưu best predict
                            torch.save(model.state_dict(), save_model_path) # save model
                            best_vali_loss=loss_eval/len(val_drugs) # lưu best loss vali
                            best_predicted_labels=predicted_labels # lưu best predict
                            print("epoch ", i, " , train loss ", loss_epoch * 1.0 / len(train_drugs),"  , vali loss ",loss_eval/len(val_drugs))

                    rperf = prfmeasure(val_Y, best_predicted_labels) # tính CI-i
                    rperf = rperf[0]

                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, MSE = %f" % # ghi log
                           (param1ind, param2ind, param3ind, foldind, rperf, best_vali_loss),FLAGS)

                    all_predictions[pointer][foldind] = rperf  # lưu lại chỉ số rperf của từng lần training  k fold và các tham số
                    all_losses[pointer][foldind] = best_vali_loss # lưu lại chỉ số loss của từng lần training k fold và các tham số khác nhau
                    pointer += 1
    bestperf = -float('Inf')
    bestpointer = None # chọn ra bộ hyper parameter tốt nhất
    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):
            for param3ind in range(len(paramset3)):
                avgperf = 0.
                # tính avgperf trung bình của từng  bộ hyper parameter ( lấy trung bình theo k =5)
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                # print(epoch, batchsz, avgperf)
                if avgperf > bestperf: # chọn avgperf tốt nhất và lưu các bộ tham số
                    bestperf = avgperf # lưu lại avgperf tốt nhất
                    bestpointer = pointer # lưu lại lần training tốt nhất
                    best_param_list = [param1ind, param2ind, param3ind] # lưu lại bộ hyper parameter tốt nhấ
                pointer += 1
    # trả về các bộ tham số tốt nhất và CI-i
    return bestpointer, best_param_list, bestperf, all_predictions, all_losses

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


    dataset = DataSet(fpath=FLAGS.dataset_path,  # class lấy dataset
                      setting_no=FLAGS.problem_type,
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(fpath=FLAGS.dataset_path)

    XD = np.asarray(XD) # chuyển sang array
    XT = np.asarray(XT) # chuyển sang array
    Y = np.asarray(Y) # chuyển sang array

    drugcount = XD.shape[0] # lấy số lượng dữ liệu XD
    print(drugcount)
    targetcount = XT.shape[0] # lấy số lượng dữ liệu XT
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    perfmeasure, FLAGS, dataset) # training

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    perfmeasure = get_cindex # khai báo hàm định nghĩa loss CI
    experiment(FLAGS, perfmeasure)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"
    FLAGS.log_dir = "tmp1545016336.2369363/" # khai báo path lưu log
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS) # bắt đầu tiến trình training với các bộ hyperparameter khác nhau
