import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

from src.utils import *
from src.rnn import *

#---------------------------------------------------------------------
# build a parametric ROM 
#---------------------------------------------------------------------
class pROM():

    # initial setup of this class
    def __init__(self, dir, output_dir, fname, cfg):

        self.data_dir = dir
        self.output_dir = output_dir
        self.fname = fname
        self.cfg = cfg
        self.ns = self.cfg["ns"]
        self.nstep = self.cfg["nstep"]
        self.nforcast = self.cfg["nforcast"]
        self.aspect = self.cfg["aspect"]
        self.t0 = time.time()

        if self.cfg["seq_s"] == True:
            self.nstep_s = self.nstep
        else:
            self.nstep_s = 1

        self.new_dir = "{}/{}".format(self.output_dir, self.fname)
        if os.path.exists(self.new_dir):
            #shutil.rmtree(self.new_dir)
            pass
        else:
            os.makedirs(self.new_dir)
            print(">> CAUTION: A new folder is created for this model.")

    # Hankel matrix needed for RNN training
    def hankel(self, datax, datay):
        X, y = list(), list()
        i_start = 0
        for _ in range(len(datax)):
            i_end = i_start + self.nstep
            o_end = i_end + self.nforcast
            if o_end <= len(datax):
                x_input = datax[i_start:i_end, :]
                X.append(x_input)
                y.append(datay[i_end:o_end, :])
            i_start += 1

        return np.array(X), np.array(y)

    # load data
    def loadData(self, ID_lst, scale=True):

        # import parameters
        para_list = np.genfromtxt("{}/para.txt".format(self.data_dir), skip_header=1, delimiter=',', dtype=None, encoding=None)
        self.num_cases = len(para_list)
        num_columns = len(para_list[0])
        para = np.zeros((self.num_cases, num_columns))
        for i in range(self.num_cases):
            for j in range(1, num_columns):
                para[i, j] = para_list[i][j]
    
        # import generalized  coordiantes
        self.n_snapshots = np.zeros(self.num_cases)
        for j in range(len(ID_lst)): 
            for i in range(self.num_cases):
                Q_ = np.loadtxt('{}/q_C{}_{}.txt'.format(self.data_dir, i, ID_lst[j]), skiprows=0, delimiter=',').T
                if i == 0:
                    Q = Q_.copy()
                else:
                    Q = np.concatenate((Q, Q_), axis=0)

                # save number of snapshots for each case 
                # this will be used as indexing data later for each case
                if j == 0:
                    self.n_snapshots[i] = Q_.shape[0]

            if j == 0:
                self.Q = Q.copy()
            else:
                self.Q = np.concatenate((self.Q, Q), axis=1)

        # input parameters (design variables) are after the 3rd column in txt file.
        self.parameters = para[:, 4:]
        assert self.ns == self.parameters.shape[1], "WARNING: number of design parameters is not correct!"

        # if scale is required
        if (scale):
            self.scaler_features = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_parameters = MinMaxScaler(feature_range=(-1, 1))
            self.Q = self.scaler_features.fit_transform(self.Q)
            self.parameters = self.scaler_parameters.fit_transform(self.parameters)
            joblib.dump(self.scaler_features, '{}/scaler_feature_{}.gz'.format(self.new_dir, self.fname))
            joblib.dump(self.scaler_parameters, '{}/scaler_parameters.gz'.format(self.new_dir))

    # build dataset for training
    def buildDataset(self):

        # ncase: this is the last id number in the para.txt that used
        # to set which cases should be considered in the training. Then we can use the 
        # rest of data as unseen cases later.
        # Note: put unseen cases at the end of para.txt list, so you can cut-off using this option.

        # feature: number of generalized coordinates (or rank)
        self.nfeature_in = self.Q.shape[1]
        self.nfeature_out = self.nfeature_in

        # update cfg dictionary
        self.cfg["feature_size"] = self.nfeature_in

        # check the number of cases should be taken into training
        if self.cfg["ncase"] != None:
            self.ncase = self.cfg["ncase"]
        else:
            self.ncase = self.parameters.shape[0]


        # build dataset using Hankel
        self.hankel_idx = []
        i_start = 0

        for i in range(self.ncase):
            i_step = int(self.n_snapshots[i])
            i_end = i_start + i_step
            Qr = self.Q[i_start:i_end, :]

            # build hankel matrix
            x, y = self.hankel(Qr, Qr)
            assert x.shape[0] == y.shape[0], "size of x and y in the Hankel matrix are not the same"

            s = np.ones((x.shape[0], self.nstep_s, self.ns))
            s *= self.parameters[i, :]

            print("({}) Data chunk -> [{}, {}], Num of samples:{}".format(i, i_start, i_end, i_end - i_start))
            if i == 0:
                self.X = x.copy()
                self.Y = y.copy()
                self.S = s.copy()
                self.Xic = (x[0, :, :].reshape((-1, self.nstep, self.nfeature_in))).copy()
            else:
                self.X = np.vstack((self.X, x))
                self.Y = np.vstack((self.Y, y))
                self.S = np.vstack((self.S, s))
                self.Xic = np.vstack((self.Xic, x[0, :, :].reshape((-1, self.nstep, self.nfeature_in))))

            i_start = i_end
            self.hankel_idx.append(int(self.X.shape[0]))

        # unseen data if exists
        if self.ncase < self.parameters.shape[0]:

            for i in range(self.ncase, self.parameters.shape[0]):
                i_step = int(self.n_snapshots[i])
                i_end = i_start + i_step
                Qr = self.Q[i_start:i_end, :]

                # build hankel matrix
                x, y = self.hankel(Qr, Qr)
                assert x.shape[0] == y.shape[0], "size of x and y in the Hankel matrix are not the same"

                s = np.ones((x.shape[0], x.shape[1], self.ns))
                s *= self.parameters[i, :]

                print("({}) Unseen Data chunk -> [{}, {}], Num of samples:{}".format(i, i_start, i_end, i_end - i_start))
                if i == self.ncase:
                    self.X_unseen = x.copy()
                    self.Y_unseen = y.copy()
                    self.S_unseen = s.copy()
                    self.Xic_unseen = (x[0, :, :].reshape((-1, self.nstep, self.nfeature_in))).copy()
                else:
                    self.X_unseen = np.vstack((self.X_unseen, x))
                    self.Y_unseen = np.vstack((self.Y_unseen, y))
                    self.S_unseen = np.vstack((self.S_unseen, s))
                    self.Xic_unseen = np.vstack((self.Xic_unseen, x[0, :, :].reshape((-1, self.nstep, self.nfeature_in))))

                i_start = i_end

        return

    # split data into training and test datasets
    def splitData(self):

        # dimensions and numbers
        self.nsample = self.X.shape[0]
        self.ntest = int(self.nsample * self.cfg["test_ratio"])
        self.use_unseen = self.cfg["unseen_validation"]

        if self.cfg["splitOption"] == 'chunk':

            for i in range(len(self.hankel_idx)):

                i_start_test = self.hankel_idx[i] - self.ntest
                i_end_test = self.hankel_idx[i]

                i_start_train = self.hankel_idx[i-1]
                i_end_train = self.hankel_idx[i] - self.ntest

                if i == 0:
                    i_start_train = 0

                x_train, x_test = self.X[i_start_train:i_end_train, :, :], self.X[i_start_test:i_end_test, :, :]
                y_train, y_test = self.Y[i_start_train:i_end_train, :, :], self.Y[i_start_test:i_end_test, :, :]
                s_train, s_test = self.S[i_start_train:i_end_train, :, :], self.S[i_start_test:i_end_test, :, :]

                if i == 0:
                    self.X_train, self.X_test = x_train.copy(), x_test.copy()
                    self.Y_train, self.Y_test = y_train.copy(), y_test.copy()
                    self.S_train, self.S_test = s_train.copy(), s_test.copy()
                else:
                    self.X_train = np.vstack((self.X_train, x_train))
                    self.Y_train = np.vstack((self.Y_train, y_train))
                    self.S_train = np.vstack((self.S_train, s_train))

        elif self.cfg["splitOption"] == 'random':

            import random
            # build test dataset
            idx_test = random.sample(range(self.nsample), self.ntest)
            self.X_test = np.zeros((self.ntest, self.X.shape[1], self.X.shape[2]))
            self.Y_test = np.zeros((self.ntest, self.Y.shape[1], self.Y.shape[2]))
            self.S_test = np.zeros((self.ntest, self.S.shape[1], self.S.shape[2]))
            
            # build testing dataset
            c = 0
            for i in idx_test:
                self.X_test[c, :, :] = self.X[i, :, :]
                self.Y_test[c, :, :] = self.Y[i, :, :]
                self.S_test[c, :, :] = self.S[i, :, :]
                c += 1

            # build training dataset
            self.ntrain = self.nsample - self.ntest
            self.X_train = np.zeros((self.ntrain, self.X.shape[1], self.X.shape[2]))
            self.Y_train = np.zeros((self.ntrain, self.Y.shape[1], self.Y.shape[2]))
            self.S_train = np.zeros((self.ntrain, self.S.shape[1], self.S.shape[2]))
            c = 0
            for i in range(self.nsample):
                if i in idx_test:
                    pass
                else:
                    self.X_train[c, :, :] = self.X[i, :, :]
                    self.Y_train[c, :, :] = self.Y[i, :, :]
                    self.S_train[c, :, :] = self.S[i, :, :]
                    c += 1

        else:
            print("ERROR -> split method is not chosen correct.")
            exit()

        if (self.use_unseen):
            self.X_test = np.vstack((self.X_test, self.X_unseen))
            self.Y_test = np.vstack((self.Y_test, self.Y_unseen))
            self.S_test = np.vstack((self.S_test, self.S_unseen))

        # do you want to delete X and Y now?
        #del self.X, self.Y, self.S

        return

    # generate synthenized data
    def augmentData(self, aug=None, stage=1, level=1, sigma_x=0.0, sigma_s=0.0, verbose=True, augtest=False):
        # set stage = 0, to avoid augmentation in the train ing data.
        # level changes the amplitude of noises. You can activate it if you want different level of noises.

        # for training
        self.X_train_aug = self.X_train.copy()
        self.Y_train_aug = self.Y_train.copy()
        self.S_train_aug = self.S_train.copy()

        # for test
        self.X_test_aug = self.X_test.copy()
        self.Y_test_aug = self.Y_test.copy()
        self.S_test_aug = self.S_test.copy()

        # for IC
        self.Xic_aug = self.Xic.copy()
        self.Sic_aug = self.parameters[:self.ncase, :].copy()

        # set up level of augmentation
        if level < 1:
            level = 1

        if stage == 0:
            aug = None

        # some stuff about plotting
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        if aug != None:
            for i in range(stage):
                for j in range(level):

                    if aug == 'org':
                        noise_x_tr = (np.random.random(self.X_train.shape) - 0.5) * (level - j) * sigma_x
                        noise_s_tr = (np.random.random(self.S_train.shape) - 0.5) * (level - j) * sigma_s
                        noise_x_te = (np.random.random(self.X_test.shape) - 0.5) * (level - j) * sigma_x
                        noise_s_te = (np.random.random(self.S_test.shape) - 0.5) * (level - j) * sigma_s
                        noise_ic = (np.random.random(self.parameters[:self.ncase, :].shape) - 0.5) * (level - j) * sigma_s
                    elif aug == 'jitter':
                        noise_x_tr = np.random.normal(loc=0., scale=sigma_x, size=self.X_train.shape) * (level - j)
                        noise_s_tr = np.random.normal(loc=0., scale=sigma_s, size=self.S_train.shape) * (level - j)
                        noise_x_te = np.random.normal(loc=0., scale=sigma_x, size=self.X_test.shape) * (level - j)
                        noise_s_te = np.random.normal(loc=0., scale=sigma_s, size=self.S_test.shape) * (level - j)
                        noise_ic = np.random.normal(loc=0., scale=sigma_s, size=self.parameters[:self.ncase, :].shape) * (level - j)
                    else:
                        print(">> Warning: error with the augmentation method.")

                    noise_x_tr += self.X_train
                    noise_s_tr += self.S_train
                    noise_x_te += self.X_test
                    noise_s_te += self.S_test
                    noise_ic += self.parameters[:self.ncase, :]

                    # for train
                    self.X_train_aug = np.vstack((self.X_train_aug, noise_x_tr))
                    self.Y_train_aug = np.vstack((self.Y_train_aug, self.Y_train))
                    self.S_train_aug = np.vstack((self.S_train_aug, noise_s_tr))

                    # for IC
                    self.Sic_aug = np.vstack((self.Sic_aug, noise_ic))
                    self.Xic_aug = np.vstack((self.Xic_aug, self.Xic))

                    # if test data is augmented
                    if (augtest):
                        self.X_test_aug = np.vstack((self.X_test_aug, noise_x_te))
                        self.Y_test_aug = np.vstack((self.Y_test_aug, self.Y_test))
                        self.S_test_aug = np.vstack((self.S_test_aug, noise_s_te))                        

                    # some cheesy plots
                    ax1.plot(noise_x_tr[:20, 0, 0], '.b')
                    ax2.plot(noise_s_tr[:, 0], '.b')
        else:
            pass

        ax1.plot(self.X_train[:20, 0, 0], '.-r')
        ax2.plot(self.S_train[:, 0], '.-r')
        ax1.set_title("Augmented data")
        ax2.set_title("Augmented data")
        fig1.savefig('{}/Aug_plot_X_{}.pNG'.format(self.new_dir, self.fname))
        fig2.savefig('{}/Aug_plot_S_{}.pNG'.format(self.new_dir, self.fname))

        if (verbose):
            print("=================================================")
            print("BASE DATA")
            print("X data size (train)    -> ", self.X_train.shape)
            print("Y data size (train)    -> ", self.Y_train.shape)
            print("S data size (train)    -> ", self.S_train.shape)
            print("X data size (test)     -> ", self.X_test.shape)
            print("Y data size (test)     -> ", self.Y_test.shape)
            print("S data size (test)     -> ", self.S_test.shape)
            print("S ic data size (base)  -> ", self.parameters[:self.ncase, :].shape)
            print("X ic data size (base)  -> ", self.Xic.shape)
            print("=================================================")            
            print("AUGMENTED DATA")
            print("X data size (train)    -> ", self.X_train_aug.shape)
            print("Y data size (train)    -> ", self.Y_train_aug.shape)
            print("S data size (train)    -> ", self.S_train_aug.shape)
            print("X data size (test)     -> ", self.X_test_aug.shape)
            print("Y data size (test)     -> ", self.Y_test_aug.shape)
            print("S data size (test)     -> ", self.S_test_aug.shape)
            print("S ic data size (train) -> ", self.Sic_aug.shape)
            print("X ic data size (train) -> ", self.Xic_aug.shape)
            print("=================================================")
            print("UNSEEN DATA")
            if (self.use_unseen):
                print("X data size (unseen) -> ", self.X_unseen.shape)
                print("Y data size (unseen) -> ", self.Y_unseen.shape)
                print("S data size (unseen) -> ", self.S_unseen.shape)
            else:
                print("Unseen data are not used in this training procedure.")
            print("=================================================")

        return

    # model evaluation
    def evaluateModel(self):

        # load model
        model = keras.models.load_model('{}/LSTM{}.tf'.format(self.new_dir, self.fname))
        #model.summary()

        # evaluate train
        if self.aspect == '2A':
            y_prediction_train = model.predict([self.X_train_aug, self.S_train_aug])
        elif self.aspect == '1A':
            y_prediction_train = model.predict([self.X_train_aug])

        self.mse = np.mean(np.square( abs(self.Y_train_aug[:, 0, :] - y_prediction_train[:, 0, :]) ))
        print("Error of train -> mse:{:e} ".format(self.mse))

        # evaluate test
        if self.aspect == '2A':
            y_prediction_test = model.predict([self.X_test_aug, self.S_test_aug])
        elif self.aspect == '1A':
            y_prediction_test = model.predict([self.X_test_aug])
        
        self.mse = np.mean(np.square( abs(self.Y_test_aug[:, 0, :] - y_prediction_test[:, 0, :]) ))
        print("Error of test -> mse:{:e}".format(self.mse))

    # test model on data used (both train and test)
    def testModel_seenData(self, Niter_tot=1, eps=0):

        # load model
        model = keras.models.load_model('{}/LSTM{}.tf'.format(self.new_dir, self.fname))
        #model.summary()

        # start-end indices for self.X and self.Y for each case (seen)
        indx = []
        n_start, n_end = 0, 0
        for i in range(self.ncase):
            n_end += int(self.n_snapshots[i] - self.nstep - self.nforcast) + 1
            indx.append((n_start, n_end))
            n_start = n_end

        # loop over all seen cases
        for i in range(self.ncase):

            # set input parameters
            if self.aspect == '2A':
                S_input = np.ones((1, self.nstep_s, self.ns))
                S_input *= self.parameters[i, :]
                assert S_input.shape[2] == self.ns
                print(">>> Test #{} | model S_input={}".format(i, self.scaler_parameters.inverse_transform(S_input[:, 0, :])))
            elif self.aspect == '1A':
                print(">>> Test #{} | model S_input=[]".format(i))

            # define number of iterations for each case
            niter = int((self.n_snapshots[i] - 2 * self.nstep - self.nforcast) / self.nforcast)

            # build solution matrices
            X_in = np.zeros((1, self.nstep, self.nfeature_in))
            X_out = np.zeros((1, self.nforcast, self.nfeature_out))
            X_history = np.zeros((Niter_tot*niter * self.nforcast + self.nstep, self.nfeature_out))

            # set initial conditions
            X_in[0, :, :] = self.Xic[i, :, :] * (1 + (np.random.random(X_in.shape) - 0.5) * eps)
            X_history[:self.nstep, :] = X_in[0, :, :]

            # loop over time
            fp = Verbose(Niter_tot*niter)
            for iter in range(Niter_tot*niter):

                # solve
                if self.aspect == '2A':
                    X_out[0, :, :] = model.predict([X_in, S_input], verbose=0)
                elif self.aspect == '1A':
                    X_out[0, :, :] = model.predict([X_in], verbose=0)

                # update solutions
                X_in[0, :(self.nstep - self.nforcast), :] = X_in[0, self.nforcast:, :]
                X_in[0, (self.nstep - self.nforcast):, :] = X_out[0, :, :]
                X_history[self.nstep + iter * self.nforcast:self.nstep + (iter + 1) * self.nforcast, :] = X_out[0, :, :]
                fp.fprint()

            # scale back to real values and then save it
            X_history = self.scaler_features.inverse_transform(X_history)
            Y_exact = self.scaler_features.inverse_transform(self.Y[indx[i][0]:indx[i][1] - self.nstep, 0, :])

            # save data
            np.savetxt('{}/Qlstm{}_case{}.txt'.format(self.new_dir, self.fname, i), X_history, delimiter=',')            
            np.savetxt('{}/Qexact{}_case{}.txt'.format(self.new_dir, self.fname, i), Y_exact, delimiter=',')

            # add plot
            plt.clf() 
            plt.plot(Y_exact[:, :], '-k', label='exact')
            plt.plot(X_history[self.nstep:, :], '-r', label='predict', linewidth=0.5)
            plt.xlabel('Number of iterations')
            plt.ylabel('Values')
            plt.tight_layout()
            plt.savefig('{}/test_u{}_s{}.png'.format(self.new_dir, self.fname, i))
            plt.close()

            plt.clf() 
            plt.plot(Y_exact[:, 0], Y_exact[:, 1], '-b', label='exact', linewidth=2.5)
            plt.plot(X_history[:, 0], X_history[:, 1], '-r', label='predict', linewidth=0.75)
            plt.plot(X_history[:self.nstep, 0], X_history[:self.nstep, 1], '-og', label='IC', linewidth=2.5)
            plt.xlabel('Values')
            plt.ylabel('Values')
            plt.legend()
            plt.tight_layout()
            plt.savefig('{}/test_u{}_s{}_traj.png'.format(self.new_dir, self.fname, i))
            plt.close()

        return

    # test model for unseen cases that are not used in the training
    def testModel_UnseenData(self, Niter_tot=1, eps=0):

        # load model
        model = keras.models.load_model('{}/LSTM{}.tf'.format(self.new_dir, self.fname))
        #model.summary()

        # start-end indices for self.X and self.Y for each case (unseen)
        indx = []
        n_start, n_end = 0, 0
        for i in range(self.ncase, self.parameters.shape[0]):
            n_end += int(self.n_snapshots[i] - self.nstep - self.nforcast) + 1
            indx.append((n_start, n_end))
            n_start = n_end

        # loop over all unseen cases
        c = 0
        for i in range(self.ncase, self.parameters.shape[0]):

            # set input parameters
            if self.aspect == '2A':
                S_input = np.ones((1, self.nstep_s, self.ns))
                S_input *= self.parameters[i, :]
                assert S_input.shape[2] == self.ns
                print(">>> Test (Unseen) #{} | model S_input={}".format(i, self.scaler_parameters.inverse_transform(S_input[:, 0, :])))
            elif self.aspect == '1A':
                print(">>> Test (Unseen) #{} | model S_input=[]".format(i))

            # define number of iterations for each case
            niter = int((self.n_snapshots[i] - 2 * self.nstep - self.nforcast) / self.nforcast)

            # build solution matrices
            X_in = np.zeros((1, self.nstep, self.nfeature_in))
            X_out = np.zeros((1, self.nforcast, self.nfeature_out))
            X_history = np.zeros((Niter_tot*niter * self.nforcast + self.nstep, self.nfeature_out))

            # set initial conditions
            X_in[0, :, :] = self.X_unseen[indx[c][0], :, :] * (1 + (np.random.random(X_in.shape) - 0.5) * eps)
            X_history[:self.nstep, :] = X_in[0, :, :]

            # loop over time
            fp = Verbose(Niter_tot*niter)
            for iter in range(Niter_tot*niter):

                # solve
                if self.aspect == '2A':
                    X_out[0, :, :] = model.predict([X_in, S_input], verbose=0)
                elif self.aspect == '1A':
                    X_out[0, :, :] = model.predict([X_in], verbose=0)

                # update solutions
                X_in[0, :(self.nstep - self.nforcast), :] = X_in[0, self.nforcast:, :]
                X_in[0, (self.nstep - self.nforcast):, :] = X_out[0, :, :]
                X_history[self.nstep + iter * self.nforcast:self.nstep + (iter + 1) * self.nforcast, :] =  X_out[0, :, :]
                fp.fprint()

            # scale back to real values and then save it
            X_history = self.scaler_features.inverse_transform(X_history)
            Y_exact = self.scaler_features.inverse_transform(self.Y_unseen[indx[c][0]:indx[c][1] - self.nstep, 0, :])

            # save data
            np.savetxt('{}/Qlstm{}_case{}.txt'.format(self.new_dir, self.fname, i), X_history, delimiter=',')
            np.savetxt('{}/Qexact{}_case{}.txt'.format(self.new_dir, self.fname, i), Y_exact, delimiter=',')
            c += 1

            # add plot
            plt.clf() 
            plt.plot(Y_exact[:, :], '-k', label='exact')
            plt.plot(X_history[self.nstep:, :], '-r', label='predict', linewidth=0.5)
            plt.xlabel('Number of iterations')
            plt.ylabel('Values')
            plt.tight_layout()
            plt.savefig('{}/test_u{}_s{}.png'.format(self.new_dir, self.fname, i))
            plt.close()

            plt.clf() 
            plt.plot(Y_exact[:, 0], Y_exact[:, 1], '-b', label='exact', linewidth=2.5)
            plt.plot(X_history[:, 0], X_history[:, 1], '-r', label='predict', linewidth=0.75)
            plt.plot(X_history[:self.nstep, 0], X_history[:self.nstep, 1], '-og', label='IC', linewidth=2.5)
            plt.xlabel('Values')
            plt.ylabel('Values')
            plt.legend()
            plt.tight_layout()
            plt.savefig('{}/test_u{}_s{}_traj.png'.format(self.new_dir, self.fname, i))
            plt.close()
            plt.clf()

        return

    # run ROM for input parameters and niter steps
    def run(self, niter, S1_input, S2_input=None):

        # load files
        scaler_features = joblib.load('{}/scaler_feature_{}.gz'.format(self.new_dir, self.fname))
        scaler_parameters = joblib.load('{}/scaler_parameters.gz'.format(self.new_dir))
        ICmodel = keras.models.load_model('{}/IC{}.tf'.format(self.new_dir, self.fname))
        model = keras.models.load_model('{}/LSTM{}.tf'.format(self.new_dir, self.fname))

        # build history of input parameters  
        if S2_input == None:
            S2_input = S1_input
        else:
            pass

        # Hey: niter + 1 means the first set of parameters are for IC and adter that, niter iteraitons will proceed. Gotcha?
        S_input = real_time_parameters(S1_input, S2_input, niter+self.nstep, initial=int(niter/2), ramp=20, type='tanh')
        S_scaled = scaler_parameters.transform(S_input)

        # predict IC using input parameter and then scale it 
        q_scaled = ICmodel.predict(S_scaled[0, :].reshape(1, -1), verbose=0)[0, :, :]
        [self.nstep, self.nfeature_in] = q_scaled.shape
        self.nfeature_out = self.nfeature_in

        # build arrays for running sequential predictions
        X_in = np.zeros((1, self.nstep, self.nfeature_in))
        X_out = np.zeros((1, self.nforcast, self.nfeature_out))
        X_history = np.zeros((self.nstep + niter * self.nforcast, self.nfeature_out))

        # set initial condition
        X_in[0, :, :] = q_scaled
        X_history[:self.nstep, :] = X_in[0, :, :]

        # loop over time
        fp = Verbose(niter)
        for iter in range(niter):

            # solve (predict)
            if self.aspect == '2A':
                X_out[0, :, :] = model.predict([X_in, S_scaled[iter:iter + self.nstep_s, :].reshape(1, self.nstep_s, self.ns)], verbose=0)
            elif self.aspect == '1A':
                X_out[0, :, :] = model.predict([X_in], verbose=0)

            # update input solution for the next step
            X_in[0, :(self.nstep - self.nforcast), :] = X_in[0, self.nforcast:, :]
            X_in[0, (self.nstep - self.nforcast):, :] = X_out[0, :, :]

            # add new solution to history
            X_history[self.nstep + iter * self.nforcast:self.nstep + (iter + 1) * self.nforcast, :] = X_out[0, :, :]
            fp.fprint()

        # scale-back to the real values and then save it.
        X_history = scaler_features.inverse_transform(X_history)
        np.savetxt('{}/Qlstm{}.txt'.format(self.new_dir, self.fname), X_history, delimiter=',')
        np.savetxt('{}/S_input{}.txt'.format(self.new_dir, self.fname), S_input, delimiter=',')

        # add plot
        plt.clf() 
        plt.plot(X_history, '-b', label='predict', linewidth=0.5)
        plt.xlabel('Number of iterations')
        plt.ylabel('Values')
        plt.tight_layout()
        plt.savefig('{}/Q_run_{}.png'.format(self.new_dir, self.fname))
        plt.close()

        plt.clf()
        m = int(X_history.shape[0]/3)
        plt.plot(X_history[:, 0], X_history[:, 1], '-k', label='History', linewidth=1, alpha=0.7)
        plt.plot(X_history[:m, 0], X_history[:m, 1], '-b', label='Initial (S1)', linewidth=1)
        plt.plot(X_history[-m:, 0], X_history[-m:, 1], '-r', label='Target (S2)', linewidth=1)
        plt.plot(X_history[:self.nstep, 0], X_history[:self.nstep, 1], '-og', label='IC', linewidth=2.5, alpha=0.6)
        plt.xlabel('Values')
        plt.ylabel('Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/Q_run_traj_{}.png'.format(self.new_dir, self.fname))
        plt.close()
        plt.clf()

        plt.clf()
        plt.plot(S_input/S_input[0, :], linewidth=1.25)
        plt.xlabel('Iterations')
        plt.ylabel('S / S_1')
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/S_{}.png'.format(self.new_dir, self.fname))
        plt.close()
        plt.clf()

        return

#---------------------------------------------------------------------
# build a temporal variaitons for design parameters
#---------------------------------------------------------------------
def real_time_parameters(s1, s2, niter, initial, ramp, type):

    # s1: first set of parameters
    # s2: second set of parameters
    # niter: total number of iterations
    # initial: start iteraiton to apply changes
    # ramp: slope of changes (the higher the smoother changes)
    # type: type of transition from s1 to s2

    assert len(s1) == len(s2)
    ns = len(s1)

    # define history of parameters 
    s1_ = np.ones((niter, ns)) * s1
    s2_ = np.ones((niter, ns)) * s2

    # define history of iterations
    t = np.ones((niter, ns)) 
    t[:, 0] = np.linspace(0, niter-1, niter, endpoint=True)
    for i in range(t.shape[1]): t[:, i] = t[:, 0]

    # define dynamic variations
    if type == 'tanh':
        s = s1_ + 0.5 * (s2_ - s1_) * (1 + np.tanh((t - initial)/ramp))

    return s

#---------------------------------------------------------------------
# 
#---------------------------------------------------------------------
