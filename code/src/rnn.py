import tensorflow as tf
from src.utils import *


#---------------------------------------------------------------------
# Conventional RNN model (LSTM)
#---------------------------------------------------------------------
class RNN():

    def __init__(self, cfg):
        self.cfg = cfg
        self.ns = self.cfg["ns"]
        self.nstep = self.cfg["nstep"]
        self.nforcast = self.cfg["nforcast"]
        self.units = self.cfg["units"]
        self.num_layers = len(self.units)
        self.nfeature_in = self.cfg["feature_size"]
        self.nfeature_out = self.cfg["feature_size"]
        self.num_epochs = self.cfg["epochs"]
        self.batch_size = self.cfg["batch"]

    def build_lstm_network(self):

        # net u
        self.input_u = tf.keras.Input(shape=(self.nstep, self.nfeature_in))

        # LSTM
        previous_layer = self.input_u
        for i in range(self.num_layers):
            return_seq = (i < self.num_layers - 1)  # Return sequences for all but the last layer
            lstm_layer = tf.keras.layers.LSTM(units=self.units[i], activation='relu', return_sequences=return_seq)(previous_layer)
            previous_layer = lstm_layer

        # output layer 
        x_lstm = tf.keras.layers.Dense(self.nforcast * self.nfeature_out, activation='linear')(previous_layer)
        self.output_u = tf.keras.layers.Reshape((self.nforcast, self.nfeature_out))(x_lstm)

        # model
        self.model = tf.keras.Model(inputs=[self.input_u], outputs=[self.output_u], name='LSTM')

        return


    def train_model(self, setup, lr0=1e-2, verbose=0):

        # learning rate
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0, decay_steps = self.num_epochs/10, decay_rate = 0.8, staircase=True)

        # callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
                                                        monitor='val_loss',
                                                        min_delta=0,
                                                        patience=10,
                                                        verbose=0,
                                                        mode='min',
                                                        baseline=None,
                                                        restore_best_weights=False
                                                        )
        
        # model
        self.model.compile(loss=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss_weights=[1.0])
        self.model.summary()

        # train
        history = self.model.fit(
                                [setup.X_train_aug],
                                setup.Y_train_aug,
                                epochs=self.num_epochs,
                                batch_size=self.batch_size,
                                verbose=verbose,
                                shuffle=True,
                                validation_data=([setup.X_test_aug], setup.Y_test_aug),
                                callbacks=early_stopping
                                )

        self.model.save("{}/LSTM{}.tf".format(setup.new_dir, setup.fname))

        # plot
        plotLoss(history=history, fname=[setup.new_dir, "Model{}".format(setup.fname)])

        return

#---------------------------------------------------------------------
# New variantof RNN model (pLSTM, pGRU, pCNN, ghost-pLSTM, and customized pLSTM)
#---------------------------------------------------------------------
class pRNN():

    def __init__(self, cfg):
        self.cfg = cfg
        self.ns = self.cfg["ns"]
        self.nstep = self.cfg["nstep"]
        self.nforcast = self.cfg["nforcast"]
        self.units = self.cfg["units"]
        self.num_layers = len(self.units)
        self.nfeature_in = self.cfg["feature_size"]
        self.nfeature_out = self.cfg["feature_size"]
        self.num_epochs = self.cfg["epochs"]
        self.batch_size = self.cfg["batch"]
        self.num_epochs_ic = self.cfg["epochs_ic"]
        self.batch_size_ic = self.cfg["batch_ic"]

        if self.cfg["seq_s"] == True:
            self.nstep_s = self.nstep
        else:
            self.nstep_s = 1

    def build_pIC_network(self):

        # input layer
        self.input_s = tf.keras.Input(shape=(self.ns,))

        # net2 (dense)
        x = tf.keras.layers.Dense(self.nstep, activation='relu', name='HL1_IC')(self.input_s)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.nstep, activation='relu', name='HL2_IC')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(self.nstep * self.nfeature_in, activation='linear', name='OL_IC')(x)
        self.output_u = tf.keras.layers.Reshape((self.nstep, self.nfeature_in))(x)

        # model
        self.model = tf.keras.Model(inputs=[self.input_s], outputs=[self.output_u], name='IC')

        return self.model.summary()

    def build_design_gate(self):

        if self.nstep_s == self.nstep:
            # net A
            xs_A = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(self.nstep, activation='relu') )(self.input_s)
            #xs_A = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(self.nstep, activation='relu') )(xs_A)
            xs_A = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(self.nfeature_in, activation='relu') )(xs_A)
            #xs_A = tf.keras.layers.Dropout(0.05)(xs_A)
            #xs_A = tf.keras.layers.BatchNormalization()(xs_A) 

            # net B
            xs_B = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(self.nstep, activation='relu') )(self.input_s)
            #xs_B = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(self.nstep, activation='relu') )(xs_B)
            xs_B = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(self.nfeature_in, activation='relu') )(xs_B)
            #xs_B = tf.keras.layers.Dropout(0.05)(xs_B)
            #xs_B = tf.keras.layers.BatchNormalization()(xs_B)

        elif self.nstep_s == 1:

            # net A
            xs_A = tf.keras.layers.Flatten()(self.input_s)
            xs_A = tf.keras.layers.Dense(self.nstep, activation='relu')(xs_A)
            xs_A = tf.keras.layers.Dense(self.nstep, activation='relu')(xs_A)
            #xs_A = tf.keras.layers.Dropout(0.05)(xs_A)
            #xs_A = tf.keras.layers.BatchNormalization()(xs_A) 

            # net B
            xs_B = tf.keras.layers.Flatten()(self.input_s)
            xs_B = tf.keras.layers.Dense(self.nstep, activation='relu')(xs_B)
            xs_B = tf.keras.layers.Dense(self.nstep, activation='relu')(xs_B)
            #xs_B = tf.keras.layers.Dropout(0.05)(xs_B)
            #xs_B = tf.keras.layers.BatchNormalization()(xs_B)

            # repeat vectors (A and B)
            xs_A = tf.keras.layers.RepeatVector(self.nfeature_in)(xs_A)
            xs_B = tf.keras.layers.RepeatVector(self.nfeature_in)(xs_B)

            # reshape matrices (A and B)
            xs_A = tf.keras.layers.Reshape((self.nstep, self.nfeature_in))(xs_A)
            xs_B = tf.keras.layers.Reshape((self.nstep, self.nfeature_in))(xs_B)

        # A*u + B
        xus_mult = tf.keras.layers.Multiply()([self.input_u, xs_A])
        #xus_mult = tf.keras.layers.BatchNormalization()(xus_mult)
        self.xus = tf.keras.layers.Add()([xus_mult, xs_B])
        #xus= tf.keras.layers.BatchNormalization()(xus)
        #xus = tf.keras.layers.Activation('tanh')(xus)

        return

    def build_plstm_network(self):

        # net s (design parameters)
        self.input_s = tf.keras.Input(shape=(self.nstep_s, self.ns))

        # net u
        self.input_u = tf.keras.Input(shape=(self.nstep, self.nfeature_in))
        
        # design gate
        self.build_design_gate()
        
        # LSTM
        previous_layer = self.xus
        for i in range(self.num_layers):
            return_seq = (i < self.num_layers - 1)  # Return sequences for all but the last layer
            lstm_layer = tf.keras.layers.LSTM(units=self.units[i], activation='tanh', return_sequences=return_seq)(previous_layer)
            previous_layer = lstm_layer

        # output layer
        x_lstm = tf.keras.layers.Dense(self.nforcast * self.nfeature_out, activation='linear')(lstm_layer)
        self.output_u  = tf.keras.layers.Reshape((self.nforcast, self.nfeature_out))(x_lstm)

        # model
        self.model = tf.keras.Model(inputs=[self.input_u, self.input_s], outputs=[self.output_u], name='pLSTM')

        return self.model.summary()
    


    def train_IC(self, setup, lr0=1e-2, verbose=0):

        # learning rate
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0,decay_steps = self.num_epochs/5, decay_rate = 0.8, staircase=True)
        lr = 1e-3

        # train
        self.model.compile(loss=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss_weights=[1.0])
        history = self.model.fit(
                                [setup.Sic_aug],
                                [setup.Xic_aug],
                                epochs=self.num_epochs_ic,
                                batch_size=self.batch_size_ic,
                                verbose=verbose,
                                shuffle=True,
                                validation_split=0.1,
                                callbacks=None
                                )

        # save model
        self.model.save("{}/IC{}.tf".format(setup.new_dir, setup.fname))

        # plot
        plotLoss(history=history, fname=[setup.new_dir, "IC{}".format(setup.fname)])
        
        return

    def train_model(self, setup, lr0=1e-2, verbose=0):

        # learning rate
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0, decay_steps = self.num_epochs/2, decay_rate = 0.8, staircase=True)

        # callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
                                                        monitor='val_loss',
                                                        min_delta=0,
                                                        patience=10,
                                                        verbose=0,
                                                        mode='min',
                                                        baseline=None,
                                                        restore_best_weights=False
                                                        )

        # train
        self.model.compile(loss=['mse'], optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss_weights=[1.0])

        # loop over training steps
        for step in range(2):

            # check callback function
            if step == 0:
                callback = None
            else:
                callback = early_stopping

            # train 
            history = self.model.fit(
                                    [setup.X_train_aug, setup.S_train_aug],
                                    setup.Y_train_aug,
                                    epochs=int(self.num_epochs/10),
                                    batch_size=self.batch_size,
                                    verbose=verbose,
                                    shuffle=True,
                                    validation_data=([setup.X_test_aug, setup.S_test_aug], setup.Y_test_aug),
                                    callbacks=callback
                                    )
        
            # plot those pretty results :-)
            plotLoss(history=history, fname=[setup.new_dir, "Model{}".format(setup.fname)])

        # save model
        self.model.save("{}/LSTM{}.tf".format(setup.new_dir, setup.fname))

        return

#---------------------------------------------------------------------
#
#---------------------------------------------------------------------
