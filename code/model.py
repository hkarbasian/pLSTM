from src.utils import *
from src.prom import *

#---------------------------------------------------------------------
# Function to setup a pROM
#---------------------------------------------------------------------
def train_model(data_dir, output_dir, fname, ID_lst, cfg):

    # build pROM class
    model = pROM(dir=data_dir, output_dir=output_dir, fname=fname, cfg=cfg)
    model.loadData(ID_lst=ID_lst)
    model.buildDataset()
    model.splitData()
    model.augmentData(aug='org', stage=6, level=3, sigma_x=0.075, sigma_s=0.075, verbose=True, augtest=True)

    # build network architecture and train it - IC
    arc_ic = pRNN(cfg)
    arc_ic.build_pIC_network()
    arc_ic.train_IC(model, verbose=1)

    # build network architecture and train it - pLSTM
    arc = pRNN(cfg)
    arc.build_plstm_network()
    arc.train_model(model, verbose=1)

    # build network architecture and train it - LSTM
    #arc = RNN(cfg)
    #arc.build_lstm_network()
    #arc.train_model(model, verbose=1)

    # evaluate model using train and test datasets
    model.evaluateModel()
    model.testModel_seenData(Niter_tot=2, eps=0)
    if cfg["ncase"] != None: 
        model.testModel_UnseenData(Niter_tot=5, eps=0)

    return

#---------------------------------------------------------------------
# Example of training and executing a pROM
#---------------------------------------------------------------------

if __name__ == "__main__":

    # directories
    data_dir = '/root directory/data'
    output_dir = '/root directory/results'

    # configutaitons dictionary
    cfg = {
            "aspect": '2A', # 1A (conv) and 2A (parametric)

            "ns": 3,
            "nstep": 10,
            "nforcast": 1,
            "seq_s": False, # True -> dim(s)=[nstep, ns] and False -> dim(s)=[1, ns] 

            "ncase":18 ,
            "unseen_validation": False,

            "epochs_ic": 30, 
            "batch_ic": 10,

            "epochs": 20, 
            "batch": 1000,

            "test_ratio": 0.1,
            "splitOption": 'random',

            "units": [30],

            "feature_size": 0, # it will be determined automatically when data are loaded

        }

    # train
    train_model(data_dir=data_dir, output_dir=output_dir, fname="model", ID_lst=["R1"], cfg=cfg)

    # execute
    model = pROM(dir=data_dir, output_dir=output_dir, fname="model", cfg=cfg)
    model.run(niter=700, S1_input=[0.22, 0.55, 2.39], S2_input=[0.42, 0.58, 3.76])











