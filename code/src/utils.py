import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def wash_out(data_dir):
	wanted_files = ["{}/para.txt".format(data_dir), "{}/para copy.txt".format(data_dir)]

	file_lst =pathlib.Path(data_dir).glob("*")
	for item in file_lst:
		if item.is_file():
			if str(item) not in wanted_files:
				print("file {} is removed".format(item))
				os.remove(item)

	return print("Oh yay! I washed all out :-)")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Verbose():
	def __init__(self, m, title=""):

		self.m = m
		self.counter = 1
		self.t0 = time.time()
		self.t1 = time.time()
		self.title = title

	def progress_bar(self, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (self.counter / float(self.m)))
		filledLength = int(length * self.counter // self.m)
		bar = fill * filledLength + ' ' * (length - filledLength)
		print(f'\r{prefix} |{bar}| {percent} {suffix}', end = printEnd)
		# Print New Line on Complete
		if self.counter == self.m:
			print('\n')

	def fprint(self, nv=1):
		if self.counter%nv == 0:
			command = f">> {self.title} iter:{self.counter}/{self.m}"
			suffix =  '% t:' + str(np.round(time.time() - self.t0, 3)) + ' | t_tot:' + str(np.round((time.time() - self.t1), 2))
			self.progress_bar(prefix=command, suffix=suffix, length = 20)
		self.counter += 1
		self.t0 = time.time()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Callback():
	def __init__(self, initial_epoch, patience):

		self.iter = 0
		self.best_val_loss = 1e6
		self.best_epoch = 0
		self.init_epochs = initial_epoch
		self.patience = patience
		self.wait = 0
		self.stop_training = False

	def check_early_stoping(self, vall_loss, epoch):

		if epoch > self.init_epochs:
			if vall_loss < self.best_val_loss:
				self.best_val_loss = vall_loss
				self.best_epoch = epoch
				# reset patience
				self.wait = 0
			else:
				self.wait += 1
				if self.wait > self.patience:
					self.stop_training = True
		else:
			pass

		return self.stop_training
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plotLoss(history, fname):
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    plt.clf()
    plt.plot(loss, '-b', label='Training')
    plt.plot(loss_val, '-r', label='Validation')
    plt.yscale('log')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('{}/lossFig_{}.pdf'.format(fname[0], fname[1]))
    plt.close()

def plt_loss(loss_train, loss_valid, fname):
    plt.clf()
    plt.plot(loss_train, '-k', label='Training')
    plt.plot(loss_valid, '-r', label='Validation')
    plt.yscale('log')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('{}/lossFig_{}.pdf'.format(fname[0], fname[1]))
    plt.close()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++