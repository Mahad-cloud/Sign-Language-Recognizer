import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)

        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))

        self.i += 1

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend(['train', 'val'])
        plt.show()
        plt.grid(True)

        plt.plot(self.x, self.accuracy, label='acc')
        plt.plot(self.x, self.val_accuracy, label='val_acc')
        plt.legend(['train', 'val'])
        plt.show()
        plt.grid(True)
