import argparse
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, model_from_yaml

from models.basic_model import RegressionModel


class NeuralNetworkModel(RegressionModel):
    def __init__(self,
                 epochs=50,  # arbitrary
                 batch_size=1024,  # (1, n_samples)
                 scaling='standard',  # standard|minmax|none
                 validation_split=0.25,  # (0,1)
                 out=None,  # overrides all below
                 out_loss_plot_file=None,
                 out_model_file=None,
                 out_scaler_file=None,
                 out_checkpoint_file=None,
                 out_arch_file=None):
        RegressionModel.__init__(self, 'Neural network')

        if out is not None:
            out_loss_plot_file = '%s.loss' % out
            out_model_file = '%s.nn.h5' % out
            out_scaler_file = '%s.scaler.pkl' % out
            out_checkpoint_file = '%s.weights_{epoch:03d}_{val_loss:.5f}.h5' % out
            out_arch_file = '%s.arch.yml' % out

        self.epochs = epochs
        self.batch_size = batch_size
        self.scaling = scaling
        self.validation_split = validation_split
        self.out = out
        self.out_loss_plot_file = out_loss_plot_file
        self.out_model_file = out_model_file
        self.out_scaler_file = out_scaler_file
        self.out_checkpoint_file = out_checkpoint_file
        self.out_arch_file = out_arch_file

        self.net = None

        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling == 'none':
            self.scaler = None
        else:
            raise ValueError('Unknown scaler.')

    @staticmethod
    def load(model=None, arch=None, weights=None, scaler=None):
        loaded = NeuralNetworkModel()

        if scaler is not None:
            loaded.scaler = joblib.load(scaler)

        if model is not None:
            loaded.net = load_model(model)
        elif arch is not None and weights is not None:
            with open(arch, 'r') as f:
                yaml_string = f.read()
            loaded.net = model_from_yaml(yaml_string)
            loaded.net.load_weights(weights)
        else:
            raise ValueError('Model or (weights, arch) required')

        return loaded

    @staticmethod
    def parse_cli_args(args):
        parser = argparse.ArgumentParser(
            description='Train neural network with Keras')
        parser.add_argument('--epochs', type=int,
                            help='number of epochs')
        parser.add_argument('--batch-size', type=int,
                            help='batch size')
        parser.add_argument('--validation-split', type=float,
                            help='keras batch size')
        parser.add_argument('--scaling', type=str,
                            help='how to scale data before training')
        parser.add_argument('--out', type=str,
                            help='where to save all outputs')
        parser.add_argument('--out-loss-plot-file', type=str,
                            help='where to save loss plot')
        parser.add_argument('--out-model-file', type=str,
                            help='where to save trained model')
        parser.add_argument('--out-scaler-file', type=str,
                            help='where to save scaler')
        parser.add_argument('--out-checkpoint-file', type=str,
                            help='where to save checkpoints')
        parser.add_argument('--out-arch-file', type=str,
                            help='where to save net architecture')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def create_architecture(self, n_inputs):
        model = Sequential()
        model.add(Dense(128, input_dim=n_inputs, activation='relu'))

        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))

        model.add(Dense(1, activation='linear'))

        metrics = ['mae', 'mse']
        model.compile(optimizer='adam',
                      loss='mean_absolute_error',
                      metrics=metrics)
        model.summary()
        return model

    def train(self, X_train, y_train):
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            if self.out_scaler_file is not None:
                joblib.dump(self.scaler, self.out_scaler_file)

        self.net = self.create_architecture(X_train.shape[1])

        if self.out_arch_file is not None:
            with open(self.out_arch_file, 'w') as f:
                f.write(self.net.to_yaml())

        callbacks_list = []
        if self.out_checkpoint_file is not None:
            checkpoint = ModelCheckpoint(self.out_checkpoint_file,
                                         monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            callbacks_list.append(checkpoint)

        # train model
        history = self.net.fit(X_train, y_train, epochs=self.epochs,
                               batch_size=self.batch_size, validation_split=self.validation_split,
                               callbacks=callbacks_list)

        # save trained model
        if self.out_model_file is not None:
            self.net.save(self.out_model_file)

        # save loss history plot
        if self.out_loss_plot_file is not None:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.tight_layout()
            plt.savefig('%s.png' % self.out_loss_plot_file)
            plt.savefig('%s.pdf' % self.out_loss_plot_file)

    def evaluate(self, X_test, y_test):
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        return self.net.evaluate(X_test, y_test, batch_size=self.batch_size)

    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.net.predict(X, batch_size=self.batch_size)
