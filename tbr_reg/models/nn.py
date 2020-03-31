import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import joblib
from keras import Sequential
from keras.layers import Dense, Conv1D, Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, model_from_yaml
from sklearn.metrics import mean_absolute_error

from .basic_model import RegressionModel


class NeuralNetworkSavedModelFactory:
    def __init__(self):
        self.extension = 'nn'

    def get_suffix(self):
        return '.%s.h5' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return NeuralNetworkModel(**NeuralNetworkModel.parse_cli_args(cli_args)) if arg_dict is None \
            else NeuralNetworkModel(**arg_dict)

    def load_model(self, fname):
        return NeuralNetworkModel.load(
            model='%s.%s.h5' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class NeuralNetworkCheckpointFactory:
    def __init__(self):
        self.extension = 'nncp'

    def get_suffix(self):
        return '.%s.h5' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return NeuralNetworkModel(**NeuralNetworkModel.parse_cli_args(cli_args)) if arg_dict is None \
            else NeuralNetworkModel(**arg_dict)

    def load_model(self, fname):
        fname_without_util = os.path.join(os.path.dirname(
            fname), os.path.basename(fname).split('.')[0])
        return NeuralNetworkModel.load(
            weights='%s.%s.h5' % (fname, self.extension),
            arch='%s.arch.yml' % fname_without_util,
            scaler='%s.scaler.pkl' % fname_without_util)


class NeuralNetworkModel(RegressionModel):
    '''A feed-forward multi-layer neural network implemented with Keras.'''

    def __init__(self,
                 epochs=50,  # arbitrary
                 batch_size=1024,  # (1, n_samples)
                 scaling='standard',  # standard|minmax|none
                 validation_split=0.25,  # (0,1)
                 arch_type='1H_3F_256',
                 out=None,  # overrides all below
                 out_loss_plot_file=None,
                 out_model_file=None,
                 out_scaler_file=None,
                 out_checkpoint_file=None,
                 out_arch_file=None):
        RegressionModel.__init__(self, 'Neural network', scaling=scaling)

        if out is not None:
            out_loss_plot_file = '%s.loss' % out
            out_model_file = '%s.nn.h5' % out
            out_scaler_file = '%s.scaler.pkl' % out
            out_checkpoint_file = '%s.{epoch:03d}_{val_loss:.5f}.nncp.h5' % out
            out_arch_file = '%s.arch.yml' % out

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.arch_type = arch_type
        self.out = out
        self.out_loss_plot_file = out_loss_plot_file
        self.out_model_file = out_model_file
        self.out_scaler_file = out_scaler_file
        self.out_checkpoint_file = out_checkpoint_file
        self.out_arch_file = out_arch_file

        self.net = None

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
        parser.add_argument('--arch-type', type=str,
                            help='network architecture')
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

    def create_architecture(self, n_inputs, arch_type):
        model = Sequential()

        # 1xDense(N/2) + 3xDense(N)
        if arch_type.startswith('1H_3F_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(int(n/2), input_dim=n_inputs, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(1, activation='linear'))
        # 1xDense(N)
        elif arch_type.startswith('1F_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(n, input_dim=n_inputs, activation='relu'))
            model.add(Dense(1, activation='linear'))
        # 2xDense(N)
        elif arch_type.startswith('2F_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(n, input_dim=n_inputs, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(1, activation='linear'))
        # 4xDense(N)
        elif arch_type.startswith('4F_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(n, input_dim=n_inputs, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(1, activation='linear'))
        # 8xDense(N)
        elif arch_type.startswith('8F_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(n, input_dim=n_inputs, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(1, activation='linear'))
        # Dense(N) + Dense(N * 2/3) + Dense(N * 1/3)
        elif arch_type.startswith('3pyramid_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(n, input_dim=n_inputs, activation='relu'))
            model.add(Dense(int(2*n/3), activation='relu'))
            model.add(Dense(int(n/3), activation='relu'))
            model.add(Dense(1, activation='linear'))
        # Dense(N) + Dense(N * 2/3) + Dense(N) + Dense(N * 2/3) + Dense(N) + Dense(N * 2/3)
        elif arch_type.startswith('6pump_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(n, input_dim=n_inputs, activation='relu'))
            model.add(Dense(int(2*n/3), activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(int(2*n/3), activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(int(2*n/3), activation='relu'))
            model.add(Dense(1, activation='linear'))
        # Dense(N * 1/3) + Dense(N * 2/3) + Dense(N) + Dense(N * 2/3) + Dense(N * 1/3)
        elif arch_type.startswith('3diamond_'):
            n = int(arch_type.split('_')[-1])
            model.add(Dense(int(n/3), input_dim=n_inputs, activation='relu'))
            model.add(Dense(int(2*n/3), activation='relu'))
            model.add(Dense(n, activation='relu'))
            model.add(Dense(int(2*n/3), activation='relu'))
            model.add(Dense(int(n/3), activation='relu'))
            model.add(Dense(1, activation='linear'))
        # 3xConv1D(64) of variable kernel size
        elif arch_type.startswith('3conv_'):
            kernel = int(arch_type.split('_')[-1])
            model.add(Reshape((n_inputs, 1), input_shape=(n_inputs,)))
            model.add(Conv1D(64, kernel, activation='relu'))
            model.add(Conv1D(64, kernel, activation='relu'))
            model.add(Conv1D(64, kernel, activation='relu'))
            model.add(Flatten())
            model.add(Dense(1, activation='linear'))

        metrics = ['mae', 'mse']
        model.compile(optimizer='adam',
                      loss='mean_absolute_error',
                      metrics=metrics)
        model.summary()
        return model

    def train(self, X_train, y_train):
        X_train, y_train = self.scale_training_set(
            X_train, y_train, out_scaler_file=self.out_scaler_file)
        self.net = self.create_architecture(X_train.shape[1], self.arch_type)

        if self.out_arch_file is not None:
            with open(self.out_arch_file, 'w') as f:
                f.write(self.net.to_yaml())

        callbacks_list = []
        if self.out_checkpoint_file is not None:
            checkpoint = ModelCheckpoint(self.out_checkpoint_file,
                                         monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            callbacks_list.append(checkpoint)

        # train model
        history = self.net.fit(X_train, y_train,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               validation_split=self.validation_split,
                               shuffle=True,
                               callbacks=callbacks_list)

        # save trained model
        if self.out_model_file is not None:
            self.net.save(self.out_model_file)

        # save loss history plot
        if self.out_loss_plot_file is not None:
            plt.plot(self.inverse_scale_errors(
                np.array(history.history['loss']).T))
            plt.plot(self.inverse_scale_errors(
                np.array(history.history['val_loss']).T))
            plt.ylabel('Loss (MAE)')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.tight_layout()
            plt.savefig('%s.png' % self.out_loss_plot_file)
            plt.savefig('%s.pdf' % self.out_loss_plot_file)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    def predict(self, X):
        X, _ = self.scale_testing_set(X, None)
        y = self.net.predict(X, batch_size=self.batch_size)
        return self.inverse_scale_predictions(y)
