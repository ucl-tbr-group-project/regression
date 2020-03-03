import argparse
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


class NeuralNetwork:
    def __init__(self):
        pass

    def create_model(self, n_inputs):
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

    def parse_cli_args(self, args):
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

    def train(self, X_train, y_train,
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
        if out is not None:
            out_loss_plot_file = '%s.loss' % out
            out_model_file = '%s.nn.h5' % out
            out_scaler_file = '%s.scaler.pkl' % out
            out_checkpoint_file = '%s.weights_{epoch:03d}_{val_loss:.5f}.h5' % out
            out_arch_file = '%s.arch.yml' % out

        if scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
        elif scaling == 'none':
            scaler = None
        else:
            raise ValueError('Unknown scaler.')

        if scaler is not None:
            X_train = scaler.fit_transform(X_train)

            if out_scaler_file is not None:
                joblib.dump(scaler, out_scaler_file)

        model = self.create_model(X_train.shape[1])

        if out_arch_file is not None:
            with open(out_arch_file, 'w') as f:
                f.write(model.to_yaml())

        callbacks_list = []
        if out_checkpoint_file is not None:
            checkpoint = ModelCheckpoint(
                out_checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            callbacks_list.append(checkpoint)

        # train model
        history = model.fit(X_train, y_train, epochs=epochs,
                            batch_size=batch_size, validation_split=validation_split,
                            callbacks=callbacks_list)

        # save trained model
        if out_model_file is not None:
            model.save(out_model_file)

        # save loss history plot
        if out_loss_plot_file is not None:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.tight_layout()
            plt.savefig('%s.png' % out_loss_plot_file)
            plt.savefig('%s.pdf' % out_loss_plot_file)

        return scaler, model
