import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, LeakyReLU, BatchNormalization
from keras.optimizers import Adam, RMSprop


def create_discriminator(input_shape):
    model = Sequential()

    model.add(Dense(512, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


def create_generator(input_shape, output_shape):
    model = Sequential()

    model.add(Dense(256, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))
    model.summary()

    return model


def create_discriminator_model(D, optimizer=Adam(lr=0.0002, beta_1=0.5)):
    # optimizer = RMSprop(lr=0.0002, decay=6e-8)
    DM = Sequential()
    DM.add(D)
    DM.compile(loss='binary_crossentropy', optimizer=optimizer,
               metrics=['accuracy'])
    return DM


def create_adversarial_model(D, G, optimizer=Adam(lr=0.0002, beta_1=0.5)):
    # optimizer = RMSprop(lr=0.0001, decay=3e-8)
    D.trainable = False
    AM = Sequential()
    AM.add(G)
    AM.add(D)
    AM.compile(loss='binary_crossentropy', optimizer=optimizer,
               metrics=['accuracy'])
    return AM


def train_gan(X_train, DM, AM, G, latent_dims=100, train_steps=2000, batch_size=256, save_interval=0, n_save_points=16, save_callback=None, true_label=1, false_label=0):
    noise_input = None

    if save_interval > 0:
        noise_input = np.random.uniform(-1.0,  1.0,
                                        size=[n_save_points, latent_dims])

    DM_losses, AM_losses = [], []

    for i in range(train_steps):
        # train discriminator
        if i % 2 == 0:
            in_indices = np.random.randint(0, X_train.shape[0],
                                           size=batch_size)
            X = X_train[in_indices, :]
            y = true_label * np.ones([batch_size, 1])
        else:
            noise = np.random.uniform(-1.0, 1.0,
                                      size=[batch_size, latent_dims])
            X = G.predict(noise)
            y = false_label * np.ones([batch_size, 1])

        DM_loss = DM.train_on_batch(X, y)
        DM_losses.append(DM_loss)

        # train adversarial
        y = true_label * np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_dims])
        AM_loss = AM.train_on_batch(noise, y)
        AM_losses.append(AM_loss)

        log_mesg = '%d: [D loss: %f, acc: %f]' % (i, DM_loss[0], DM_loss[1])
        log_mesg = '%s  [A loss: %f, acc: %f]' % (
            log_mesg, AM_loss[0], AM_loss[1])
        print(log_mesg)

        if save_interval > 0 and save_callback is not None:
            if (i+1) % save_interval == 0:
                save_callback(noise_input, i+1, DM, AM, G)

    return np.array(DM_losses), np.array(AM_losses)
