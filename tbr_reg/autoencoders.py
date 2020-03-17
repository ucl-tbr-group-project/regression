def create_autoencoder(input_dim, encoding_dim, deep_dims=[], regularize=False):
    '''Create autoencoder for given input and bottleneck size. Optionally add deep hidden layers or bottleneck regularization.'''

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers

    input_layer = Input(shape=(input_dim,))

    # build encoding layers
    encode_layers = []
    prev_layer = input_layer
    for dim in deep_dims:
        prev_layer = Dense(dim, activation='relu')(prev_layer)
        encode_layers.append(prev_layer)

    # build middle (smallest) layer
    if regularize:
        encoded = Dense(encoding_dim, activation='relu',
                        activity_regularizer=regularizers.l1(10e-5))(prev_layer)
    else:
        encoded = Dense(encoding_dim, activation='relu')(prev_layer)

    # build decoding layers
    decode_layers = []
    prev_layer = encoded
    for dim in reversed(deep_dims):
        prev_layer = Dense(dim, activation='relu')(prev_layer)
        decode_layers.append(prev_layer)

    decoded = Dense(input_dim, activation='tanh')(prev_layer)

    # autoencoder: encodes and then decodes
    autoencoder = Model(input_layer, decoded)

    # encoder: only encodes raw input
    encoder = Model(input_layer, encoded)

    # decoder: only decodes encoded input
    encoded_input = Input(shape=(encoding_dim,))
    prev_layer = encoded_input
    for layer in autoencoder.layers[(2+len(deep_dims)):]:
        prev_layer = layer(prev_layer)

    decoder = Model(encoded_input, prev_layer)

    # compile and return
    autoencoder.compile(optimizer='adadelta', loss='mse')
    autoencoder.summary()
    encoder.summary()
    decoder.summary()
    return autoencoder, encoder, decoder


def train_autoencoder(X_train, encoding_dim,
                      deep_dims=[],
                      regularize=False,
                      epochs=500,
                      batch_size=1024,
                      shuffle=True,
                      validation_split=0.25):
    '''Create and train autoencoder on the given data set.'''
    autoencoder, encoder, decoder = create_autoencoder(
        X_train.shape[1], encoding_dim,
        deep_dims=deep_dims,
        regularize=regularize)

    autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    validation_split=validation_split)

    return autoencoder, encoder, decoder
