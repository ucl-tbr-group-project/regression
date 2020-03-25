
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import ATE
from ..data_utils import load_batches, encode_data_frame, x_y_split
from ..plot_utils import set_plotting_style
from ..gans import create_adversarial_model, create_discriminator_model, create_discriminator, create_generator, train_gan

scaler = None
columns = None


def save_callback(noise_input, step, DM, AM, G):
    global scaler
    global columns

    print('Samples after step %d' % step)
    X_fake = G.predict(noise_input)
    X_fake = scaler.inverse_transform(X_fake)

    df = pd.DataFrame(data=X_fake, columns=columns)
    print(df)

    DM.save('gan_%d_dm.pkl' % step)
    AM.save('gan_%d_am.pkl' % step)


def main():
    global scaler
    global columns

    np.random.seed(1)
    set_plotting_style()

    params = [  # TODO: be able to change these
        'blanket_breeder_fraction',
        'blanket_breeder_li6_enrichment_fraction',
        'blanket_breeder_packing_fraction',
        'blanket_coolant_fraction',
        'blanket_multiplier_fraction',
        'blanket_multiplier_packing_fraction',
        'blanket_structural_fraction',
        'blanket_thickness',
        'firstwall_armour_fraction',
        'firstwall_coolant_fraction',
        'firstwall_structural_fraction',
        'firstwall_thickness',
        'tbr',
        'tbr_error',
    ]

    print('Loading data')
    df = load_batches('../../data/run1/', (0, 500))
    discrete_columns = sorted(
        list(df.select_dtypes(include=['object']).columns))
    groups = [(group_name, group_data)
              for group_name, group_data in df.groupby(discrete_columns)]

    g_name, g_data = groups[0]
    print('Selecting group: ' + str(g_name))
    df = g_data.copy()

    X = encode_data_frame(df, ATE.Domain())
    X = X[params].copy()
    X = X.sort_index(axis=1)
    columns = list(X.columns)
    print('X:')
    print(X.iloc[0])
    print(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print('Scaled X:')
    print(X[0, :].T)
    print(X)
    print(X.shape)

    latent_dims = 100
    D = create_discriminator((X.shape[1],))
    G = create_generator((latent_dims,), (X.shape[1],))

    DM = create_discriminator_model(D)
    AM = create_adversarial_model(D, G)

    print('Training')
    DM_losses, AM_losses = train_gan(
        X, DM, AM, G,
        latent_dims=latent_dims,
        train_steps=1000,
        batch_size=8192,
        save_interval=100,
        n_save_points=16,
        save_callback=save_callback
    )

    print('Plotting')
    plt.figure()

    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(DM_losses[:, 0], label='Discriminator loss')
    plt.plot(AM_losses[:, 0], label='Combined loss')
    plt.legend()

    # plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(DM_losses[:, 1], label='Discriminator accuracy')
    plt.plot(AM_losses[:, 1], label='Combined accuracy')
    plt.legend()

    # save plot to file
    plt.savefig('gan_loss.png')
    plt.close()


if __name__ == '__main__':
    main()
