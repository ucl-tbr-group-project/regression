import argparse
from sklearn.preprocessing import StandardScaler

from ..data_utils import load_batches, encode_data_frame, x_y_split
from ..autoencoders import train_autoencoder
import ATE


def main():
    '''Main command line entry point. Trains model with given parameters.'''

    random_state = 1

    # parse args
    parser = argparse.ArgumentParser(description='Train TBR autoencoder')
    parser.add_argument('in_dir', type=str,
                        help='directory containing input batches')
    parser.add_argument('batch_low', type=int,
                        help='start batch index (inclusive)')
    parser.add_argument('batch_high', type=int,
                        help='end batch index (exclusive)')
    parser.add_argument('--encoding-dim', type=int, default=5,
                        help='dimension of the bottleneck')
    parser.add_argument('--deep-dims', nargs='*', type=int,
                        help='dimensions of the deep hidden layers')
    parser.add_argument('--regularize', default=False, action='store_true',
                        help='impose regularization penalty to minimize used neuron count')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='training batch size')
    args = parser.parse_args()

    # load data
    df = load_batches(args.in_dir, (args.batch_low, args.batch_high))
    df_enc = encode_data_frame(df, ATE.Domain())
    X, y_multiple = x_y_split(df_enc)
    X = X.sort_index(axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    print(
        f'True dim. is {X_train.shape[1]}, encoding dim. is {args.encoding_dim}.')
    print(f'Data compression factor is {X_train.shape[1]/args.encoding_dim}')
    autoencoder, encoder, decoder = train_autoencoder(
        X_train, args.encoding_dim,
        deep_dims=args.deep_dims if args.deep_dims is not None else [],
        regularize=args.regularize,
        epochs=args.epochs,
        batch_size=args.batch_size)
    print('Done.')


if __name__ == '__main__':
    main()
