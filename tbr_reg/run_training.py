import sys
import argparse
from sklearn.model_selection import train_test_split

import ATE
from data_utils import load_batches, encode_data_frame, x_y_split
from model_loader import get_model_factory


def main():
    '''Main command line entry point. Trains model with given parameters.'''

    # parse args
    parser = argparse.ArgumentParser(description='Train TBR model')
    parser.add_argument('type', type=str,
                        help='which model to train')
    parser.add_argument('in_dir', type=str,
                        help='directory containing input batches')
    parser.add_argument('batch_low', type=int,
                        help='start batch index (inclusive)')
    parser.add_argument('batch_high', type=int,
                        help='end batch index (exclusive)')
    parser.add_argument('test_set_size', type=float,
                        help='fractional size of the test set')
    args = parser.parse_args(sys.argv[1:6])
    model = get_model_factory()[args.type](sys.argv[6:])

    # load data
    df = load_batches(args.in_dir, (args.batch_low, args.batch_high))
    df_enc = encode_data_frame(df, ATE.Domain())
    X, y = x_y_split(df_enc)
    X = X.sort_index(axis=1)

    if args.test_set_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y['tbr'], test_size=args.test_set_size, random_state=1)
    else:
        X_train, y_train = X, y['tbr']
        X_test, y_test = None, None

    print(f'Training regressor on set of size {X_train.shape[0]}')
    model.train(X_train, y_train)

    if X_test is not None:
        print(f'Testing regressor on set of size {X_test.shape[0]}')
        evaluation = model.evaluate(X_test, y_test)
        print(
            f'Evaluation on test set of size {X_test.shape[0]} gives result: {evaluation}')

    print('Done.')


if __name__ == '__main__':
    main()
