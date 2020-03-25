import os
import argparse

from ..data_utils import load_batches


def main():
    '''Main command line entry point. Splits TBR data into discrete slices.'''

    # parse args
    parser = argparse.ArgumentParser(
        description='Split/concatenate TBR data into multiple discrete slices')
    parser.add_argument('in_dir', type=str,
                        help='directory containing input batches')
    parser.add_argument('out_dir', type=str,
                        help='directory where outputs are saved')
    parser.add_argument('batch_low', type=int,
                        help='start batch index (inclusive)')
    parser.add_argument('batch_high', type=int,
                        help='end batch index (exclusive)')
    parser.add_argument('--concatenate', default=False, action='store_true',
                        help='impose regularization penalty to minimize used neuron count')
    args = parser.parse_args()

    if args.concatenate:
        outputs = [('batch0_out.csv', args.batch_low, args.batch_high)]
    else:
        outputs = [(f'batch{n}_out.csv', n, n + 1)
                   for n in range(args.batch_low, args.batch_high)]

    for out_file_name, low, high in outputs:
        print(f'Loading batches {low}-{high}')

        df = load_batches(args.in_dir, (low, high))
        discrete_columns = sorted(
            list(df.select_dtypes(include=['object']).columns))

        for group_name, group_data in df.groupby(discrete_columns):
            dir_path = os.path.join(args.out_dir, '_'.join(group_name))

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            out_path = os.path.join(dir_path, out_file_name)
            print(f'Saving group: {group_name} to file {out_path}')
            group_data.to_csv(out_path, index=False)

        print('')

    print('Done.')


if __name__ == '__main__':
    main()
