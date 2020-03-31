import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import ATE
from ..data_utils import load_batches, encode_data_frame, x_y_split
from ..plot_utils import set_plotting_style
from ..plot_reg_performance import plot_reg_performance
from ..model_loader import get_model_factory, load_model_from_file


def main():
    '''
    Perform quality-adaptive sampling algorithm
    '''
    
    args = input_parse()
    
    init_samples = args.init_samples
    step_samples = args.step_samples
    step_candidates = args.step_candidates

    model = get_model_factory()[args.type](cli_args=sys.argv[7:])

    #set_plotting_style()
    
    domain = ATE.Domain()
    
    #for param in domain.params:
    #    if type(param) == ATE.param.DiscreteParameter:
            

    if True: #args.saved_init:
        # load data
        df = load_batches(args.saved_init, (0, 1 + int(init_samples/1000)))
        df_enc = encode_data_frame(df.iloc[1:init_samples], domain)
        X, y_multiple = x_y_split(df_enc)
        y = y_multiple['tbr']
    else:
        domain = Domain()


    print(df_enc)

    print('QASS finished.')


def input_parse():
    '''
    Parse input arguments from command line
    '''
    parser = argparse.ArgumentParser(description='Quality-Adaptive Surrogate Sampling')
    parser.add_argument('type', type=str,
                        help='which model to train')
    parser.add_argument('init_samples', type=int,
                        help='number of samples for initial surrogate building')
    parser.add_argument('step_samples', type=int,
                        help='number of samples kept per update')
    parser.add_argument('step_candidates', type=int,
                        help='number of candidate samples generated in MCMC per update')
    parser.add_argument('--saved-init', type=str, default=False,
                        help='Retrieve inital sample parameters from this file (max init_samples samples will be retrieved)')

    args = parser.parse_args(sys.argv[1:7])
    return args




if __name__ == '__main__':
    main()
