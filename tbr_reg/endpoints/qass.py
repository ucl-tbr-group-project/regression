import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from ATE import UniformSamplingStrategy, Domain, Samplerun
from ..data_utils import load_batches, encode_data_frame, x_y_split, c_d_y_split, c_d_split
from ..plot_utils import set_plotting_style
from ..plot_reg_performance import plot_reg_performance
from ..model_loader import get_model_factory, load_model_from_file
from tbr_reg.endpoints.training import train, test, plot


def main():
    '''
    Perform quality-adaptive sampling algorithm
    '''
    
    # Parse inputs and store in relevant variables.
    args = input_parse()
    
    init_samples = args.init_samples
    step_samples = args.step_samples
    step_candidates = args.step_candidates
    d_params = disctrans(args.disc_fix)
    
    # Collect surrogate model type and theory under study.
    thismodel = get_model_factory()[args.model](cli_args=sys.argv[7:])
    thistheory = globals()["theory_" + args.theory]
    
    
    domain = Domain()

    if args.saved_init:
        # load data as initial evaluated samples
        df = load_batches(args.saved_init, (0, 1 + int(init_samples/1000)))
        X_init, d, y_multiple = c_d_y_split(df.iloc[0:init_samples])
        d_params = d.values[0]
        print(d.values[0][0])
        y_init = y_multiple['tbr']
        
    domain.fix_param(domain.params[1], d_params[0])
    domain.fix_param(domain.params[2], d_params[1])
    domain.fix_param(domain.params[3], d_params[2])
    domain.fix_param(domain.params[5], d_params[3])
    domain.fix_param(domain.params[6], d_params[4])
    domain.fix_param(domain.params[7], d_params[5])
    domain.fix_param(domain.params[8], d_params[6])
    
    if not args.saved_init:
        # generate initial parameters
        sampling_strategy = UniformSamplingStrategy()
        c = domain.gen_data_frame(sampling_strategy, init_samples)
        print(c)
        # evaluate initial parameters in given theory
        print("Evaluating initial " + str(init_samples) + " samples in " + args.theory + " theory.")
        output = thistheory(params = c, domain = domain, n_samples = init_samples)
        X_init, d, y_multiple = c_d_y_split(output)
        y_init = y_multiple['tbr']
        print(y_init)
    
    
    X, y = X_init, y_init
    samp_size = X.shape[0]
    print(samp_size)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                       test_size=0.5, random_state=1)
                                       
    X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, 
                                       test_size=0.5, random_state=1)
                                       
    train(thismodel, X_train1, y_train1)
    train(thismodel, X_train2, y_train2)
    test(thismodel, X_test, y_test)
    plot("qassplot", thismodel, X_test, y_test)
    
    train(thismodel, X_train, y_train)
    test(thismodel, X_test, y_test)
    plot("qassplot2", thismodel, X_test, y_test)
        
    
    print(normalize_c(X_init))
    print('QASS finished.')


def input_parse():
    '''
    Parse input arguments from command line
    '''
    parser = argparse.ArgumentParser(description='Quality-Adaptive Surrogate Sampling')
    parser.add_argument('model', type=str,
                        help='which model to train')
    parser.add_argument('theory', type=str,
                        help='which theory to build a surrogate for')
    parser.add_argument('disc_fix', type=str,
                        help='fixed values of discrete parameters, as id code e.g.0120112')
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
    
    
def disctrans(discids):
    param_dict = [{'0':'tungsten'},
                  {'0':'SiC'      , '1':'eurofer'},
                  {'0':'H2O'      , '1':'He'      , '2':'D2O'},
                  {'0':'SiC'      , '1':'eurofer'},
                  {'0':'Li4SiO4'  , '1':'Li2TiO3'},
                  {'0':'Be'       , '1':'Be12Ti' },
                  {'0':'H2O'      , '1':'He'      , '2':'D2O'}]
    
    trans = []
    for index, id in enumerate(discids):
        param = str(param_dict[index][id])
        trans.append(param)
    return np.array(trans)
    

def theory_TBR(params, domain, n_samples=1):
    
    run = Samplerun()
    return run.perform_sample(domain=domain,
                              n_samples=n_samples,
                              param_values=params)


def theory_spherical(params, domain, n_samples=1):

    tbr = [];    
    params = params.iloc[0:n_samples]
    
    c, d = c_d_split(params)
    c = normalize_c(c)
    for i in range(n_samples):
        vec = c.iloc[i].values
        rad = np.linalg.norm(vec-0.5)  #radius of vector from center of parameter space
        tbr.append(st.norm(1,0.02).pdf(rad)/10)  #normal dist with max 2, mean 1, std 0.02
    
    return params.assign(tbr = tbr, tbr_error = tbr)


def theory_sinusoidal(params, domain, n_samples=1, waven=1):
    
    tbr = [];    
    params = params.iloc[0:n_samples]
    
    c, d = c_d_split(params)
    c = normalize_c(c)
    for i in range(n_samples):
        vec = c.iloc[i].values
        sinvec = (1 + np.sin(waven*2*np.pi*(vec-0.5)))/2  #waven sine waves in parameter
        
        tbr.append(np.mean(sinvec)*2)  #normal dist with max 2, mean 1, std 0.02
    
    return params.assign(tbr = tbr, tbr_error = tbr)


def theory_npeaks(params, domain, n_samples=1, numpeaks=5):

    tbr = [];    
    params = params.iloc[0:n_samples]
    
    c, d = c_d_split(params)
    c = normalize_c(c)
    for i in range(n_samples):
        vec = c.iloc[i].values
        peaks = np.random.rand(numpeaks,len(vec))
        
        thistbr = 0
        for peak in peaks:
            rad = np.max(vec-peak)  #radius of vector from peak center
            thistbr = max(thistbr,st.norm(0,0.2).pdf(rad))
        tbr.append(thistbr*1.5)  #normal dist with max 2, mean 1, std 0.02
           
    
    return params.assign(tbr = tbr, tbr_error = tbr)
    
def theory_nsteps(params, domain, n_samples=1, numpeaks=5):

    tbr = [];    
    params = params.iloc[0:n_samples]
    
    c, d = c_d_split(params)
    c = normalize_c(c)
    for i in range(n_samples):
        vec = c.iloc[i].values
        peaks = np.random.rand(numpeaks,len(vec))
        
        thistbr = 0
        for peak in peaks:
            rad = np.max(vec-peak)  #radius of vector from peak center
            thistbr = max(thistbr,st.norm(0,0.2).pdf(rad))
        tbr.append(int(thistbr*1.5))
           
    return params.assign(tbr = tbr, tbr_error = tbr)
    
def theory_flat(params, domain, n_samples=1):

    tbr = [];    
    params = params.iloc[0:n_samples]
    
    c, d = c_d_split(params)
    c = normalize_c(c)
    for i in range(n_samples):
        tbr.append(1) 
           
    return params.assign(tbr = tbr, tbr_error = tbr)
    
def normalize_c(c):
    c['blanket_thickness'] /= 500
    c['firstwall_thickness'] /= 20
    return c




if __name__ == '__main__':
    main()
