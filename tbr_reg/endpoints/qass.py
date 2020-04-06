import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from scipy import interpolate
from scipy.spatial import Delaunay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from ATE import UniformSamplingStrategy, Domain, Samplerun
from ..data_utils import load_batches, encode_data_frame, x_y_split, c_d_y_split, c_d_split
from ..plot_utils import set_plotting_style
from ..plot_reg_performance import plot_reg_performance
from ..model_loader import get_model_factory, load_model_from_file
from tbr_reg.endpoints.training import train, test, plot, plot_results


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
    
    # Train surrogate for theory and plot
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                       test_size=0.5, random_state=1)
                                       
    X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, 
                                       test_size=0.1, random_state=1)
                                       
    train(thismodel, X_train, y_train)
    test(thismodel, X_test, y_test)
    
    plot("qassplot", thismodel, X_test, y_test)
    
    
    # Calculate error data for this training iteration
    
    y_train_pred = thismodel.predict(X_train)
    y_test_pred = thismodel.predict(X_test)
    
    train_err = abs(y_train - y_train_pred)
    test_err = abs(y_test - y_test_pred)
   
    
    
    # Train surrogate (nearest neighbor interpolator) for error function
    
    X_test1, X_test2, test_err1, test_err2 = train_test_split(X_test, test_err, 
                                       test_size=0.5, random_state=1)
    
        #errmodel = get_model_factory()["nn"](cli_args=["--epochs=100", "--batch-size=100"
        #                                              ,"--arch-type=1H_3F_256"])
        #errmodel = get_model_factory()["idw"](cli_args=["--p=20"])
                                       
        #train(errmodel, X_test1, test_err1)
        #test(errmodel, X_test2, test_err2)
                                       
        #tri = Delaunay(X_test1.values, qhull_options="Qc QbB Qx Qz")                 
        #f = interpolate.LinearNDInterpolator(tri, test_err1.values)                  
    errordist_test = interpolate.NearestNDInterpolator(X_test1.values, test_err1.values)
    pred_err1 = errordist_test(X_test1.values)    
    pred_err2 = errordist_test(X_test2.values)
    
    errordist = interpolate.NearestNDInterpolator(X_test.values, test_err.values)
    pred_err = errordist(X_test.values)
    
    print('Max error: ' + str(max(test_err.values)))
    
    #plot("qassplot2", errmodel, X_test2, test_err2)
    plot_results("qassplot2", pred_err1, test_err1)
    plot_results("qassplot3", pred_err2, test_err2) 
    
    plt.figure()
    plt.hist(test_err.values, bins=100)
    plt.savefig("qassplot4.pdf", format="pdf")   
    
    # Perform MCMC on error function
    
    saveinterval = 5
    nburn = 10000
    nrun = 100000
    
    initial_sample = X_train.iloc[0]
    print(initial_sample.values)
    print(errordist(initial_sample.values))
    burnin_sample, burnin_dist, burnin_acceptance = burnin_MH(errordist, initial_sample.values, nburn)
    saved_samples, saved_dists, run_acceptance = run_MH(errordist, burnin_sample, nrun, saveinterval)
    
    print(burnin_acceptance)
    print(run_acceptance)
    
    plt.figure()
    plt.hist(saved_dists, bins=100)
    plt.savefig("qassplot5.pdf", format="pdf")  

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

def step_MH(errordist, current_sample, dist_current):

    nvar = current_sample.size
    cov = np.identity(nvar) * 10
    
    #print("Current sample:")
    #print(current_sample)
    candidate_sample = np.random.multivariate_normal(current_sample, cov)
    
    #print("Candidate sample:")
    #print(candidate_sample)

    dist_candidate = errordist(candidate_sample)

    acceptance_prob = min(1.0,dist_candidate/dist_current); # Metropolos-Hastings acceptance condition
    accept_rand = np.random.uniform(0,1,1)
    #print(dist_candidate)
    #print(dist_current)
    #print(acceptance_prob)
    #print(accept_rand)
    if(accept_rand < acceptance_prob and acceptance_prob>0):
        #print("Candidate accepted")
        return candidate_sample, dist_candidate, 1
    else:
        #print("Candidate rejected")
        return current_sample, dist_current, 0
        
def burnin_MH(errordist, initial_sample, nburn):

    current_sample = initial_sample
    dist_current = errordist(current_sample)
    
    n_accept = 0
    
    for i in range(nburn):
        current_sample, dist_current, if_accepted = step_MH(errordist, current_sample, dist_current);  # Run one iteration of Markov Chain
        n_accept += if_accepted
        
    return current_sample, dist_current, n_accept/nburn

def run_MH(errordist, initial_sample, nrun, saveinterval):

    nvar = initial_sample.size
    current_sample = initial_sample
    dist_current = errordist(current_sample)
    
    n_accept = 0
    
    saved_samples = np.empty((0,nvar), float)
    saved_dists = np.empty((0,1), float)
    
    for i in range(nrun):
        current_sample, dist_current, if_accepted = step_MH(errordist, current_sample, dist_current);  # Run one iteration of Markov Chain
        n_accept += if_accepted
        
        if if_accepted and n_accept/saveinterval == int(n_accept/saveinterval):
            saved_samples = np.append(saved_samples, current_sample)
            saved_dists = np.append(saved_dists, dist_current)
    
    return saved_samples, saved_dists, n_accept/nrun

                        




if __name__ == '__main__':
    main()
