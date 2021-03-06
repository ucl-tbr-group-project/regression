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
from ..metric_loader import get_metric_factory
from tbr_reg.endpoints.training import train, test, plot, plot_results, get_metrics


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
        print(c.columns)
        # evaluate initial parameters in given theory
        print("Evaluating initial " + str(init_samples) + " samples in " + args.theory + " theory.")
        output = thistheory(params = c, domain = domain, n_samples = init_samples)
        X_init, d, y_multiple = c_d_y_split(output)
        y_init = y_multiple['tbr']
        current_samples, current_tbr = X_init, y_init
    
    
    # MAIN QASS LOOP
    
    complete_condition = False
    iter_count = 0
    
    err_target = 0.0001
    max_iter_count = 70
    
    all_metrics = pd.DataFrame()
    
    current_samples = current_samples.sort_index(axis=1)
    
    print(f'Features in order are: {list(current_samples.columns)}')
    
    X_train, X_test, y_train, y_test = train_test_split(current_samples, current_tbr, 
                                           test_size=0.5, random_state=1)
                                           
    thismodel.enable_renormalization(100)
     
        
    while not complete_condition:
        iter_count += 1
        samp_size = X_train.shape[0] * 2
        print("Iteration " + str(iter_count) + " -- Total samples: " + str(samp_size))
        
        # Train surrogate for theory, and plot results
                
        if iter_count == 1:                           
            new_samples, new_tbr = X_train, y_train
        train(thismodel, new_samples, new_tbr)
        test(thismodel, X_test, y_test)
        
        plot("qassplot", thismodel, X_test, y_test)
        this_metrics = get_metrics(thismodel, X_test, y_test)
        this_metrics['numdata'] = samp_size
        print(this_metrics)
        
        
        # Calculate error data for this training iteration
        
        y_train_pred = thismodel.predict(X_train.to_numpy())
        y_test_pred = thismodel.predict(X_test.to_numpy())
        
        train_err = abs(y_train - y_train_pred)
        test_err = abs(y_test - y_test_pred)
       
        
        
        # Train neural network surrogate for error function (Failed)
        
        X_test = X_test.sort_index(axis=1)
        
        X_test1, X_test2, test_err1, test_err2 = train_test_split(X_test, test_err, 
                                               test_size=0.5, random_state=1)
            
            #errmodel = get_model_factory()["nn"](cli_args=["--epochs=50", "--batch-size=200"
                                                             # ,"--arch-type=4F_512"])
            #errmodel = get_model_factory()["rbf"](cli_args=["--d0=20"])
                                               
            #scaled_X_test1, scaled_test_err1 = errmodel.scale_training_set(X_test1, test_err1)
            #scaled_X_test2, scaled_test_err2 = errmodel.scale_testing_set(X_test2, test_err2)
            #dtest1 = pd.DataFrame(scaled_X_test1, columns = X_test1.columns,
                                                #  index = X_test1.index)
            #dtest2 = pd.DataFrame(scaled_X_test2, columns = X_test2.columns,
                                                #  index = X_test2.index)
            #derr1 = pd.Series(scaled_test_err1, index = test_err1.index)
            #derr2 = pd.Series(scaled_test_err2, index = test_err2.index)
            
            #print(type(test_err1))
            #print(type(scaled_test_err1))
            #train(errmodel, dtest1, derr1)
            #test(errmodel, dtest2, derr2)
            #print(X_test1)
            #print(scaled_X_test1)
            #print(dtest1)
            
            #plot("qassplot3", errmodel, dtest2, derr2) 
            
            
                                               
                #tri = Delaunay(X_test1.values, qhull_options="Qc QbB Qx Qz")                 
                #f = interpolate.LinearNDInterpolator(tri, test_err1.values)
                
                 
        # Test surrogate (nearest neighbor interpolator) on split error data        
                                 
        errordist_test = interpolate.NearestNDInterpolator(X_test1.values, test_err1.values)
        pred_err1 = errordist_test(X_test1.values)    
        pred_err2 = errordist_test(X_test2.values)
        
        # Train surrogate (nearest neighbor interpolator) for error function
        
        errordist = interpolate.NearestNDInterpolator(X_test.values, test_err.values)
        pred_err = errordist(X_test.values)
        
        max_err = max(test_err.values)
        print('Max error: ' + str(max_err))
        this_metrics['maxerr'] = max_err
        
        plot_results("qassplot2", pred_err1, test_err1)
        plt.figure()
        plot_results("qassplot3", pred_err2, test_err2) 
        
        plt.figure()
        plt.hist(test_err.values, bins=100)
        plt.savefig("qassplot4.pdf", format="pdf")   
        
        
        
        # Perform MCMC on error function
        
        saveinterval = 1
        nburn = 1000
        nrun = 10000
        
        initial_sample = X_train.iloc[0]
        #print(initial_sample.values)
        #print(errordist(initial_sample.values))
        burnin_sample, burnin_dist, burnin_acceptance = burnin_MH(errordist, initial_sample.values, nburn)
        saved_samples, saved_dists, run_acceptance = run_MH(errordist, burnin_sample, nrun, saveinterval)
        
        plt.figure()
        plt.hist(saved_dists, bins=100)
        plt.savefig("qassplot5.pdf", format="pdf") 
        
        print('MCMC run finished.')
        print('Burn-In Acceptance: ' + str(burnin_acceptance))
        print('Run Acceptance: ' + str(run_acceptance))
        this_metrics['burn_acc'] = burnin_acceptance
        this_metrics['run_acc'] = run_acceptance
        
                
        # Extract candidate samples from MCMC output and calculate mutual crowding distance
        
        cand_cdms = []
        print(saved_samples.shape)
        samplestep = int(saved_samples.shape[0] / step_candidates)
        print(samplestep)
        candidates = saved_samples[::samplestep]

        for candidate in candidates:
            cand_cdms.append( cdm(candidate,candidates) )

        # Rank candidate samples by error value, and filter out crowded samples
        
        new_samples = pd.DataFrame(candidates, columns = current_samples.columns)
        new_samples['error'] = saved_dists[::samplestep]
        new_samples['cdm'] = cand_cdms 
        
        #print(new_samples)
        #print(new_samples.shape)
            
        new_samples = new_samples.sort_values(by='error', ascending=False)

        indexNames = new_samples[ new_samples['cdm'] <= new_samples['cdm'].median() ].index
        new_samples.drop(indexNames , inplace=True)
        
        new_samples.drop(columns=['error', 'cdm'])
        new_samples = new_samples.head(step_samples).reset_index()
        
        
        # Add new samples and corresponding TBR evaluations to current sample set
        
        new_output = thistheory(params = new_samples.join(pd.concat([d.head(1)]*step_samples, ignore_index=True)), domain = domain, n_samples = step_samples)
        new_samples, new_d, new_y_multiple = c_d_y_split(new_output)
        new_tbr = new_y_multiple['tbr']
        
        #print(new_samples) 
        
        new_samples = new_samples.sort_index(axis=1)
        
        #new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_samples, new_tbr,test_size=0.5, random_state=1)

        X_train = pd.concat([X_train, new_samples], ignore_index=True)
        #X_test = pd.concat([X_test, new_X_test], ignore_index=True)
        y_train = pd.concat([y_train, new_tbr], ignore_index=True)
        #y_test = pd.concat([y_test, new_y_test], ignore_index=True)
    
        # Check completion conditions and close loop
    
        if max_err < err_target or iter_count > max_iter_count:
            complete_condition = True
        
        all_metrics = pd.concat([all_metrics,this_metrics], ignore_index=True)
        print(all_metrics)
        all_metrics.to_csv('qassmetrics.csv')


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
    print(c)
    print(c.shape)
    print(n_samples)
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

def step_MH(errordist, current_sample, dist_current, verbose=False):

    maxs = np.array([1, 1, 1, 1, 1, 1, 1, 500, 1, 1, 1, 20])    
    
    nvar = current_sample.size
    cov = np.diag(maxs) * 0.01
    
    if verbose:
        print("Current sample:")
        print(current_sample)
    candidate_sample = np.random.multivariate_normal(current_sample, cov)
    
    if verbose:
        print("Candidate sample:")
        print(candidate_sample)
    
    it = np.nditer(candidate_sample, op_flags = ['readwrite'], flags = ['f_index'])
    while not it.finished:
        ind = it.index
        if it[0] != max(min(it[0],maxs[ind]),0):
            #print("Candidate rejected")
            return current_sample, dist_current, 0
        it.iternext()

    dist_candidate = errordist(candidate_sample)

    acceptance_prob = min(1.0,dist_candidate/dist_current); # Metropolis-Hastings condition
    accept_rand = np.random.uniform(0,1,1)
    if(accept_rand < acceptance_prob and acceptance_prob>0):
        if verbose:
            print("Candidate accepted")
        return candidate_sample, dist_candidate, 1
    else:
        if verbose:
            print("Candidate rejected")
        return current_sample, dist_current, 0
        
def burnin_MH(errordist, initial_sample, nburn):

    current_sample = initial_sample
    dist_current = errordist(current_sample)
    
    n_accept = 0
    
    for i in range(nburn):
        current_sample, dist_current, if_accepted = step_MH(errordist, current_sample, dist_current, verbose=False);  # Run one iteration of Markov Chain
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
    
    return saved_samples.reshape((int(n_accept/saveinterval),nvar)), saved_dists, n_accept/nrun
    
def cdm(candidate, dataset):
    cdm = 0
    for sample in dataset:
        cdmadd = np.linalg.norm(candidate-sample)
        cdm += cdmadd
    return cdm                  



if __name__ == '__main__':
    main()
