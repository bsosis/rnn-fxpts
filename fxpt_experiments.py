"""
Large-scale experiments that evaluate rnn-fxpts on many randomly sampled networks
File names provided to methods in this module should follow these naming conventions:
  <test data id>: base name for a set of test networks
  traverse_<test data id>_N_<N>_s_<s>: results for traverse on the s^{th} network of size N
  baseline_<test data id>_N_<N>_s_<s>: results for traverse on the s^{th} network of size N
  TvB_<test data id>_N_<N>_s_<s>: results of traverse-baseline comparison on the s^{th} network of size N

  <test data id>_Wc_N_<N>_s_<s>: results of c-choice comparison on the s^{th} network of size N

  traverse_re_<test data id>_N_<N>_s_<s>: relative errors for round-off in traverse on the s^{th} network of size N
  traverse_rd_<test data id>_N_<N>_s_<s>: relative distances for round-off in traverse on the s^{th} network of size N
  baseline_re_<test data id>_N_<N>_s_<s>: relative errors for round-off in baseline on the s^{th} network of size N
  baseline_rd_<test data id>_N_<N>_s_<s>: relative distances for round-off in baseline on the s^{th} network of size N
"""
import os
import sys
import time
import multiprocessing as mp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
import plotter as ptr
import rnn_fxpts as rfx
import pickle as pkl

def generate_test_data(network_sizes, num_samples, test_data_id=None):
    """
    Randomly sample networks for testing with some fixed points by construction
    network_sizes[i] should be the i^{th} network size to include in the test data
    num_samples[i] should be the number of networks to generate with size network_sizes[i]
    test_data_id should be a file name with which to save the test data
      test data is saved as a numpy .npz archive
      if None, no file is saved
    returns test_data, a dictionary with keys
      "network_sizes": the list of network sizes (as a flat numpy.array)
      "num_samples": the list of sample counts at each network size (as a flat numpy.array)
      "N_%d_W_%d"%(N,s): the s^{th} weight matrix sampled at network size N (N by N numpy.array)
      "N_%d_V_%d"%(N,s): the corresponding known fixed points (N by N numpy.array), where
         test_data["N_%d_V_%d"%(N,s)][:,p] is the p^{th} known fixed point
    """
    test_data = {"network_sizes": np.array(network_sizes), "num_samples": np.array([num_samples])}
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            # Random V
            V = 2*np.random.rand(N,N) - 1
            # Construct W
            W = rfx.mrdivide(np.arctanh(V), V)
            # Refine V
            V, _ = rfx.refine_fxpts(W, V)
            # Store
            test_data["N_%d_W_%d"%(N,s)] = W
            test_data["N_%d_V_%d"%(N,s)] = V
    # Save test data to file
    if test_data_id is not None:
        np.savez(test_data_id, **test_data)
    return test_data

def generate_scarce_test_data(network_sizes, num_samples, numfx=8, test_data_id=None):
    """
    Randomly sample networks for testing with some fixed points by construction.
    Like generate_test_data, but the networks will tend to have fewer fixed points.
    network_sizes, num_samples, and test_data_id should be as in generate_test_data.
    numfx should be the number of known fixed points to include in the construction.
    The smaller numfx is, the fewer total fixed points the networks are expected to have.
    returns test_data, with same format as in generate_test_data.
    """
    # exposed an issue: W must have no eigenvalues = 1 (else DF low rank at origin)
    test_data = {"network_sizes": np.array(network_sizes), "num_samples": np.array([num_samples])}
    for N in network_sizes:
        for s in range(num_samples):
            # Random V
            V = 2*np.random.rand(N,numfx) - 1
            # Construct W
            A = np.linalg.svd(np.arctanh(V))[0][:,numfx:].dot(np.random.randn(N-numfx,N-numfx)/N)
            B = (np.random.randn(N-numfx,N-numfx)/N).dot(np.linalg.svd(V)[0][:,numfx:].T)
            X = np.concatenate((np.arctanh(V),A),axis=1)
            Y = np.concatenate((rfx.mrdivide(np.eye(numfx),V),B),axis=0)
            W = X.dot(Y)
            # Refine V
            V, _ = rfx.refine_fxpts(W, V)
            # Store
            test_data["N_%d_W_%d"%(N,s)] = W
            test_data["N_%d_V_%d"%(N,s)] = V
    # Save test data to file
    if test_data_id is not None:
        np.savez(test_data_id, **test_data)
    return test_data

def load_test_data(filename):
    """
    Load test data that was generated and saved by generate_test_data.
    filename should be the file name where the data was saved.
    returns test_data, with the same format as in generate_test_data.
    """
    f = open(filename,"r")
    test_data = np.load(f)
    test_data = {k: test_data[k] for k in test_data.files}
    f.close()
    network_sizes = test_data.pop("network_sizes")
    num_samples = test_data.pop("num_samples")[0]
    return network_sizes, num_samples, test_data

def save_pkl_file(filename, data):
    """
    Convenience function for pickling data to a file
    """
    pkl_file = open(filename,'w')
    pkl.dump(data, pkl_file)
    pkl_file.close()
def load_pkl_file(filename):
    """
    Convenience function for loading pickled data from a file
    """
    pkl_file = open(filename,'r')
    data = pkl.load(pkl_file)
    pkl_file.close()
    return data
def save_npz_file(filename, **kwargs):
    """
    Convenience function for saving numpy data to a file
    Each kwarg should have the form
      array_name=array
    """
    npz_file = open(filename,"w")
    np.savez(npz_file, **kwargs)
    npz_file.close()
def load_npz_file(filename):
    """
    Convenience function for loading numpy data from a file
    returns npz, a dictionary with key-value pairs of the form
      array_name: array
    """    
    npz = np.load(filename)
    npz = {k:npz[k] for k in npz.files}
    return npz

def test_traverse(W, V, c=None, result_key=None, logfilename=os.devnull, save_result=False, save_npz=False):
    """
    Test the traverse algorithm on a single test network.
    W should be the weight matrix (N by N numpy.array)
    V should be the known fixed points (N by K numpy.array)
    c should be the direction vector (N by 1 numpy.array)
      if None, a random c is automatically chosen
    result_key is a unique string identifier for the test
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on result_key
    if save_npz == True, traverse numpy outputs are saved in a file with name based on result_key
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output from traverse
    """
    N = W.shape[0]
    logfile = open(logfilename,'w')

    # run traversal
    logfile.write('Running traversal: %s...\n'%result_key)
    start = time.clock()
    status, fxV, VA, c, step_sizes, s_mins, residuals = rfx.traverse(W, c=c, max_traverse_steps = 2**20, logfile=logfile)
    runtime = time.clock()-start
    num_steps = VA.shape[1]

    results = {
        "result_key": result_key,
        "status": status,
        "runtime": runtime,
        "N": W.shape[0],
        "num_steps": num_steps,
        "path_length": step_sizes.sum(),
        "min_s_min": s_mins.min(),
        "num_fxV": fxV.shape[1]
    }
    npz = {"W":W, "V":V, "VA":VA, "fxV":fxV, "c":c, "residuals":residuals}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # Post-process
    # count unique fixed points found
    start = time.clock()
    fxV_unique, fxV_converged = rfx.post_process_fxpts(W, fxV, logfile=logfile)
    post_runtime = time.clock()-start
    results['post_runtime'] = post_runtime
    results['num_fxV_unique'] = fxV_unique.shape[1]
    npz["fxV_unique"] = fxV_unique
    npz["fxV_converged"] = fxV_converged
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # check for ground truth inclusion
    logfile.write('Checking ground truths...\n')
    V_found = np.zeros(N, dtype=bool)
    for j in range(V.shape[1]):
        identical, _, _ = rfx.identical_fixed_points(W, fxV_converged, V[:,[j]])
        V_found[j] = identical.any()
    results["num_V_found"] = V_found.sum(),
    npz["V_found"] = V_found
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    finish_str = "pid %d: %s, %d fxV (%d unique), %d of %d gt, %d iters (length ~ %f, step_size ~ %f).  restarting..."%(os.getpid(), result_key, fxV.shape[1], fxV_unique.shape[1], V_found.sum(), N, num_steps, step_sizes.sum(), step_sizes.mean())
    logfile.write('%s\n'%finish_str)
    print(finish_str)

    logfile.close()
    return results, npz

def pool_test_traverse(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_traverse(*args)
    return results

def run_traverse_experiments(test_data_id, num_procs):
    """
    Run test_traverse on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            W = test_data['N_%d_W_%d'%(N,s)]
            V = test_data['N_%d_V_%d'%(N,s)]
            c = None
            result_key = 'traverse_%s_N_%d_s_%d'%(test_data_id, N, s)
            if num_procs > 0:
                logfilename = 'logs/%s.log'%result_key
                save_result=True
                save_npz=True
            else:
                logfilename = 'logs/temp.txt'
                save_result=False
                save_npz=False
            pool_args.append((W,V,c,result_key,logfilename,save_result,save_npz))
    start_time = time.time()
    test_fun = pool_test_traverse
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/traverse_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_Wc(W, V, result_key=None, logfilename=os.devnull, save_result=False):
    """
    Test traverse with different c choices on a single test network.
    One choice is tested for each of the 2^N possible values of numpy.sign(W.dot(c)).
    W should be the weight matrix (N by N numpy.array)
    V should be the known fixed points (N by K numpy.array)
    result_key is a unique string identifier for the test
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on result_key
    returns results, a list where
      results[i] is a dictionary summarizing the test results for the i^{th} choice of c
    """
    N = W.shape[0]
    logfile = open(logfilename,'w')
    logfile.write('Running Wc: %s...\n'%result_key)

    signs = ptr.lattice(-np.ones((N,1)),np.ones((N,1)),2)
    C = rfx.solve(W, signs + 0.1*(np.random.rand(*signs.shape)-0.5))
    all_fxV = []
    results = []
    for s in range(signs.shape[1]):

        # run traversal
        logfile.write('Running traversal %d...\n'%s)
        start = time.clock()
        status, fxV, VA, c, step_sizes, s_mins, residuals = rfx.traverse(W, c=C[:,[s]], max_traverse_steps = 2**20, logfile=logfile)
        runtime = time.clock()-start
        num_steps = VA.shape[1]

        # count unique fixed points found
        fxV_unique, fxV_converged = rfx.post_process_fxpts(W, fxV, logfile=logfile)

        all_fxV.append(fxV_unique)
        result = {
            "result_key": result_key,
            "status": status,
            "runtime": runtime,
            "N": W.shape[0],
            "num_steps": num_steps,
            "path_length": step_sizes.sum(),
            "min_s_min": s_mins.min(),
            "num_fxV_unique": fxV_unique.shape[1],
        }

        # check for ground truth inclusion
        logfile.write('Checking ground truths...\n')
        V_found = np.zeros(N, dtype=bool)
        for j in range(V.shape[1]):
            identical, _, _ = rfx.identical_fixed_points(W, fxV_unique, V[:,[j]])
            V_found[j] = identical.any()
        result["num_V_found"] = V_found.sum(),

        results.append(result)

    # union of known fxV
    V = np.concatenate((-V, np.zeros((N,1)), V), axis=1)
    fxV_union = np.concatenate(all_fxV + [V], axis=1)
    logfile.write('post-processing union...\n')
    fxV_union, _ = rfx.post_process_fxpts(W, fxV_union, logfile=logfile)
    for s in range(signs.shape[1]):
        results[s]["num_fxV_union"] = fxV_union.shape[1]

    # return results
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    logfile.write('%s\n'%str([r["num_fxV_unique"] for r in results]))
    best = max([r["num_fxV_unique"] for r in results])
    finish_str = "pid %d: %s, best=%d fxV of %d union.  restarting..."%(os.getpid(), result_key, best, fxV_union.shape[1])
    logfile.write('%s\n'%finish_str)
    print(finish_str)

    return results

def pool_test_Wc(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    return test_Wc(*args)

def run_Wc_experiments(test_data_id, num_procs):
    """
    Run test_Wc on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            W = test_data['N_%d_W_%d'%(N,s)]
            V = test_data['N_%d_V_%d'%(N,s)]
            result_key = '%s_Wc_N_%d_s_%d'%(test_data_id, N, s)
            logfilename =  'logs/%s.log'%result_key
            save_result=True
            pool_args.append((W,V,result_key,logfilename,save_result))
    start_time = time.time()
    test_fun = pool_test_Wc
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/%s_Wc.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_baseline(W, V, timeout=60, result_key=None, logfilename=os.devnull, save_result=False, save_npz=False):
    """
    Test the baseline solver on a single test network.
    W should be the weight matrix (N by N numpy.array)
    V should be the known fixed points (N by K numpy.array)
    timeout is the number of seconds before the solver is terminated
    result_key is a unique string identifier for the test
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on result_key
    if save_npz == True, solver numpy outputs are saved in a file with name based on result_key
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output from the solver
    """
    N = W.shape[0]
    logfile = open(logfilename,'w')

    # run baseline
    start = time.clock()
    fxV, num_reps = rfx.baseline_solver(W, timeout=timeout, logfile=logfile)
    runtime = time.clock()-start
    results = {
        "result_key": result_key,
        "runtime": runtime,
        "N": W.shape[0],
        "num_reps": num_reps,
        "num_fxV": fxV.shape[1],
    }

    npz = {"W":W, "V":V, "fxV":fxV}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    logfile.write("Post-processing...\n")
    start = time.clock()
    fxV_unique, fxV_converged = rfx.post_process_fxpts(W, fxV, logfile=logfile)
    post_runtime = time.clock()-start
    results["post_runtime"] = post_runtime
    results["num_fxV_unique"] = fxV_unique.shape[1]
    npz["fxV_unique"] = fxV_unique
    npz["fxV_converged"] = fxV_converged
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # check for ground truth inclusion
    logfile.write('checking ground truths...\n')
    V_found = np.zeros(N, dtype=bool)
    for j in range(V.shape[1]):
        identical, _, _ = rfx.identical_fixed_points(W, fxV_converged, V[:,[j]])
        V_found[j] = identical.any()
    results["num_V_found"] = V_found.sum()
    npz["V_found"] = V_found
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    finish_str = "pid %d: %s, %d fxV (%d unique), %d of %d gt, %d reps.  restarting..."%(os.getpid(), result_key, fxV_converged.shape[1], fxV_unique.shape[1], V_found.sum(), N, num_reps)
    logfile.write("%s\n"%finish_str)
    print(finish_str)

    logfile.close()
    return results, npz

def pool_test_baseline(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_baseline(*args)
    return results

def run_baseline_experiments(test_data_id, num_procs):
    """
    Run test_baseline on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    for (N, S) in zip(network_sizes, num_samples):
        for s in range(S):
            W = test_data['N_%d_W_%d'%(N,s)]
            V = test_data['N_%d_V_%d'%(N,s)]
            traverse_result_key = 'traverse_%s_N_%d_s_%d'%(test_data_id, N, s)
            traverse_results_file = open('results/%s.pkl'%traverse_result_key, 'r')
            traverse_results = pkl.load(traverse_results_file)
            traverse_results_file.close()
            timeout = traverse_results['runtime']
            result_key = 'baseline_%s_N_%d_s_%d'%(test_data_id, N, s)
            logfilename = 'logs/baseline_%s_N_%d_s_%d.log'%(test_data_id, N, s)
            save_result=True
            save_npz=True
            pool_args.append((W,V,timeout,result_key,logfilename,save_result,save_npz))
    start_time = time.time()
    test_fun = pool_test_baseline
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/baseline_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def test_TvB(test_data_id, N, s, logfilename=os.devnull, save_result=False, save_npz=False):
    """
    Compare the traverse and baseline results on a single test network.
    test_data_id should be as in generate_test_data (without file extension)
    Inspects the s^{th} network of size N
    logfilename is a file name at which to write progress updates
    if save_result == True, results are saved in a file with name based on test_data_id
    if save_npz == True, numpy outputs are saved in a file with name based on test_data_id
    returns results, npz, where
      results is a dictionary summarizing the test results
      npz is a dictionary with full numpy output
    """
    logfile = open(logfilename,'w')

    logfile.write('Loading results...\n')
    _,_,test_data = load_test_data('%s.npz'%test_data_id)
    W = test_data['N_%d_W_%d'%(N,s)]
    baseline_npz = np.load('results/baseline_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    traverse_npz = np.load('results/traverse_%s_N_%d_s_%d.npz'%(test_data_id, N, s))
    fxV_baseline = baseline_npz["fxV_unique"]
    fxV_traverse = traverse_npz["fxV_unique"]
    T = fxV_traverse.shape[1]
    B = fxV_baseline.shape[1]

    result_key = 'TvB_%s_N_%d_s_%d'%(test_data_id, N, s)
    results = {
        'result_key': result_key,
        'N':N,
        'T':T,
        'B':B,
    }
    npz = {"W":W, "fxV_baseline":fxV_baseline,"fxV_traverse":fxV_traverse}
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # Get union
    logfile.write('unioning %d + %d...\n'%(T, B))
    fxV_union = np.concatenate((fxV_traverse, fxV_baseline), axis=1)
    neighbors = lambda X, y: rfx.identical_fixed_points(W, X, y)[0]
    fxV_union = rfx.get_unique_points_recursively(fxV_union, neighbors=neighbors)
    TB = fxV_union.shape[1]
    finish_str = 'N:%d,T:%d, B:%d, T|B:%d, T&B:%d, T-B:%d(%f), B-T:%d(%f)'%(N,T,B,TB,T+B-TB,TB-B,1.*(TB-B)/TB,TB-T,1.*(TB-T)/TB)
    logfile.write('%s\n'%finish_str)
    print(finish_str)

    results['T|B']=TB
    results['T&B']=T+B-TB
    results['T-B']=TB-B
    results['B-T']=TB-T
    npz['fxV_union'] = fxV_union
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)
    if save_npz: save_npz_file('results/%s.npz'%result_key, **npz)

    # distances around means
    baseline_mean = fxV_baseline.mean(axis=1)
    traverse_mean = fxV_traverse.mean(axis=1)
    baseline_dist = np.mean(np.sqrt(((fxV_baseline-baseline_mean[:,np.newaxis])**2).sum(axis=0)))
    traverse_dist = np.mean(np.sqrt(((fxV_traverse-traverse_mean[:,np.newaxis])**2).sum(axis=0)))
    results['baseline_dist'] = baseline_dist
    results['traverse_dist'] = traverse_dist
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)

    # vertex proximities:
    fxV_baseline_dist_v = np.sqrt(((fxV_baseline - np.sign(fxV_baseline))**2).sum(axis=0))
    fxV_traverse_dist_v = np.sqrt(((fxV_traverse - np.sign(fxV_traverse))**2).sum(axis=0))
    results['baseline_dist_v'] = fxV_baseline_dist_v.mean()
    results['traverse_dist_v'] = fxV_traverse_dist_v.mean()
    if save_result: save_pkl_file('results/%s.pkl'%result_key, results)

    logfile.close()
    return results, npz

def pool_test_TvB(args):
    """
    Wrapper function passed to multiprocessing.Pool
    """
    results, _ = test_TvB(*args)
    return results

def run_TvB_experiments(test_data_id, num_procs):
    """
    Run test_TvB on every network in the test data
    Uses multi-processing to test on multiple networks in parallel
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    returns pool_results, a list of results with one entry per network
    """

    cpu_count = mp.cpu_count()
    print('%d cpus, using %d'%(cpu_count, num_procs))

    pool_args = []
    network_sizes, num_samples, test_data = load_test_data('%s.npz'%test_data_id)
    for (N,S) in zip(network_sizes, num_samples):
        for s in range(S):
            logfilename = 'logs/tvb_%s_N_%d_s_%d.log'%(test_data_id, N, s)
            save_result=True
            save_npz=True
            pool_args.append((test_data_id, N, s, logfilename, save_result, save_npz))
    start_time = time.time()
    test_fun = pool_test_TvB
    if num_procs < 1: # don't multiprocess
        pool_results = [test_fun(args) for args in pool_args]
    else:
        pool = mp.Pool(processes=num_procs)
        pool_results = pool.map(test_fun, pool_args)
        pool.close()
        pool.join()
    print('total time: %f. saving results...'%(time.time()-start_time))

    results_file = open('results/tvb_%s.pkl'%test_data_id, 'w')
    pkl.dump(pool_results, results_file)
    results_file.close()

    return pool_results

def show_tvb_results(test_data_ids=['dl50','dm10','dh5']):
    """
    Plot the results of traverse-baseline performance comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    results = []
    for test_data_id in test_data_ids:
        results += load_pkl_file('results/tvb_%s.pkl'%test_data_id)
    results = [r for r in results if r['N'] in [2,4,7,10,13,16,24,32,48,64,128]]
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    Ns = np.array([r['N'] for r in results])
    uNs = np.unique(Ns)
    dats = [('T|B','v','k'),('T&B','^','none'),('T-B','s','none'),('B-T','o','k')]
    handles = []
    for ym in dats:
        y = np.array([np.log2(r[ym[0]]) if r[ym[0]]>0 else -1 for r in results])
        handles.append(scatter_with_errors(Ns, uNs, y, ym[1],ym[2]))
    # handles.append(scatter_with_errors(Ns, uNs, np.log2(Ns), marker='d'))
    handles.append(plt.plot(uNs, np.log2(uNs), 'dk--', ms=9)[0])
    # plt.legend(handles, [ym[0] for ym in dats]+['Known'], loc='upper left')
    plt.legend(handles, ['$T\cup\,B$', '$T\cap\,B$', '$T-B$', '$B-T$', 'Known'], loc='lower right')
    plt.xlim([uNs[0]-1,uNs[-1]+1])
    plt.ylim([-1,15])
    plt.ylabel('# of fixed points')
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    plt.tight_layout()
    plt.show()

def show_tvb_dist_results(test_data_ids=['dl50','dm10','dh5']):
    """
    Plot the results of traverse-baseline spatial distribution comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    results = []
    for test_data_id in test_data_ids:
        # results += load_pkl_file('results/tvb_dist_%s.pkl'%test_data_id)
        results += load_pkl_file('results/tvb_%s.pkl'%test_data_id)
    results = [r for r in results if r['N'] in [2,4,7,10,13,16,24,32,48,64,128]]
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    Ns = np.array([r['N'] for r in results])
    uNs = np.unique(Ns)
    handles = []
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['mean_dist'] for r in results])))
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist'] for r in results]), 'o','k'))
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist'] for r in results]), 'o','none'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist_p'] for r in results]),'^'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist_p'] for r in results]),'v'))
    # handles.append(scatter_with_errors(Ns, uNs, np.array([r['mean_dist_p'] for r in results]), 'd')) # accidental tuple
    # plt.legend(handles, ['$||T - T_{mean}||$','$||B - B_{mean}||$','$||T^{+} - T^{+}_{mean}||$','$||B^{+} - B^{+}_{mean}||$','$||T^{+}_{mean} - B^{+}_{mean}||$'], loc='upper left')
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['traverse_dist_v'] for r in results]),'^','k'))
    handles.append(scatter_with_errors(Ns, uNs, np.array([r['baseline_dist_v'] for r in results]),'^','none'))
    plt.legend(handles, ['$||T - T_{mean}||$','$||B - B_{mean}||$','$||T - [T]||$','$||B - [B]||$'], loc='upper left')
    plt.xlim([uNs[0]-1,uNs[-1]+1])
    # plt.ylim([-1,15])
    plt.ylabel('Average distances')
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

def show_tvb_runtimes(test_data_ids):
    """
    Plot the results of traverse-baseline runtime comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})

    t_res, b_res = [], []
    for test_data_id in test_data_ids:
        t_res += load_pkl_file('results/traverse_%s.pkl'%test_data_id)
        b_res += load_pkl_file('results/baseline_%s.pkl'%test_data_id)
    t_res = [r for r in t_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128]]
    b_res = [r for r in b_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128]]

    Ns = np.array([r['N'] for r in t_res])
    uNs = np.unique(Ns)
    handles = []
    handles.append(scatter_with_errors(Ns, uNs, (np.array([r['runtime']+r['post_runtime'] for r in b_res]))/60, '^','none'))
    handles.append(scatter_with_errors(Ns, uNs, (np.array([r['runtime']+r['post_runtime'] for r in t_res]))/60, 'o','none'))
    handles.append(scatter_with_errors(Ns, uNs, (np.array([r['runtime'] for r in t_res]))/60, 'x','none'))
    plt.legend(handles, ['With B post-processing','With T post-processing','Runtime'], loc='upper left')
    plt.xlim([uNs[0]-1,uNs[-1]+1])
    plt.ylim([-10,200])
    plt.ylabel('Running time (minutes)')
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

def show_tvb_rawcounts(test_data_ids):
    """
    Plot the results of traverse-baseline un-post-processed count comparison on one or more testing data sets
    test_data_ids should be the list of ids, each as in generate_test_data (without file extension)
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    # mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams.update({'font.size': 12})
    # mpl.rcParams['lines.linewidth'] = 2

    t_res, b_res = [], []
    for test_data_id in test_data_ids:
        t_res += load_pkl_file('results/traverse_%s.pkl'%test_data_id)
        b_res += load_pkl_file('results/baseline_%s.pkl'%test_data_id)
    t_res = [r for r in t_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128]]
    b_res = [r for r in b_res if r['N'] in [2,4,7,10,13,16,24,32,48,64,128]]

    Ns = np.array([r['N'] for r in t_res])
    uNs = np.unique(Ns)
    handles = []
    handles.append(scatter_with_errors(Ns, uNs, np.log2(np.array([r['num_fxV'] for r in b_res])), 'o', 'k'))
    handles.append(scatter_with_errors(Ns, uNs, np.log2(np.array([r['num_fxV'] for r in t_res])), 'o', 'none'))
    plt.legend(handles, ['Raw B counts','Raw T counts'], loc='lower right')
    plt.xlim([uNs[0]-1,uNs[-1]+1])
    # plt.ylim([-2,90])
    plt.ylabel('Raw point counts')
    plt.yticks(range(2,19,2),['$2^{%d}$'%yl for yl in range(2,19,2)])
    #plt.title('Traverse vs Baseline')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    # plt.yticks(range(-1,15,2),['0']+['$2^{%d}$'%yl for yl in range(1,15,2)])
    # plt.tight_layout()
    plt.show()

# def show_Wc_results(results):
def show_Wc_results(test_data_id='dl15'):
    """
    Plot the results of c choice comparison
    test_data_id should be as in generate_test_data (without file extension)
    """
    results = load_pkl_file('results/%s_Wc.pkl'%test_data_id)
    mpl.rcParams['mathtext.default'] = 'regular'
    Ns = np.array([r[0]['N'] for r in results])
    uNs = np.unique(Ns)
    handles = []
    y = np.array([r[0]['num_fxV_union'] for r in results])
    handles.append(scatter_with_errors(Ns, uNs, np.log2(y), 'o','k'))
    for (fun, m,fc) in [(np.max,'^','none'),(np.mean,'d','k',),(np.min,'v','none')]:
        y = np.array([fun([r['num_fxV_unique'] for r in res]) for res in results])
        handles.append(scatter_with_errors(Ns, uNs, np.log2(y),m,fc))
    handles.append(plt.plot(uNs, np.log2(uNs), 'dk--')[0])
    plt.legend(handles, ['Union','Max','Mean','Min','Known'], loc='upper left')
    plt.xlim([uNs[0]-1,uNs[-1]+1])
    plt.ylim([0,10])
    plt.ylabel('# of fixed points')
    #plt.title('Different Regular Regions')
    # plt.draw()
    # ytick_labels = plt.gca().get_yticklabels()
    # plt.gca().set_yticklabels(['2^%s'%(yl.get_text()) for yl in ytick_labels])
    plt.yticks(range(0,11,2),['$2^{%d}$'%yl for yl in range(0,11,2)])
    plt.show()

def scatter_with_errors(Ns, uNs, y, marker, facecolor, show_scatter=False):
    """
    Helper function for generating scatter plots with error bars for means and standard deviations.
    Ns[i] should be the size of the i^{th} network being plotted
    y[i] should be the value of the statistic being plotted on the i^{th} network 
    uNs should be a list of the unique network sizes included in the plot
    marker and facecolor should be as in matplotlib.pyplot.scatter
    if show_scatter==False, only means and standard deviations are shown for each network size N
    returns scat, a legend handle for use with matplotlib.pyplot.legend
    """
    y_by_N = [y[Ns==N] for N in uNs]
    y_by_N_means = [yy.mean() for yy in y_by_N]
    y_by_N_stds = [yy.std() for yy in y_by_N]
    if show_scatter:
        scat = plt.scatter(Ns, y, s=30,marker=marker, facecolor=facecolor, edgecolor='0.7')
        plt.errorbar(uNs, y_by_N_means, yerr=y_by_N_stds, ecolor='k', c='k', marker=marker, ms=9, mfc='none')
    else:
        scat = plt.errorbar(uNs, y_by_N_means, yerr=y_by_N_stds, ecolor='k', c='k', marker=marker, ms=9, mfc=facecolor)
    plt.xlabel('N')
    return scat

def baseline_comparison_experiments(test_data_id, num_procs):
    """
    Run traverse, the baseline, and the comparison on every network in the test data
    test_data_id should be as in generate_test_data (without file extension)
    num_procs is the number of processors to use in parallel
    """
    _ = run_traverse_experiments(test_data_id,num_procs)
    _ = run_baseline_experiments(test_data_id,num_procs)
    _ = run_TvB_experiments(test_data_id,num_procs)