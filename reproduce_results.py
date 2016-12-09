#!/usr/bin/env python

import os
import sys
import time
import fxpt_experiments as fe
import roundoff as ro
import figs

# Get configuration options
num_cpus = fe.mp.cpu_count()
print('%d processors detected for multiprocessing.'%num_cpus)
num_procs = int(raw_input('Please enter # of cpus to use: '))
if num_procs not in range(1,num_cpus+1):
    print('Invalid choice... terminating.')
    sys.exit()
scale_option = raw_input('Please enter scale option ("micro", "mini" or "full"): ')
if scale_option not in ['micro','mini','full']:
    print('Invalid choice... terminating.')
    sys.exit()

start_time = time.time()

# Generate test data
if scale_option == 'micro':
    network_sizes = [2,4,7]
    num_samples = [2,2,2]
    _ = fe.generate_test_data(network_sizes, num_samples, test_data_id='micro_base.npz')
    network_sizes = [2,4]
    num_samples = [2,2]
    _ = fe.generate_test_data(network_sizes, num_samples, test_data_id='micro_choose.npz')
    ro_Ns = [2, 7]
    ro_samp_range = range(2)
if scale_option == 'mini':
    network_sizes = [10, 16, 24, 32, 64]
    num_samples = [16, 8, 2, 2, 2]
    _ = fe.generate_test_data(network_sizes, num_samples, test_data_id='mini_base.npz')
    network_sizes = [4,7,10]
    num_samples = [8,4,2]
    _ = fe.generate_test_data(network_sizes, num_samples, test_data_id='mini_choose.npz')
    ro_Ns = [10, 64]
    ro_samp_range = range(2)
if scale_option == 'full':
    network_sizes = range(2,17) + [24,32,48,64,128]
    num_samples = [50]*len(range(2,17)) + [10,10,10,10,5]
    _ = fe.generate_test_data(network_sizes, num_samples, test_data_id='full_base.npz')
    network_sizes = range(2,11)
    num_samples = [15]*len(range(2,11))
    _ = fe.generate_test_data(network_sizes, num_samples, test_data_id='full_choose.npz')
    ro_Ns = [10, 24, 32, 48, 64]
    ro_samp_range = range(5)

# setup sub-directories
for dirname in ['results/','logs/']:
    if not os.path.exists(dirname):
        os.mkdir(dirname)

# Comparison with baseline
test_data_id = '%s_base'%scale_option
_ = fe.run_traverse_experiments(test_data_id,num_procs)
_ = fe.run_baseline_experiments(test_data_id,num_procs)
_ = fe.run_TvB_experiments(test_data_id,num_procs)

# Comparision of c choices
test_data_id = '%s_choose'%scale_option
_ = fe.run_Wc_experiments(test_data_id, num_procs)

# Assessment of round-off errors
test_data_id = '%s_base'%scale_option
_ = ro.get_relative_errors(test_data_id)
_ = ro.run_traverse_rd(test_data_id, ro_Ns, num_procs)
_ = ro.run_baseline_rd(test_data_id, ro_Ns, num_procs)

total_time = time.time()-start_time
print('Finished.  Took a total of %f hours.'%(total_time/3600.))

# Show all the figures (except regular regions)
print('close each figure to open the next one...')
figs.show_all(comp_test_data_ids=[test_data_id]*len(ro_Ns), Ns=ro_Ns, samp_range=ro_samp_range, Wc_test_data_id='%s_choose'%scale_option)

# make regular region figure)
yn=raw_input('Render regular regions figure (takes a few minutes)? Enter "y" or "n": ')
if yn == 'y':
    figs.bad_c_fig()
