import numpy as np
import plotter as pt
import rnn_fxpts as rfx
import fxpt_experiments as fe

pt.plt.ion()

# simple duplicate test
neighbors = lambda X, y: (np.fabs(X-y) < 2**-21).all(axis=0)

def setdiff(fxV1, fxV2):
    fxV = []
    for j in range(fxV1.shape[1]):
        if ~(neighbors(fxV2, fxV1[:,[j]])).any():
            fxV.append(fxV1[:,[j]])
    if len(fxV) > 0:
        fxV = np.concatenate(fxV,axis=1)
    else:
        fxV = np.empty((fxV1.shape[0],0))
    return fxV

def union(fxV1, fxV2):
    fxV = np.concatenate((fxV1, fxV2), axis=1)
    fxV = rfx.get_unique_points_recursively(fxV, neighbors=neighbors)
    return fxV 

def add_alpha_mins(W, VA, fxV):
    N = W.shape[0]
    abs_alpha = np.fabs(VA[N,:])
    local_mins = (abs_alpha[:-2] < abs_alpha[1:-1]) & (abs_alpha[1:-1] < abs_alpha[2:])
    seed_mask = np.zeros(abs_alpha.shape,dtype=bool)
    seed_mask[:-2] |= local_mins
    seed_mask[1:-1] |= local_mins
    seed_mask[2:] |= local_mins
    slowV = VA[:N, seed_mask]
    fxV = np.concatenate((fxV,slowV),axis=1)
    fxV, _ = rfx.post_process_fxpts(W, fxV)
    return fxV

N = 10
all_cloops = 0.
all_comps = 0.
for samp in range(50):

    npz = {'T': fe.load_npz_file('results/traverse_full_base_N_%d_s_%d.npz'%(N,samp)),
         'B': fe.load_npz_file('results/baseline_full_base_N_%d_s_%d.npz'%(N,samp))}
    W = npz['T']['W']
    c = npz['T']['c']

    # add alpha mins
    npz['T']['fxV_unique'] = add_alpha_mins(W, npz['T']['VA'], npz['T']['fxV_unique'])
    
    k = 0 # current component
    
    found = [npz['T']['fxV_unique']] # initial Traverse results
    seeds = setdiff(npz['B']['fxV_unique'],found[0]) # B - T initial seeds
    seed = [np.zeros((N,1))]
    new = [np.empty((N,0))]
    statuses = ['success']
    
    # print('%d: %s'%(k,statuses[k]))
    # print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds.shape[1]))
    
    # pt.plotNd(npz['T']['VA'],3*np.ones((N,1))*np.array([[-1,1]]),'r-')
    # raw_input('...')
    
    while seeds.shape[1] > 0:
        k += 1
        seed.append(seeds[:,[0]])
        seeds = seeds[:,1:]
    
        # need to go both directions! and fxV shouldn't include origin if none found
        va = np.concatenate((seed[k],np.array([[0]])),axis=0) # include alpha
        status, fxV, VA, c, _, _, _ = rfx.traverse(W, va=va, c=c, max_traverse_steps = 2**20)
        statuses.append(status)
    
        if status=='Success':
        
            #N!=3:
            # pt.plotNd(npz['T']['VA'],3*np.ones((N,1))*np.array([[-1,1]]),'r-')
            # pt.plotNd(VA,3*np.ones((N,1))*np.array([[-1,1]]),'b-')

            # #N=3:
            # ax = pt.plt.gca(projection='3d')
            # pt.quiver(ax, npz['T']['VA'][:N,:], c*npz['T']['VA'][[N],:])
            # pt.plot(ax, npz['T']['VA'][:N,:],'r-')
            # pt.plot(ax, npz['T']['fxV_unique'][:N,:],'ko')
            # pt.plot(ax, seed[k],'go')
            # pt.plot(ax, VA[:N,:],'b-')
            # # pt.plot(ax, -VA[:N,:],'b-')

            # # alpha:
            # pt.plt.plot(npz['T']['VA'][N,:])
            # pt.plt.plot(np.zeros(npz['T']['VA'][N,:].shape))
            # pt.plt.ylim([-1,1])
            # print('%g'%np.fabs(np.tanh(W.dot(seed[k]))-seed[k]).max())
            # print('%g'%np.fabs(found[0]-seed[k]).max(axis=0).min())
            # raw_input('...')
            pass
    
        # need rfx.post_process to use simpler neighbor!
        fxV = np.concatenate((seed[k],fxV),axis=1) # be sure to include seed(t)
        fxV, _ = rfx.post_process_fxpts(W, fxV)
    
        new.append(fxV)
        found.append(union(found[k-1], new[k]))
        seeds = setdiff(seeds, new[k])
    
        # print('%d: %s'%(k,statuses[k]))
        # print('|found|=%d, |new|=%d, |seeds|=%d'%(found[k].shape[1], new[k].shape[1], seeds.shape[1]))
        # # print(new[k]) # new includes origin every time because of post process
    
    # print(all([s=='Closed loop detected' for s in statuses[1:]]))
    cloops = sum([s=='Closed loop detected' for s in statuses[1:]])
    print('%d: %d of %d cloop?'%(samp,cloops, len(statuses)-1))
    all_cloops += cloops
    all_comps += len(statuses)-1
    # Not all cloops all the time! Local minima of |alpha| that don't change sign!!
        

print('In all, %d cloops of %d comps'%(all_cloops,all_comps))
