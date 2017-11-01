import numpy as np

import sys, os

parent_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path+'/../mountainsort/packages/pyms')

from basic.p_compute_templates import compute_templates_helper
from basic.p_extract_clips import extract_clips_helper

from mlpy import readmda, writemda64, DiskReadMda

processor_name='pyms.anneal_segments'
processor_version='0.1'

def anneal_segments(*, timeseries_list, firings_list, firings_out, time_offsets):
    """
    Combine a list of firings files to form a single firings file
    Link firings labels to first firings.mda, all other firings labels are incremented

    Parameters
    ----------
    timeseries_list : INPUT
        A list of paths of timeseries mda files to be used for drift adjustment / time offsets
    firings_list : INPUT
        A list of paths of firings mda files to be concatenated/drift adjusted
    firings_out : OUTPUT
        ...

    time_offsets : string
        An array of time offsets for each firings file. Expect one offset for each firings file.
        ...
    """
    if time_offsets:
        time_offsets=np.fromstring(time_offsets,dtype=np.float_,sep=',')
    else:
        print('No time offsets provided - assuming zero time gap/continuously recorded data')
        time_offsets=np.zeros(len(timeseries_list))
        # Get toffsets based on length of preceeding timeseries - first one left as zero
        for timeseries in range(len(timeseries_list)-1):
            X = DiskReadMda(timeseries_list[timeseries])
            time_offsets[timeseries+1] = time_offsets[timeseries] + X.N2()

    concatenated_firings = concat_and_increment(firings_list, time_offsets)

    (dmatrix, templates, Kmaxes) = get_dmatrix_templates(timeseries_list, firings_list)
    dmatrix[dmatrix < 0] = np.nan  # replace all negative dist numbers (no comparison) with NaN

    #TODO: Improve join function
    pairs_to_merge = get_join_matrix(dmatrix, templates, Kmaxes) # Returns with base 1 adjustment

    pairs_to_merge = np.reshape(pairs_to_merge, (-1, 2))
    pairs_to_merge = pairs_to_merge[~np.isnan(pairs_to_merge).any(axis=1)] # Eliminate all rows with NaN
    pairs_to_merge = pairs_to_merge[np.argsort(pairs_to_merge[:, 0])]  # Assure that input is sorted

    #Propagate merge pairs to lowest label number
    for idx, label in enumerate(pairs_to_merge[:,1]):
        pairs_to_merge[np.isin(pairs_to_merge[:,0],label),0] = pairs_to_merge[idx,0] # Input should be sorted

    #Merge firing labels
    for merge_pair in range(pairs_to_merge.shape[0]):
        concatenated_firings[2, np.isin(concatenated_firings[2, :], pairs_to_merge[merge_pair, 1])] = pairs_to_merge[merge_pair,0] # Already base 1 corrected

    #Write
    return writemda64(concatenated_firings, firings_out)

def get_join_matrix(dmatrix, templates, Kmaxes):
    #Sweep forward in time, linking clust to min dist away
    pairs_to_merge=np.array([])
    f1_adj = 0
    f2_adj = Kmaxes[0]
    for dframe in range(dmatrix.shape[2]):
        for f1_idx in range(dmatrix.shape[1]):
            f2_pair = _nanargmin(dmatrix[f1_idx,:,dframe]) #Ignore nan's and if all nans, return nan
            pairs_to_merge = np.append(pairs_to_merge, np.array([f1_idx + f1_adj + 1, f2_pair + f2_adj + 1])) #Base 1 adjustment to match label
        f1_adj+=Kmaxes[dframe]
        f2_adj+=Kmaxes[dframe+1]
    return pairs_to_merge

def _nanargmin(X):
    #If all nans in slice, return nan; no axis
    try:
        return np.nanargmin(X)
    except ValueError:
        return np.nan

def concat_and_increment(firings_list, time_offsets, increment_labels='true'):
    if len(firings_list) == len(time_offsets):
        concatenated_firings=np.zeros((3,0)) #default to case where the list is empty
        first=True
        for idx, firings in enumerate(firings_list):
            to_append=readmda(firings)
            to_append[1,:]+=time_offsets[idx]
            if not first:
                if increment_labels=='true':
                    to_append[2,:]+=max(concatenated_firings[2,:]) #add the Kmax from previous
            if first:
                concatenated_firings = to_append
            else:
                concatenated_firings = np.append(concatenated_firings, to_append, axis=1)
            first=False
        return concatenated_firings
    else:
        print('Mismatch between number of firings files and number of offsets')

def get_dmatrix_templates(timeseries_list, firings_list):
    X = DiskReadMda(timeseries_list[0])
    M = X.N1()
    clip_size = 100
    num_segments = len(timeseries_list)
    firings_arrays = []
    Kmaxes=[]
    for j in range(num_segments):
        F = readmda(firings_list[j])
        firings_arrays.append(F)
    Kmax = 0;
    for j in range(num_segments):
        F = firings_arrays[j]
        labels = F[2, :]
        Kmax = int(max(Kmax, np.max(labels)))
        Kmaxes.append(Kmax)
    dmatrix = np.ones((Kmax, Kmax, num_segments - 1)) * (-1)
    templates = np.zeros((M, clip_size, Kmax, 2 * (num_segments - 1)))

    for j in range(num_segments - 1):
        print('Computing dmatrix between segments %d and %d' % (j, j + 1))
        (dmatrix0, templates1, templates2) = compute_dmatrix(timeseries_list[j], timeseries_list[j + 1],
                                                             firings_arrays[j], firings_arrays[j + 1],
                                                             clip_size=clip_size)
        dmatrix[0:dmatrix0.shape[0], 0:dmatrix0.shape[1], j] = dmatrix0
        templates[:, :, 0:dmatrix0.shape[0], j * 2] = templates1
        templates[:, :, 0:dmatrix0.shape[1], j * 2 + 1] = templates2
    return (dmatrix, templates, Kmaxes)

def compute_dmatrix(timeseries1, timeseries2, F1, F2, *, clip_size):
    X = DiskReadMda(timeseries1)
    M = X.N1()
    F1b = get_last_events(F1, 100)
    F2b = get_first_events(F2, 100)
    times1 = F1b[1, :].ravel()
    labels1 = F1b[2, :].ravel()
    clips1 = extract_clips_helper(timeseries=timeseries1, times=times1, clip_size=clip_size)
    times2 = F2b[1, :].ravel()
    labels2 = F2b[2, :].ravel()
    clips2 = extract_clips_helper(timeseries=timeseries2, times=times2, clip_size=clip_size)

    K1 = int(max(labels1))
    K2 = int(max(labels2))
    dmatrix = np.zeros((K1, K2))
    templates1 = np.zeros((M, clip_size, K1))
    templates2 = np.zeros((M, clip_size, K2))
    for k1 in range(1, K1 + 1):
        # times1_k1=times1[np.where(labels1==k1)[0]]
        inds_k1 = np.where(labels1 == k1)[0]
        clips1_k1 = clips1[:, :, inds_k1]
        templates1[:, :, k1 - 1] = np.mean(clips1_k1, axis=2)
        for k2 in range(1, K2 + 1):
            # times2_k2=times2[np.where(labels2==k2)[0]]
            inds_k2 = np.where(labels2 == k2)[0]
            clips2_k2 = clips2[:, :, inds_k2]
            templates2[:, :, k2 - 1] = np.mean(clips2_k2, axis=2)
            dmatrix[k1 - 1, k2 - 1] = compute_distance_between_clusters(clips1_k1, clips2_k2)
    return (dmatrix, templates1, templates2)


def get_first_events(firings, num):
    L = firings.shape[1]
    times = firings[1, :]
    labels = firings[2, :]
    K = int(max(labels))
    to_use = np.zeros(L)
    for k in range(1, K + 1):
        inds_k = np.where(labels == k)[0]
        times_k = times[inds_k]
        if (len(times_k) <= num):
            to_use[inds_k] = 1
        else:
            times_k_sorted = np.sort(times_k)
            cutoff = times_k_sorted[num]
            to_use[inds_k[np.where(times_k <= cutoff)[0]]] = 1
    return firings[:, np.where(to_use == 1)[0]]


def get_last_events(firings, num):
    L = firings.shape[1]
    times = firings[1, :]
    labels = firings[2, :]
    K = int(max(labels))
    to_use = np.zeros(L)
    for k in range(1, K + 1):
        inds_k = np.where(labels == k)[0]
        times_k = times[inds_k]
        if (len(times_k) <= num):
            to_use[inds_k] = 1
        else:
            times_k_sorted = np.sort(times_k)
            cutoff = times_k_sorted[len(times_k_sorted) - num]
            to_use[inds_k[np.where(times_k >= cutoff)[0]]] = 1
    return firings[:, np.where(to_use == 1)[0]]

def compute_distance_between_clusters(clips1, clips2):
    centroid1 = np.mean(clips1, axis=2)
    centroid2 = np.mean(clips2, axis=2)
    dist = np.sum((centroid2 - centroid1) ** 2)
    return dist

anneal_segments.name = processor_name
anneal_segments.version = processor_version
anneal_segments.author = 'J Chung and J Magland'