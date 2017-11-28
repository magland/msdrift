import numpy as np

import sys, os

parent_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path+'/../mountainsort/packages/pyms')

#from basic.p_compute_templates import compute_templates_helper
from basic.p_extract_clips import extract_clips_helper

from mlpy import writemda64, writemda32, DiskReadMda

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

processor_name='pyms.reptrack'
processor_version='0.16'

def reptrack(*, timeseries, firings_out, detect_threshold=3, detect_sign=0, section_size=60*30000, detect_interval=20, detect_channel=0):
    """
    Find representative spikes for the single "best"unit that stretches all the way through the dataset

    Parameters
    ----------
    timeseries : INPUT
        The preprocessed timeseries array
    firings_out : OUTPUT
        The firings file (for the single unit)

    detect_channel : int
        Channel for detection (1-based indexing) or 0 to detect on max over all channels
    detect_threshold : float
        Threshold for detection
    detect_sign : int
        Sign for the detection -1, 0, or 1
    section_size : int
        Size of each section (in timepoints)
    """
    
    X=DiskReadMda(timeseries)    
    M=X.N1()
    N=X.N2()
    num_sections=int(np.floor(N/section_size))
    chunk_infos=[]
    
    S=3 #number of scores to track
    
    clips_prev=np.zeros(0)
    for ii in range(0,num_sections):
        # Read the current chunk
        chunk0=X.readChunk(i1=0,i2=ii*section_size,N1=M,N2=section_size)
        
        # Detect the events during this chunk and offset the times
        if (detect_channel>0):
            signal_for_detect=chunk0[detect_channel-1,:]
        else:
            if detect_sign==0:
                signal_for_detect=np.max(np.abs(chunk0),axis=0)
            elif detect_sign>0:
                signal_for_detect=np.max(chunk0,axis=0)
            else:
                signal_for_detect=np.min(chunk0,axis=0)
        times0=detect(signal_for_detect,detect_threshold,detect_sign,detect_interval)
        times0=times0+ii*section_size
        L0=len(times0)
        
        # Extract the clips for this chunk
        clips0=extract_clips_helper(timeseries=timeseries,times=times0,clip_size=50)
        if ii==0:
            # If this is the first chunk, initialize things
            scores0=np.zeros((S,L0))
            connections0=np.ones(L0)*-1
        else:
            # Some results from the previous chunk
            times_prev=chunk_infos[ii-1]['times']
            scores_prev=chunk_infos[ii-1]['scores']
                        
            # Compute PCA features on the clips from this and the previous chunk combined
            clips_combined=np.concatenate((clips_prev,clips0),axis=2)
            features_combined=compute_clips_features(clips_combined,num_features=10)
            features0=features_combined[:,len(times_prev):]
            features_prev=features_combined[:,0:len(times_prev)]
            
            # Compute the nearest neighbors (candidates for connections)
            nbrs=NearestNeighbors(n_neighbors=50, algorithm='ball_tree')
            nbrs.fit(features_prev.transpose())
            nearest_inds=nbrs.kneighbors(features0.transpose(),return_distance=False)
            
            # For each, find the best connection among the candidates
            scores0=np.zeros((S,L0))
            connections0=np.zeros(L0)
            maxmins_prev=scores_prev[0,:]
            averages_prev=scores_prev[1,:]
            for jj in range(len(times0)):
                tmp=features0[:,jj]
                nearest_inds_jj=nearest_inds[jj,:].tolist()
                dists=np.linalg.norm(features_prev[:,nearest_inds_jj]-tmp.reshape((len(tmp),1)),axis=0)
                normalized_distances=dists/np.linalg.norm(tmp)                
                maxmins=np.maximum(normalized_distances,maxmins_prev[nearest_inds_jj])
                averages=(normalized_distances+averages_prev[nearest_inds_jj]*(ii+1))/(ii+2)
                overall_scores=maxmins+averages*0.1
                ind0=np.argmin(overall_scores)
                scores0[0,jj]=maxmins[ind0]
                scores0[1,jj]=averages[ind0]
                scores0[2,jj]=overall_scores[ind0]
                connections0[jj]=nearest_inds_jj[ind0]
            
        clips_prev=clips0
                
        # Store the results for this chunk
        info0={'times':times0,'connections':connections0,'scores':scores0}
        chunk_infos.append(info0)
    
    rep_times=np.zeros(len(chunk_infos))        
    last_chunk_info=chunk_infos[len(chunk_infos)-1]
    
    last_times=last_chunk_info['times']
    last_overall_scores=last_chunk_info['scores'][S-1,:]
    last_to_first_connections=np.zeros(len(last_times))
    for kk in range(0,len(last_times)):
        ind0=kk
        for ii in range(len(chunk_infos)-2,-1,-1):
            ind0=int(chunk_infos[ii+1]['connections'][ind0])
        last_to_first_connections[kk]=ind0
    
    
    print('Unique:')
    unique1=np.unique(last_to_first_connections)
    print(len(unique1))
    print(len(chunk_infos[0]['times']))
    
    rep_times=[]
    rep_labels=[]
    for aa in range(0,len(unique1)):
        bb=np.where(last_to_first_connections==unique1[aa])[0]
        cc=np.argmax(last_overall_scores[bb])
        ind0=bb[cc]
        rep_times.append(last_chunk_info['times'][ind0])
        rep_labels.append(aa)
        for ii in range(len(chunk_infos)-1,0,-1):
            ind0=int(chunk_infos[ii]['connections'][ind0])
            rep_times.append(chunk_infos[ii-1]['times'][ind0])
            rep_labels.append(aa)        
    
    #ind0=np.argmin(last_chunk_info['scores'][S-1,:]) #Overall score is in row S-1
    #rep_times[len(chunk_infos)-1]=last_chunk_info['times'][ind0]
    #for ii in range(len(chunk_infos)-1,0,-1):
    #    ind0=int(chunk_infos[ii]['connections'][ind0])
    #    rep_times[ii-1]=chunk_infos[ii-1]['times'][ind0]

    firings=np.zeros((3,len(rep_times)))
    for jj in range(len(rep_times)):
        firings[1,jj]=rep_times[jj]
        firings[2,jj]=rep_labels[jj]
    return writemda64(firings,firings_out)
        
        
def compute_clips_features(clips,num_features=10):
    pca = PCA(n_components=num_features)    
    X=np.reshape(clips,(clips.shape[0]*clips.shape[1],clips.shape[2]))
    X=np.transpose(X)
    features=pca.fit_transform(X)
    features=np.transpose(features)
    return features
        
def detect(Y,thresh,sign,detect_interval):
    if (sign<0):
        X=-Y
    elif (sign==0):
        X=np.abs(Y)
    else:
        X=Y
    timepoints_to_consider=np.where(X>=thresh)[0]
    vals_to_consider=X[timepoints_to_consider]
    
    to_use=np.zeros(len(timepoints_to_consider))
    last_best_ind = -1
    for i in range(len(timepoints_to_consider)):
        if last_best_ind >= 0:
            if (timepoints_to_consider[last_best_ind] < timepoints_to_consider[i] - detect_interval):
                last_best_ind = -1;
                # last best ind is not within range. so update it
                if ((i > 0) and (timepoints_to_consider[i - 1] >= timepoints_to_consider[i] - detect_interval)):
                    last_best_ind = i - 1
                    jj = last_best_ind
                    while ((jj - 1 >= 0) and (timepoints_to_consider[jj - 1] >= timepoints_to_consider[i] - detect_interval)):
                        if (vals_to_consider[jj - 1] > vals_to_consider[last_best_ind]):
                            last_best_ind = jj - 1
                        jj=jj-1
                else:
                    last_best_ind = -1
        if (last_best_ind >= 0):
            if (vals_to_consider[i] > vals_to_consider[last_best_ind]):
                to_use[i] = 1
                to_use[last_best_ind] = 0
                last_best_ind = i
            else:
                to_use[i] = 0
        else:
            to_use[i] = 1
            last_best_ind = i

    times=timepoints_to_consider[np.where(to_use==1)[0]]
    return times;                
        

reptrack.name = processor_name
reptrack.version = processor_version
reptrack.author = 'J Magland'

def test_reptrack():
    M=4
    N=30000*10
    timeseries=np.random.normal(0,1,(M,N))
    writemda32(timeseries,'tmp_pre.mda')
    reptrack(timeseries='tmp_pre.mda',firings_out='tmp_firings.mda',detect_sign=1,section_size=1*30000)

if __name__ == '__main__':
    print('Running test')
    test_reptrack()

