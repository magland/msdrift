import numpy as np

import sys, os

parent_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path+'/../mountainsort/packages/pyms')

from basic.p_compute_templates import compute_templates_helper

from mlpy import readmda, writemda64, writemda32, DiskReadMda

processor_name='pyms.extract_subfirings'
processor_version='0.1'

def extract_subfirings(*, firings, t1='', t2='', channels='', channels_array='' timeseries='', firings_out):
    """
    Extract a firings subset based on times and/or channels.
    If a time subset is extracted, the firings are adjusted to t_new = t_original - t1
    If channel(s) are extracted with a timeseries, only clusters with largest amplitude on the given channel (as determined by the average waveform in the time range) will be extracted
    First developed for use with extract_timeseries in inspecting very large datasets

    Parameters
    ----------
    firings : INPUT
        A path of a firings file from which a subset is extracted
    t1 : INPUT
        Start time for extracted firings
    t2 : INPUT
        End time for extracted firings; use -1 OR no value for end of timeseries
    channel_list : INPUT
        A list of channels from which clusters with maximal energy (based on template) will be extracted
    timeseries : INPUT
        A path of a timeseries file from which templates will be calculated if a subset of channels is given
    firings_out : OUTPUT
        The extracted subfirings path
        ...
    """
    firings=readmda(firings)

    if channels:
        _channels=np.fromstring(channels,dtype=int,sep=',')
    elif channels_array:
        _channels=channels_array
    else:
        _channels=np.empty(0)

    if t1:
        print('Time extraction...')
        t_valid=(t1<firings[1,:])#Get bool mask in greater than t1
        if t2 and t2>0:
            t_valid = t_valid * (firings[1,:]<t2)
        firings = firings[:,t_valid]
    else:
        print('Using full time chunk')

    if _channels and timeseries:
        print('Channels extraction...')
        #Determine if need to parse from string
        amps = compute_templates_helper(timeseries, firings, clip_size=1) #Get only amplitude, returns zeroes if empty (M X T X K)
        #Get indices of max chan for each cluster
        main_chan=np.zeros(np.max(firings[2,:]))
        for k in range(np.max(firings[2,:])):
            if np.max(amps[:,:,k]):
                main_chan[k]=np.argmax(amps[:,:,k])+1 #base 1 adj
        labels_valid = np.argwhere(np.isin(main_chan,_channels)) +1 #base 1 adj again
        k_valid=np.isin(firings[2,:],labels_valid)
        firings = firings[:,k_valid]
    else:
        print('Using all channels')

    if t1:
        firings[1,:] -= t1 #adjust t1 to 0

    return writemda64(firings,firings_out)

extract_subfirings.name = processor_name
extract_subfirings.version = processor_version
extract_subfirings.author = 'J Chung'