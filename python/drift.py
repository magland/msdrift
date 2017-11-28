import os
import sys

parent_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path+'/../mountainsort/packages/pyms')

from mlpy import ProcessorManager

import p_concatenate_firings
import p_handle_drift_in_segment
import p_join_segments
import p_anneal_segments
import p_extract_subfirings
import p_reptrack

PM=ProcessorManager()

PM.registerProcessor(p_concatenate_firings.concatenate_firings)
PM.registerProcessor(p_handle_drift_in_segment.handle_drift_in_segment)
PM.registerProcessor(p_join_segments.join_segments)
PM.registerProcessor(p_anneal_segments.anneal_segments)
PM.registerProcessor(p_extract_subfirings.extract_subfirings)
PM.registerProcessor(p_reptrack.reptrack)

if not PM.run(sys.argv):
    exit(-1)
