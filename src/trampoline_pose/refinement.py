"""Optional auditable short-gap interpolation; disabled by default, no smoothing."""
import copy
import math
from .skeleton import NAMES,derived

def interpolate_short_gaps(frames,max_gap_ms=100.,max_gap_frames=3):
    result=copy.deepcopy(frames)
    for name in NAMES:
        i=0
        while i<len(result):
            if result[i]['joints'][name]['status']!='missing':
                i+=1
                continue
            start=i
            while i<len(result) and result[i]['joints'][name]['status']=='missing':
                i+=1
            end=i
            if start==0 or end==len(result) or end-start>max_gap_frames:
                continue
            window=result[start-1:end+1]
            if any(f['track_state']!='tracked' for f in window):
                continue
            if any(f['diagnostics'].get('possible_identity_ambiguity',False) for f in window):
                continue
            ta,tb=window[0]['timestamp_seconds'],window[-1]['timestamp_seconds']
            if (tb-ta)*1000>max_gap_ms:
                continue
            a,b=window[0]['joints'][name]['final'],window[-1]['joints'][name]['final']
            for f in result[start:end]:
                j=f['joints'][name]
                alpha=(f['timestamp_seconds']-ta)/(tb-ta)
                j['final']={axis:a[axis]+alpha*(b[axis]-a[axis]) for axis in ('x','y')}
                j['status']='inferred'
                j['flags'].append('experimental_short_gap_interpolation')
                if j['raw']['x'] is not None:
                    j['correction_displacement_pixels']=math.hypot(j['final']['x']-j['raw']['x'],j['final']['y']-j['raw']['y'])
    for f in result:
        f['derived_landmarks']=derived(f['joints'])
    return result
