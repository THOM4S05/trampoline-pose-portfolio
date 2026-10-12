"""Compare every exported/overlay frame to native source PTS; make review images.

This verifies timing and format, not manually labelled pose accuracy.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from trampoline_pose.output import validate,sha256
from trampoline_pose.video import probe,decode

p=argparse.ArgumentParser()
p.add_argument('--json',required=True)
p.add_argument('--overlay',required=True)
p.add_argument('--csv')
p.add_argument('--report',required=True)
p.add_argument('--review',required=True)
a=p.parse_args()
d=json.loads(Path(a.json).read_text(encoding='utf-8'))
validate(d)
source=d['source']
assert sha256(source['path'])==source['sha256'], 'Source hash changed'
assert source=={**probe(source['path']),'sha256':source['sha256']}, 'Source metadata changed'
frames=d['frames']
n=0
for timing,img in decode(source['path'],source):
    f=frames[n]
    assert timing['source_pts']==f['source_pts']
    assert timing['timestamp_seconds']==f['timestamp_seconds']
    assert img.shape[:2]==(source['display_height'],source['display_width'])
    n+=1
assert n==len(frames)
# MOV/MP4 edit lists can hide frames counted in nb_frames. The native decoder
# is authoritative for the playable timeline; do not infer count from metadata.
meta=probe(a.overlay)
assert (meta['display_width'],meta['display_height'])==(source['display_width'],source['display_height'])
assert meta['rotation_degrees_ccw']==0
indices=set(np.linspace(0,n-1,min(n,18),dtype=int).tolist())
indices.add(d['selection']['frame_index'])
tiles=[]
maximum_error=0.
count=0
for timing,img in decode(a.overlay,meta):
    f=frames[count]
    error=abs(timing['timestamp_seconds']-f['timestamp_seconds'])
    assert error<=1/90000+1e-9, f'Overlay PTS mismatch at frame {count}'
    maximum_error=max(maximum_error,error)
    if count in indices:
        tile=np.zeros((435,285,3),dtype=np.uint8)
        scale=min(285/img.shape[1],408/img.shape[0])
        preview=cv2.resize(img,(round(img.shape[1]*scale),round(img.shape[0]*scale)))
        top=27+(408-preview.shape[0])//2
        left=(285-preview.shape[1])//2
        tile[top:top+preview.shape[0],left:left+preview.shape[1]]=preview
        cv2.rectangle(tile,(0,0),(285,27),(0,0,0),-1)
        observed=sum(j['status']=='observed' for j in f['joints'].values())
        cv2.putText(tile,f'{count}: {f["track_state"]} {observed}/17',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        tiles.append(tile)
    count+=1
assert count==n
if a.csv:
    with open(a.csv,newline='',encoding='utf-8') as file:
        rows=list(csv.reader(file))
    assert len(rows)==n*17+1
intervals=[]
for f in frames:
    state=f['track_state']
    i=f['frame_index']
    if not intervals or intervals[-1]['state']!=state:
        intervals.append(dict(start_frame=i,end_frame=i,state=state))
    else:
        intervals[-1]['end_frame']=i
statuses=Counter(j['status'] for f in frames for j in f['joints'].values())
report=dict(source_frames=n,json_frames=len(frames),overlay_frames=count,
    container_declared_frames=source['declared_frame_count'],
    source=source,overlay_metadata=meta,max_overlay_timestamp_error_seconds=maximum_error,
    schema_and_invariants_passed=True,source_pts_all_match=True,csv_rows=n*17 if a.csv else None,
    track_states=d['diagnostics']['track_states'],track_intervals=intervals,joint_status_counts=dict(statuses),
    processing_diagnostics=d['diagnostics'],
    checks_not_covered=['Manual joint accuracy','Anatomical left/right assignment','Exhaustive athlete identity review'])
Path(a.report).parent.mkdir(parents=True,exist_ok=True)
Path(a.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
columns=6
tiles+=[np.zeros_like(tiles[0])]*((-len(tiles))%columns)
sheet=np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
Path(a.review).parent.mkdir(parents=True,exist_ok=True)
assert cv2.imwrite(a.review,sheet)
print(json.dumps({k:report[k] for k in ('source_frames','json_frames','overlay_frames','max_overlay_timestamp_error_seconds','track_states','joint_status_counts')},indent=2))
