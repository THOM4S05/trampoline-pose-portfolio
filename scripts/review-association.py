"""Review cached detector association without rerunning either model."""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from trampoline_pose.tracking import associate
from trampoline_pose.types import AthleteSelection,TrackerConfig
from trampoline_pose.video import probe,decode

p=argparse.ArgumentParser()
p.add_argument('--cache',required=True)
p.add_argument('--config',required=True)
p.add_argument('--video',required=True)
p.add_argument('--output',required=True)
p.add_argument('--frames',nargs='+',type=int,default=list(range(0,779,20)))
a=p.parse_args()
c=json.loads(Path(a.cache).read_text())
s=c['key']['selection']
selection=AthleteSelection(s['frame_index'],s['bbox_xyxy'],(50,200,1000,1510))
tracks=associate(c['records'],selection,TrackerConfig.from_yaml(a.config),c['seed_appearance'])
out=Path(a.output)
out.parent.mkdir(parents=True,exist_ok=True)
out.with_suffix('.json').write_text(json.dumps(tracks,indent=2))
print(dict(Counter(t['track_state'] for t in tracks)))
intervals=[]
for i,t in enumerate(tracks):
    if i==0 or t['track_state']!=tracks[i-1]['track_state']:
        intervals.append([i,i,t['track_state']])
    else:
        intervals[-1][1]=i
print(intervals)
tiles=[]
for timing,img in decode(a.video,probe(a.video)):
    i=timing['frame_index']
    if i not in a.frames:
        continue
    t=tracks[i]
    box=t['bbox_xyxy']
    if box is not None:
        x1,y1,x2,y2=map(round,box)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0) if t['track_state']=='tracked' else (0,180,255),4)
    tile=cv2.resize(img[200:1650,50:1000],(228,348))
    cv2.rectangle(tile,(0,0),(228,25),(0,0,0),-1)
    cv2.putText(tile,f'{i}: {t["track_state"]}',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    tiles.append(tile)
columns=8
blank=np.zeros_like(tiles[0])
tiles += [blank]*((-len(tiles))%columns)
sheet=np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
cv2.imwrite(str(out.with_suffix('.jpg')),sheet)
