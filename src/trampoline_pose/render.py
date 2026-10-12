import cv2
from .skeleton import NAMES,EDGES

COLORS={'observed':(0,220,0),'corrected':(255,180,0),'inferred':(0,180,255),'missing':(0,0,255)}

def overlay(image,record,raw=False):
    image=image.copy()
    box=record['bbox_xyxy']
    if box is not None:
        color=(0,220,0) if record['track_state']=='tracked' else (0,180,255)
        cv2.rectangle(image,tuple(round(x) for x in box[:2]),tuple(round(x) for x in box[2:]),color,2)
    joints=record['joints']
    def get(n):
        j=joints[n]
        p=j['raw' if raw else 'final']
        if p['x'] is None:
            return None
        return tuple(round(p[a]) for a in ('x','y'))
    for a,b in EDGES:
        pa,pb=get(NAMES[a]),get(NAMES[b])
        if pa is not None and pb is not None:
            inferred='inferred' in (joints[NAMES[a]]['status'],joints[NAMES[b]]['status'])
            uncertain=record['track_state']=='uncertain'
            cv2.line(image,pa,pb,COLORS['inferred' if inferred or uncertain else 'observed'],2,cv2.LINE_AA)
    for n in NAMES:
        p=get(n)
        if p is None:
            continue
        j=joints[n]
        color=COLORS[j['status']] if not raw else (255,0,255)
        if record['track_state']=='uncertain' and not raw:
            color=COLORS['inferred']
        if j['status']=='inferred' and not raw:
            cv2.circle(image,p,5,color,1,cv2.LINE_AA)
        else:
            cv2.circle(image,p,4,color,-1,cv2.LINE_AA)
    text=f"frame {record['frame_index']}  {record['timestamp_seconds']:.3f}s  {record['track_state']}  {'RAW' if raw else 'FINAL'}"
    cv2.rectangle(image,(0,0),(min(image.shape[1],850),64),(0,0,0),-1)
    cv2.putText(image,text,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(image,'green: model   orange: uncertain/ring inferred   L/R: anatomical',(8,48),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    return image
