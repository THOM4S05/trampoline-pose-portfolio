"""Conservative bidirectional association anchored to an explicit athlete seed."""
import cv2
import numpy as np
from .geometry import iou

def appearance(image,box):
    h,w=image.shape[:2]
    a=np.asarray(box,dtype=float)
    # Interior avoids some background, without inventing a body mask.
    margin=(a[2:]-a[:2])*.2
    lo=np.maximum(0,a[:2]+margin).astype(int)
    hi=np.minimum([w,h],a[2:]-margin).astype(int)
    patch=image[lo[1]:hi[1],lo[0]:hi[0]]
    if patch.size==0:
        return [0.]*64
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[8,8],[0,180,0,256]).reshape(-1)
    hist/=max(hist.sum(),1e-9)
    return hist.tolist()

def center(box):
    box=np.asarray(box,dtype=float)
    return (box[:2]+box[2:])/2

def associate(records,selection,config,seed_appearance):
    n=len(records)
    if selection.frame_index>=n:
        raise ValueError(f'Seed frame {selection.frame_index} exceeds decoded frame count {n}')
    anchor=np.asarray(selection.bbox_xyxy,dtype=float)
    anchor_scale=float(np.sqrt(np.prod(anchor[2:]-anchor[:2])))
    output=[None]*n
    seed=selection.frame_index
    output[seed]=dict(bbox_xyxy=anchor.tolist(),track_state='tracked',
                      diagnostics=dict(reason='explicit_athlete_seed',association_margin=None))
    for direction in (1,-1):
        box=anchor.copy()
        velocity=np.zeros(2)
        last_time=records[seed]['timestamp_seconds']
        last_seen=last_time
        template=np.asarray(seed_appearance,dtype=np.float32)
        seed_template=template.copy()
        pending=None
        pending_count=0
        pending_time=None
        pending_hist=None
        pending_late=None
        locked_out=False
        for i in range(seed+direction,n if direction==1 else -1,direction):
            record=records[i]
            t=record['timestamp_seconds']
            dt=abs(t-last_time)
            elapsed=abs(t-last_seen)
            recovering=elapsed*1000>config.uncertain_ms
            late=elapsed*1000>config.reacquisition_ms
            predicted=box+np.r_[velocity,velocity]*dt
            crosses_cut=records[i if direction==1 else i+1]['scene_cut']
            if crosses_cut or locked_out:
                locked_out=True
                output[i]=dict(bbox_xyxy=None,track_state='lost',diagnostics=dict(reason='scene_cut_requires_new_seed'))
                last_time=t
                continue
            ranked=[]
            for index,(candidate,score,hist) in enumerate(zip(record['boxes'],record['scores'],record['appearances'])):
                candidate=np.asarray(candidate,dtype=float)
                seed_scale=float(np.sqrt(np.maximum(1,candidate[2:]-candidate[:2]).prod())/anchor_scale)
                if not config.minimum_seed_scale<=seed_scale<=config.maximum_seed_scale:
                    continue
                c=center(candidate)
                if selection.flight_region_xyxy:
                    x1,y1,x2,y2=selection.flight_region_xyxy
                    if not (x1<=c[0]<=x2 and y1<=c[1]<=y2):
                        continue
                dims=np.maximum(1,box[2:]-box[:2])
                scale=np.sqrt(dims.prod())
                # A partial/occluded box must not shrink the recovery search to
                # spectator size. Keep the explicit athlete seed as a floor.
                search_scale=max(scale,anchor_scale) if recovering else scale
                distance=float(np.linalg.norm(c-center(predicted))/max(search_scale,20))
                ratio=float(np.sqrt(np.maximum(1,candidate[2:]-candidate[:2]).prod())/scale)
                color=float(cv2.compareHist(template,np.asarray(hist,dtype=np.float32),cv2.HISTCMP_BHATTACHARYYA))
                gate=config.reacquisition_distance if recovering else config.association_distance
                if distance>gate or not .25<ratio<4 or color>.75:
                    continue
                # Rotation can compact the athlete's box, but an abrupt size
                # collapse with different appearance is weak identity evidence.
                # Do not let it replace the last athlete box/template.
                if ratio<.55 and color>.35:
                    continue
                if late and config.automatic_reacquisition:
                    seed_color=float(cv2.compareHist(seed_template,np.asarray(hist,dtype=np.float32),cv2.HISTCMP_BHATTACHARYYA))
                    if max(color,seed_color)>config.late_reacquisition_appearance:
                        continue
                cost=float(.55*distance+.2*(1-iou(predicted,candidate))+.15*abs(np.log(ratio))+.25*color-.1*float(score))
                ranked.append((cost,index,candidate,distance,color))
            ranked.sort(key=lambda row:row[0])
            margin=ranked[1][0]-ranked[0][0] if len(ranked)>1 else None
            ambiguous=bool(margin is not None and margin<config.association_margin)
            cost_limit=config.reacquisition_cost if recovering and config.automatic_reacquisition else config.max_association_cost
            weak=bool(ranked and ranked[0][0]>cost_limit)
            reason='detection_associated'
            accepted=bool(ranked) and not ambiguous and not weak and (not late or config.automatic_reacquisition)
            if accepted and recovering:
                candidate=ranked[0][2]
                hist=np.asarray(record['appearances'][ranked[0][1]],dtype=np.float32)
                consistent=(pending is not None and pending_late==late and abs(t-pending_time)<=max(config.uncertain_ms/1000,.1)
                    and np.linalg.norm(center(candidate)-center(pending))/max(np.sqrt(np.prod(candidate[2:]-candidate[:2])),20)<=.65
                    and cv2.compareHist(pending_hist,hist,cv2.HISTCMP_BHATTACHARYYA)<=.35)
                pending_count=pending_count+1 if consistent else 1
                pending=candidate
                pending_hist=hist
                pending_time=t
                pending_late=late
                required=config.late_reacquisition_confirmations if late else 2
                if pending_count<required:
                    accepted=False
                    reason='reacquisition_requires_confirmation' if late else 'reacquisition_requires_second_detection'
                else:
                    pending=None
                    reason='reacquired_after_occlusion' if late else 'reacquired_with_two_detections'
            if accepted:
                candidate=ranked[0][2]
                measured=(center(candidate)-center(box))/max(dt,1e-6)
                velocity=np.zeros(2) if recovering else .5*velocity+.5*measured
                box=candidate
                last_seen=t
                template=.95*template+.05*np.asarray(record['appearances'][ranked[0][1]],dtype=np.float32)
                state='tracked'
            elif elapsed*1000<=config.uncertain_ms:
                box=predicted
                # Detector box centres jump as a gymnast rotates or is occluded.
                # Decay unobserved velocity rather than extrapolating indefinitely.
                velocity*=.5
                state='uncertain'
                reason=reason if reason.startswith('reacquisition') else ('association_cost_above_threshold' if weak else ('ambiguous_person_association' if ambiguous else 'short_detection_gap'))
            else:
                velocity=np.zeros(2)
                state='lost'
                reason=reason if reason.startswith('reacquisition') else (('searching_for_athlete_after_occlusion' if config.automatic_reacquisition else 'reacquisition_window_expired') if late else 'no_unambiguous_athlete')
            if accepted or reason not in ('reacquisition_requires_second_detection','reacquisition_requires_confirmation'):
                pending=None
                pending_count=0
                pending_time=None
                pending_hist=None
                pending_late=None
            output[i]=dict(bbox_xyxy=box.tolist() if state!='lost' else None,track_state=state,
                          diagnostics=dict(reason=reason,association_margin=margin,
                                           candidate_count=len(ranked),best_cost=ranked[0][0] if ranked else None,
                                           time_since_detection_ms=elapsed*1000,
                                           reacquisition_confirmation_count=pending_count,
                                           possible_identity_ambiguity=ambiguous,direction='forward' if direction==1 else 'backward'))
            last_time=t
    return output
