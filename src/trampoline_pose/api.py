import dataclasses
import importlib.metadata
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
from . import __version__
from .backend import OpenMMLabBackend,versions
from .output import sha256,save_json,save_csv
from .refinement import interpolate_short_gaps
from .render import overlay
from .skeleton import NAMES,metadata,derived
from .tracking import associate,appearance
from .types import TrackerConfig,AthleteSelection
from .video import probe,decode,VideoWriter

class TrackingResult:
    def __init__(self,data):
        self.data=data

    def save_json(self,path):
        if Path(path).resolve()==Path(self.data['source']['path']).resolve():
            raise ValueError('JSON must not overwrite source video')
        save_json(self.data,path)

    def save_csv(self,path):
        if Path(path).resolve()==Path(self.data['source']['path']).resolve():
            raise ValueError('CSV must not overwrite source video')
        save_csv(self.data,path)

    def save_video(self,path,raw=False):
        source=self.data['source']
        if Path(path).resolve()==Path(source['path']).resolve():
            raise ValueError('Overlay must not overwrite source video')
        if sha256(source['path'])!=source['sha256']:
            raise ValueError('Source video hash changed; refusing mismatched overlay')
        count=0
        with VideoWriter(path,source['display_width'],source['display_height'],source['avg_frame_rate']) as writer:
            for i,(record,image) in enumerate(decode(source['path'],source,max_frames=len(self.data['frames']))):
                cached=self.data['frames'][i]
                if record['source_pts']!=cached['source_pts']:
                    raise ValueError('Source video changed; overlay timestamps do not match export')
                writer.write(overlay(image,cached,raw),record['timestamp_seconds'])
                count+=1
        if count!=len(self.data['frames']):
            raise ValueError('Source ended before all exported frames were rendered')

class PoseTracker:
    def __init__(self,config=None,backend=None):
        self.config=config or TrackerConfig()
        # Injection supports meaningful end-to-end tests without fake GPU evidence.
        self.backend=backend or OpenMMLabBackend(self.config)

    def process_video(self,path,selection,progress=None,cancel=None,max_frames=None,cache_path=None):
        started=time.perf_counter()
        meta=probe(path)
        if cache_path and Path(cache_path).resolve()==Path(path).resolve():
            raise ValueError('Detection cache must not overwrite source video')
        width,height=meta['display_width'],meta['display_height']
        x1,y1,x2,y2=selection.bbox_xyxy
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):
            raise ValueError('Seed box must lie inside the display-oriented video')
        if max_frames is not None and max_frames<1:
            raise ValueError('max_frames must be positive')
        source_hash=sha256(path)
        key=dict(source_sha256=source_hash,models=self.backend.provenance,
                 detection_score=self.config.detection_score,max_frames=max_frames,cache_version=1,
                 selection=dict(frame_index=selection.frame_index,bbox_xyxy=selection.bbox_xyxy))
        key=json.loads(json.dumps(key,allow_nan=False))
        records=None
        peak_host=psutil.Process().memory_info().rss
        def check_cancel():
            if cancel and cancel():
                raise InterruptedError('Tracking cancelled')
        if cache_path and Path(cache_path).is_file():
            cached=json.loads(Path(cache_path).read_text(encoding='utf-8'))
            # Flight-region limits affect association, not cached detector evidence.
            cached.get('key',{}).get('selection',{}).pop('flight_region_xyxy',None)
            if cached.get('key')==key:
                records=cached['records']
                seed_appearance=cached['seed_appearance']
        used_cache=records is not None
        if records is None:
            records=[]
            seed_appearance=None
            previous=None
            for timing,image in decode(path,meta,max_frames):
                check_cancel()
                boxes,scores=self.backend.detect(image)
                tiny=cv2.resize(image,(64,64)).astype(np.float32)
                difference=float(np.mean(abs(tiny-previous))) if previous is not None else 0.
                previous=tiny
                record={**timing,'boxes':boxes.tolist(),'scores':scores.tolist(),
                        'appearances':[appearance(image,b) for b in boxes],
                        'scene_cut':difference>45.,'scene_difference':difference}
                records.append(record)
                if timing['frame_index']==selection.frame_index:
                    seed_appearance=appearance(image,selection.bbox_xyxy)
                peak_host=max(peak_host,psutil.Process().memory_info().rss)
                if progress:
                    progress('detect',len(records),meta['declared_frame_count'])
            if not records:
                raise ValueError('Input decoded zero frames')
            if seed_appearance is None:
                raise ValueError('Seed frame was not decoded')
            if cache_path:
                cache=Path(cache_path)
                cache.parent.mkdir(parents=True,exist_ok=True)
                temporary=cache.with_name(cache.name+'.tmp')
                temporary.write_text(json.dumps(dict(key=key,records=records,seed_appearance=seed_appearance),allow_nan=False),encoding='utf-8')
                temporary.replace(cache)
        tracks=associate(records,selection,self.config,seed_appearance)
        frames=[]
        for i,(timing,image) in enumerate(decode(path,meta,max_frames)):
            check_cancel()
            if i>=len(records) or timing['source_pts']!=records[i]['source_pts']:
                raise ValueError('Cached detections do not align with decoded source PTS')
            track=tracks[i]
            box=track['bbox_xyxy']
            points=scores=None
            crop=None
            if box is not None:
                points,scores,crop=self.backend.estimate(image,box)
                if np.asarray(points).shape!=(17,2) or len(scores)!=17:
                    raise ValueError('Pose adapter must return all 17 COCO joints')
            joints={}
            for k,name in enumerate(NAMES):
                flags=[]
                raw=dict(x=None,y=None,model_score=None)
                final=dict(x=None,y=None)
                visibility='unknown'
                if points is not None and np.isfinite(points[k]).all() and np.isfinite(scores[k]):
                    x,y=map(float,points[k])
                    score=float(scores[k])
                    raw=dict(x=x,y=y,model_score=score)
                    if not (0<=x<width and 0<=y<height):
                        flags.append('prediction_out_of_frame')
                        visibility='out_of_frame'
                    elif score<self.config.joint_score:
                        flags.append('low_model_score')
                    else:
                        final=dict(x=x,y=y)
                    pbox=crop['padded_bbox_xyxy']
                    distances=[x-pbox[0],pbox[2]-x,y-pbox[1],pbox[3]-y]
                    if min(distances)<.03*max(pbox[2]-pbox[0],pbox[3]-pbox[1]):
                        flags.append('near_crop_edge')
                    if track['track_state']=='uncertain':
                        flags.append('athlete_tracking_uncertain')
                else:
                    flags.append('athlete_lost' if box is None else 'nonfinite_model_prediction')
                joints[name]=dict(raw=raw,final=final,status='observed' if final['x'] is not None else 'missing',
                                  visibility=visibility,confidence=None,flags=flags,correction_displacement_pixels=None)
            frame={**timing,**track,'track_id':'gymnast_1','crop_transform':crop,
                   'joints':joints,'derived_landmarks':derived(joints)}
            frame['diagnostics'].update(scene_difference=records[i]['scene_difference'],
                anatomical_side_identity='pretrained_model_assignment_not_independently_verified')
            frames.append(frame)
            peak_host=max(peak_host,psutil.Process().memory_info().rss)
            if progress:
                progress('pose',len(frames),len(records))
        if len(frames)!=len(records):
            raise ValueError('Pose pass decoded a different frame count')
        if self.config.temporal_refinement:
            frames=interpolate_short_gaps(frames,self.config.max_gap_ms,self.config.max_gap_frames)
        observed=sum(j['status']=='observed' for f in frames for j in f['joints'].values())
        inferred=sum(j['status']=='inferred' for f in frames for j in f['joints'].values())
        states={state:sum(f['track_state']==state for f in frames) for state in ['tracked','uncertain','lost']}
        duration=frames[-1]['timestamp_seconds']
        elapsed=time.perf_counter()-started
        data=dict(schema_version='1.0.0',source={**meta,'sha256':source_hash},processed_frame_count=len(frames),
            coordinate_convention='display-oriented full-resolution pixels; origin top-left; x right; y down',
            score_semantics='Uncalibrated model response scores; visibility unknown; confidence null',
            skeleton=metadata(),models=self.backend.provenance,package_version=__version__,dependencies=versions(),
            effective_config=self.config.to_dict(),selection=dataclasses.asdict(selection),frames=frames,
            diagnostics=dict(runtime_seconds=elapsed,runtime_seconds_per_video_minute=elapsed/duration*60 if duration else None,
                sampled_peak_host_rss_bytes=peak_host,**self.backend.metrics(),track_states=states,
                joint_observed_fraction=observed/(len(frames)*17),joint_inferred_fraction=inferred/(len(frames)*17),
                detection_cache_used=used_cache,processed_entire_source=max_frames is None,
                review_video_timing='one frame per source frame, PTS preserved to 1/90000 second; last-frame duration codec-derived; no audio',
                unverified=['Localisation accuracy','Anatomical left/right identity','Wrong-athlete identity switches','Trampoline difficult poses'],
                refinement_validation='experimental_unvalidated' if self.config.temporal_refinement else 'disabled'))
        return TrackingResult(data)
