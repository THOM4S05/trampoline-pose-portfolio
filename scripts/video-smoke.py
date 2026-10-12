"""Actual CUDA video pipeline test using a translated official COCO image.

This is a functional integration test, never an accuracy benchmark or sample run.
"""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from trampoline_pose import PoseTracker,TrackerConfig,AthleteSelection
from trampoline_pose.video import VideoWriter,decode,probe
from trampoline_pose.output import validate

ROOT=Path(__file__).resolve().parents[1]

def main():
    (ROOT/'reports').mkdir(parents=True, exist_ok=True)
    output=ROOT/'outputs/synthetic-smoke'
    output.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(ROOT/'models/smoke.jpg'))
    assert image is not None
    path=output/'input.mp4'
    stamps=[i/30+(0.01 if i>=15 else 0) for i in range(60)]
    with VideoWriter(path,image.shape[1],image.shape[0],'30/1') as writer:
        for i,t in enumerate(stamps):
            shifted=cv2.warpAffine(image,np.array([[1,0,i*.3],[0,1,0]],dtype=float),(image.shape[1],image.shape[0]),borderMode=cv2.BORDER_REFLECT)
            writer.write(shifted,t)
    config=TrackerConfig.from_yaml(ROOT/'configs/default.yaml')
    tracker=PoseTracker(config)
    frame=next(decode(path))[1]
    boxes,scores=tracker.backend.detect(frame)
    assert len(boxes)==1
    # Seed later than zero to exercise backward association too.
    seed=30
    box=boxes[0].copy()
    box[[0,2]]+=seed*.3
    result=tracker.process_video(path,AthleteSelection(seed,tuple(box)),cache_path=output/'detections.json',
        progress=lambda stage,count,total:print(f'{stage} {count}/{total}',flush=True) if count%30==0 else None)
    result.save_json(output/'joints.json')
    result.save_csv(output/'joints.csv')
    started=time.perf_counter()
    result.save_video(output/'overlay.mp4')
    overlay_seconds=time.perf_counter()-started
    exported=json.loads((output/'joints.json').read_text())
    validate(exported)
    assert len(exported['frames'])==60
    assert all(f['track_state']=='tracked' for f in exported['frames'])
    assert exported['diagnostics']['joint_observed_fraction']>.8
    rendered=list(decode(output/'overlay.mp4'))
    assert len(rendered)==60
    np.testing.assert_allclose([r['timestamp_seconds'] for r,_ in rendered],stamps,atol=1/90000)
    assert exported['models']['runtime']['device']=='cuda:0'
    cv2.imwrite(str(ROOT/'reports/video-smoke-frame.jpg'),rendered[30][1])
    report=dict(passed=True,test_source='Translated official MMPose COCO image, 60 VFR frames; not trampoline footage',
        seed_frame=seed,decoded_frames=60,overlay_decoded_frames=60,overlay_timestamps_match=True,
        overlay_encode_seconds=overlay_seconds,source=probe(path),**exported['diagnostics'])
    (ROOT/'reports/video-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
