import copy
import numpy as np
from trampoline_pose.tracking import associate
from trampoline_pose.types import TrackerConfig,AthleteSelection
from trampoline_pose.refinement import interpolate_short_gaps
from trampoline_pose.skeleton import NAMES

HIST=[1.]+[0.]*63

def record(t,boxes,cut=False):
    return dict(timestamp_seconds=t,boxes=boxes,scores=[.9]*len(boxes),appearances=[HIST]*len(boxes),scene_cut=cut)

def test_mid_clip_seed_bidirectional_identity_with_neighbor():
    rows=[record(i*.04,[[i*2,0,i*2+20,60],[200,0,240,60]]) for i in range(7)]
    tracks=associate(rows,AthleteSelection(3,(6,0,26,60)),TrackerConfig(),HIST)
    assert all(t['track_state']=='tracked' for t in tracks)
    assert [t['bbox_xyxy'][0] for t in tracks]==[i*2 for i in range(7)]

def test_ambiguous_detection_does_not_silently_switch():
    rows=[record(0,[[0,0,20,60]]),record(.04,[[0,0,20,60],[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[1]['track_state']=='uncertain'
    assert tracks[1]['diagnostics']['possible_identity_ambiguity']
    import json
    json.dumps(tracks,allow_nan=False)

def test_long_loss_does_not_reacquire_official():
    rows=[record(0,[[0,0,20,60]])]+[record(i*.1,[]) for i in range(1,7)]+[record(.7,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[-1]['track_state']=='lost'
    assert tracks[-1]['bbox_xyxy'] is None

def test_only_nearby_candidate_still_needs_absolute_association_evidence():
    rows=[record(0,[[0,0,20,60]]),record(.04,[[35,30,55,90]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[1]['track_state']=='uncertain'
    assert tracks[1]['diagnostics']['reason']=='association_cost_above_threshold'
    assert tracks[1]['bbox_xyxy']==[0.,0.,20.,60.]

def test_scene_cut_is_hard_boundary():
    rows=[record(0,[[0,0,20,60]]),record(.04,[[0,0,20,60]],True),record(.08,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert [t['track_state'] for t in tracks]==['tracked','lost','lost']

def test_scale_cannot_gradually_collapse_to_small_spectator():
    boxes=[[50-s/2,50-s/2,50+s/2,50+s/2] for s in (100,75,56,42,31)]
    rows=[record(i*.04,[box]) for i,box in enumerate(boxes)]
    tracks=associate(rows,AthleteSelection(0,tuple(boxes[0])),TrackerConfig(),HIST)
    assert tracks[3]['track_state']=='tracked'
    assert tracks[4]['track_state']=='uncertain'
    assert tracks[4]['bbox_xyxy']==boxes[3]

def test_backward_scene_cut_preserves_first_frame_of_seed_scene():
    rows=[record(0,[[0,0,20,60]]),record(.04,[[0,0,20,60]],True),record(.08,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(2,(0,0,20,60)),TrackerConfig(),HIST)
    assert [t['track_state'] for t in tracks]==['lost','tracked','tracked']

def test_reacquisition_requires_two_detections():
    rows=[record(0,[[0,0,20,60]]),record(.1,[]),record(.2,[]),record(.25,[]),record(.3,[[0,0,20,60]]),record(.34,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[4]['track_state']=='lost'
    assert tracks[5]['track_state']=='tracked'
    assert tracks[5]['diagnostics']['reason']=='reacquired_with_two_detections'

def test_reacquisition_detections_must_be_consecutive():
    rows=[record(0,[[0,0,20,60]]),record(.25,[]),record(.3,[[0,0,20,60]]),record(.32,[]),record(.34,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[-1]['track_state']=='lost'


def test_default_recovers_fast_return_without_loosening_normal_tracking():
    # A returning athlete can move farther than the old 0.65-body-scale gate.
    rows=[record(0,[[0,0,20,60]]),record(.25,[]),
          record(.3,[[25,0,45,60]]),record(.34,[[25,0,45,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[2]['bbox_xyxy'] is None
    assert tracks[3]['bbox_xyxy']==[25.,0.,45.,60.]
    assert tracks[3]['diagnostics']['reason']=='reacquired_with_two_detections'


def test_default_keeps_searching_after_long_occlusion_in_both_directions():
    for direction in (1,-1):
        rows=[record(0,[[0,0,20,60]]),record(.3,[]),record(2.,[]),
              record(2.04,[[10,0,30,60]]),record(2.08,[[12,0,32,60]]),
              record(2.12,[[14,0,34,60]])]
        if direction==-1:
            rows=[{**r,'timestamp_seconds':2.12-r['timestamp_seconds']} for r in reversed(rows)]
        seed=0 if direction==1 else len(rows)-1
        tracks=associate(rows,AthleteSelection(seed,(0,0,20,60)),TrackerConfig(),HIST)
        ordered=tracks if direction==1 else list(reversed(tracks))
        assert [t['track_state'] for t in ordered[-3:]]==['lost','lost','tracked']
        assert ordered[-1]['diagnostics']['reason']=='reacquired_after_occlusion'


def test_late_recovery_rejects_different_clothing_and_ambiguous_people():
    for different_clothing in (True,False):
        rows=[record(0,[[0,0,20,60]]),record(1.,[])]
        for t in (1.04,1.08,1.12,1.16):
            r=record(t,[[0,0,20,60]] if different_clothing else [[0,0,20,60],[0,0,20,60]])
            if different_clothing:
                # Passes the old 0.75 appearance gate but fails strict recovery.
                r['appearances']=[[.5,.5]+[0.]*62]
            rows.append(r)
        tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
        assert all(t['bbox_xyxy'] is None for t in tracks[1:])


def test_late_recovery_confirmation_resets_after_missing_frame():
    rows=[record(0,[[0,0,20,60]]),record(1.,[]),
          record(1.04,[[0,0,20,60]]),record(1.08,[[0,0,20,60]]),record(1.12,[]),
          record(1.16,[[0,0,20,60]]),record(1.2,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[-1]['track_state']=='lost'
    assert tracks[-1]['diagnostics']['reacquisition_confirmation_count']==2


def test_late_recovery_cannot_leave_flight_region_or_jump_to_distant_person():
    for region,box in [((0,0,25,70),[20,0,40,60]),(None,[200,0,220,60])]:
        rows=[record(0,[[0,0,20,60]]),record(1.,[])]+[record(t,[box]) for t in (1.04,1.08,1.12)]
        tracks=associate(rows,AthleteSelection(0,(0,0,20,60),region),TrackerConfig(),HIST)
        assert all(t['bbox_xyxy'] is None for t in tracks[1:])


def test_automatic_recovery_can_be_disabled():
    rows=[record(0,[[0,0,20,60]]),record(1.,[])]+[record(t,[[0,0,20,60]]) for t in (1.04,1.08,1.12)]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(automatic_reacquisition=False),HIST)
    assert tracks[-1]['diagnostics']['reason']=='reacquisition_window_expired'
    assert tracks[-1]['bbox_xyxy'] is None


def test_confirmation_cannot_span_a_scene_cut():
    rows=[record(0,[[0,0,20,60]]),record(1.,[]),record(1.04,[[0,0,20,60]]),
          record(1.08,[[0,0,20,60]],cut=True),record(1.12,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert all(t['track_state']=='lost' for t in tracks[1:])
    assert tracks[-1]['diagnostics']['reason']=='scene_cut_requires_new_seed'


def test_late_confirmation_does_not_count_earlier_weaker_evidence():
    rows=[record(0,[[0,0,20,60]]),record(.3,[]),
          record(.49,[[0,0,20,60]]),record(.53,[[0,0,20,60]]),
          record(.57,[[0,0,20,60]]),record(.61,[[0,0,20,60]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert [t['track_state'] for t in tracks[2:]]==['lost','lost','lost','tracked']

def test_velocity_decays_during_detection_gap():
    rows=[record(0,[[0,0,20,60]]),record(.04,[[8,0,28,60]])]+[record(i*.04,[]) for i in range(2,6)]
    tracks=associate(rows,AthleteSelection(0,(0,0,20,60)),TrackerConfig(),HIST)
    assert tracks[-1]['bbox_xyxy'][0]<16.


def test_occlusion_rejects_sudden_smaller_different_person_and_recovers():
    for direction in (1,-1):
        athlete=[0,0,100,100]
        spectator=[25,25,75,75]
        rows=[record(0,[athlete])]
        for t in (.04,.08,.12,.16,.2,.24,.28):
            r=record(t,[spectator])
            r['appearances']=[[.5,.5]+[0.]*62]
            rows.append(r)
        rows.extend([record(.32,[athlete]),record(.36,[athlete])])
        if direction==-1:
            rows=[{**r,'timestamp_seconds':.36-r['timestamp_seconds']} for r in reversed(rows)]
        seed=0 if direction==1 else len(rows)-1
        tracks=associate(rows,AthleteSelection(seed,tuple(athlete)),TrackerConfig(),HIST)
        ordered=tracks if direction==1 else list(reversed(tracks))
        assert all(t['track_state']!='tracked' for t in ordered[1:-1])
        assert ordered[-1]['bbox_xyxy']==athlete
        assert ordered[-1]['track_state']=='tracked'


def test_compact_pose_with_consistent_appearance_is_still_accepted():
    rows=[record(0,[[0,0,100,100]]),record(.04,[[25,25,75,75]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,100,100)),TrackerConfig(),HIST)
    assert tracks[-1]['track_state']=='tracked'


def test_recovery_search_retains_seed_scale_after_partial_detection():
    rows=[record(0,[[0,0,100,100]]),record(.04,[[25,25,75,75]]),record(.3,[]),
          record(.34,[[80,0,180,100]]),record(.38,[[80,0,180,100]])]
    tracks=associate(rows,AthleteSelection(0,(0,0,100,100)),TrackerConfig(),HIST)
    assert tracks[-1]['track_state']=='tracked'
    assert tracks[-1]['bbox_xyxy']==[80,0,180,100]

def frames(times,missing,states=None):
    rows=[]
    for i,t in enumerate(times):
        joints={n:dict(raw=dict(x=float(i),y=0.,model_score=.9),final=dict(x=float(i),y=0.),status='observed',confidence=None,visibility='unknown',flags=[],correction_displacement_pixels=None) for n in NAMES}
        if i in missing:
            joints['left_wrist']['final']={'x':None,'y':None}
            joints['left_wrist']['status']='missing'
        rows.append(dict(timestamp_seconds=t,track_state=states[i] if states else 'tracked',joints=joints,diagnostics={}))
    return rows

def test_interpolation_uses_timestamps_preserves_raw_and_tags_inferred():
    raw=frames([0.,.02,.08],{1})
    original=copy.deepcopy(raw)
    result=interpolate_short_gaps(raw,100.,3)
    assert raw==original
    assert result[1]['joints']['left_wrist']['final']['x']==.5
    assert result[1]['joints']['left_wrist']['status']=='inferred'
    assert result[1]['joints']['left_wrist']['raw']==raw[1]['joints']['left_wrist']['raw']
    assert result[1]['joints']['left_wrist']['confidence'] is None

def test_interpolation_rejects_long_unbounded_or_identity_lost_gaps():
    for rows in [frames([0,.1,.2],{1}),frames([0,.02,.04],{0}),frames([0,.02,.04],{1},['tracked','lost','tracked'])]:
        result=interpolate_short_gaps(rows,100.,3)
        assert all(result[i]['joints']['left_wrist']['status']=='missing' for i,f in enumerate(rows) if f['joints']['left_wrist']['status']=='missing')
