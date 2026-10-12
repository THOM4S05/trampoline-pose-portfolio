import subprocess
from fractions import Fraction
import av
import numpy as np
import pytest
from trampoline_pose.geometry import apply_affine,crop_geometry,rotated_boxes_to_source
from trampoline_pose.video import decode,probe,orient,VideoWriter
from trampoline_pose.skeleton import FLIP
from trampoline_pose.types import AthleteSelection

def test_edge_crop_round_trip_and_udp_endpoints():
    crop=crop_geometry([-30,-20,50,140])
    points=np.array([[0,0],[-30,-20],[50,140],[30.25,61.5]])
    transformed=apply_affine(points,crop['source_to_input_affine'])
    np.testing.assert_allclose(apply_affine(transformed,crop['input_to_source_affine']),points,atol=1e-8)
    np.testing.assert_allclose(apply_affine([crop['center']],crop['source_to_input_affine']),[[95.5,127.5]])

def test_rotation_keeps_anatomical_names_and_flip_is_involution():
    image=np.arange(6).reshape(2,3)
    np.testing.assert_array_equal(orient(image,90),[[2,5],[1,4],[0,3]])
    np.testing.assert_array_equal(orient(orient(image,90),-90),image)
    assert [FLIP[x] for x in FLIP] == list(range(17))

def make_vfr(path):
    timestamps=[0.,.04,.10,.13]
    with VideoWriter(path,64,48,'25/1') as writer:
        for t in timestamps:
            image=np.zeros((48,64,3),dtype=np.uint8)
            image[:24,:32]=[0,0,255]
            writer.write(image,t)
    return timestamps

def test_vfr_round_trip_native_timestamps(tmp_path):
    path=tmp_path/'vfr.mp4'
    expected=make_vfr(path)
    frames=list(decode(path))
    assert len(frames)==4
    np.testing.assert_allclose([record['timestamp_seconds'] for record,_ in frames],expected,atol=1/90000)

def test_rotated_video_applies_orientation_once(tmp_path):
    source=tmp_path/'source.mp4'
    rotated=tmp_path/'rotated.mp4'
    make_vfr(source)
    subprocess.run(['ffmpeg','-v','error','-y','-display_rotation:v:0','90','-i',str(source),'-c','copy',str(rotated)],check=True)
    meta=probe(rotated)
    assert meta['rotation_degrees_ccw']==90
    frame=next(decode(rotated))[1]
    assert frame.shape[:2]==(64,48)
    assert frame[-8,8,2] > 200

def test_invalid_seed_box():
    with pytest.raises(ValueError):
        AthleteSelection(0,(1,2,0,5))

def test_rotated_detector_box_inverse_uses_full_source_coordinates():
    expected=[[10,20,30,60]]
    for rotation,box in [(0,expected),(90,[[20,70,60,90]]),(180,[[70,20,90,60]]),(270,[[20,10,60,30]])]:
        np.testing.assert_allclose(rotated_boxes_to_source(box,rotation,100,80),expected)
