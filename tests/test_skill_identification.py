"""Independent synthetic checks; supplied clip labels are not test fixtures."""
import copy
import csv
import math
import json

import numpy as np
import pytest

from trampoline_pose.skill_identification import (
    analyze, centre, compare, interpolate_short, rotation_evidence, shape_evidence, usable, write_csv, horizontal_rebound_gaps,
)


def joint(x, y, status="observed", flags=None):
    return dict(final=dict(x=x, y=y), status=status, visibility="unknown", flags=flags or [])


def bouncing_export(rate, cycles=4, period=1.4):
    frames = []
    # Nonuniform PTS, distinct from nominal FPS, with exact endpoint.
    t = np.linspace(0, period * cycles, round(rate * period * cycles) + 1)
    t[1:-1] += np.sin(np.arange(1, len(t) - 1)) / rate * .1
    for i, ts in enumerate(t):
        y = 300 - 200 * math.cos(ts * 2 * math.pi / period)
        js = {}
        for side, x in (("left", 195), ("right", 205)):
            for name, offset in (("hip", 0), ("shoulder", -100), ("knee", 50), ("ankle", 100), ("ear", -150)):
                js[side + "_" + name] = joint(x, y + offset)
        js["nose"] = joint(225, y - 150)
        frames.append(dict(frame_index=i, timestamp_seconds=float(ts), track_state="tracked", joints=js))
    return dict(schema_version="1.0.0", source=dict(sha256="synthetic"), frames=frames)


def test_bounce_count_and_rate_invariance_without_labels():
    low, high = analyze(bouncing_export(30)), analyze(bouncing_export(240))
    json.dumps(low, allow_nan=False)
    assert len(low["contacts"]) == len(high["contacts"]) == 4
    assert len(low["skills"]) == 5  # four contacts plus opening/final partial spans
    assert low["skills"][0]["start_frame"] is None
    assert low["skills"][-1]["end_frame"] is None
    assert all(s["twist"] == "deferred" for s in low["skills"])
    assert low["facing"]["screen_direction"] == "right"
    assert np.allclose([c["timestamp_seconds"] for c in low["contacts"]],
                       [c["timestamp_seconds"] for c in high["contacts"]], atol=.04)
    assert len(analyze(bouncing_export(60, cycles=2))["contacts"]) == 2


@pytest.mark.parametrize("turns,facing,direction,count", [(-1, 1, "backward", 1),
                                                        (2, -1, "backward", 2),
                                                        (1, 1, "forward", 1), (0, 1, "none", 0),
                                                        (-.25, 1, "backward", .25), (.25, 1, "forward", .25),
                                                        (-.5, 1, "backward", .5), (.75, 1, "forward", .75),
                                                        (-1.25, 1, "backward", 1.25)])
def test_projected_rotation_count_and_direction(turns, facing, direction, count):
    t = np.linspace(0, 2, 481)
    theta = t / t[-1] * turns * 2 * np.pi
    hip = np.zeros((len(t), 2))
    shoulder = np.column_stack([100 * np.sin(theta), -100 * np.cos(theta)])
    r = rotation_evidence(t, shoulder, hip, facing, .08, False)
    assert r["somersault_count"] == count
    assert r["direction"] == direction
    assert r['somersault_quarters'] == round(count * 4)
    assert rotation_evidence(t, shoulder, hip, facing, .08, True)["somersault_count"] is None
    shoulder[180:310] = np.nan
    gap_result = rotation_evidence(t, shoulder, hip, facing, .08, False)
    assert gap_result['somersault_count'] is None


def test_horizontal_rebound_brackets_require_descent_ascent_and_horizontal_body():
    t = np.arange(0, 2, .01)
    hip = np.column_stack([np.zeros(len(t)), 300 - 500 * (t - 1) ** 2])
    shoulder = hip + [100, 0]
    hip[90:111] = np.nan
    shoulder[90:111] = np.nan
    candidates = horizontal_rebound_gaps(t, shoulder, hip, .08)
    assert candidates == [(89, 111)]
    upright = hip + [0, -100]
    assert horizontal_rebound_gaps(t, upright, hip, .08) == []
    monotonic = np.column_stack([np.zeros(len(t)), t * 100])
    monotonic[90:111] = np.nan
    assert horizontal_rebound_gaps(t, monotonic + [100, 0], monotonic, .08) == []


@pytest.mark.parametrize('sign,facing,direction', [(1, 1, 'forward'), (-1, -1, 'forward'), (-1, 1, 'backward')])
def test_short_inversion_gap_has_unique_direction_supported_arc(sign, facing, direction):
    t = np.arange(0, 1.201, .01)
    theta = sign * t / 1.2 * 2 * np.pi
    hip = np.zeros((len(t), 2))
    shoulder = np.column_stack([100 * np.sin(theta), -100 * np.cos(theta)])
    shoulder[(t > .48) & (t < .72)] = np.nan
    r = rotation_evidence(t, shoulder, hip, facing, .08, False)
    assert r['somersault_quarters'] == 4
    assert r['direction'] == direction
    assert len(r['reconstructed_inversion_gaps']) == 1
    assert 'inversion_gap_arc_estimated_needs_review' in r['reasons']
    # Longer occlusions and inconsistent motion remain unresolved.
    shoulder[(t > .35) & (t < .85)] = np.nan
    assert rotation_evidence(t, shoulder, hip, facing, .08, False)['somersault_count'] is None


def test_inversion_gap_with_reversing_departure_is_not_reconstructed():
    t = np.arange(0, 1.201, .01)
    theta = t / 1.2 * 2 * np.pi
    theta[t >= .72] = theta[t >= .72][0] - (t[t >= .72] - .72) * 5
    hip = np.zeros((len(t), 2))
    shoulder = np.column_stack([100 * np.sin(theta), -100 * np.cos(theta)])
    shoulder[(t > .48) & (t < .72)] = np.nan
    r = rotation_evidence(t, shoulder, hip, 1, .08, False)
    assert r['somersault_count'] is None
    assert not r['reconstructed_inversion_gaps']


@pytest.mark.parametrize('sign', [-1, 1])
def test_gap_near_inversion_and_endpoint_smoothing_do_not_lose_a_turn(sign):
    t = np.arange(0, 1.201, .01)
    theta = sign * t / 1.2 * 2 * np.pi
    hip = np.zeros((len(t), 2))
    theta[38] += sign * .17  # a modest final observation before occlusion
    shoulder = np.column_stack([100 * np.sin(theta), -100 * np.cos(theta)])
    shoulder[39:50] = np.nan
    r = rotation_evidence(t, shoulder, hip, 1, .08, False)
    assert r['somersault_quarters'] == 4
    assert r['direction'] == ('forward' if sign > 0 else 'backward')
    assert r['reconstructed_inversion_gaps']
    # Averaged velocity must still veto a genuinely large raw axis jump.
    theta[38] += sign * .8
    shoulder[38] = [100 * np.sin(theta[38]), -100 * np.cos(theta[38])]
    assert rotation_evidence(t, shoulder, hip, 1, .08, False)['somersault_count'] is None


def test_rotation_filters_small_pose_jitter_but_rejects_large_axis_jump():
    t = np.arange(0, 2.01, .01)
    theta = -t / 2 * 2 * np.pi
    hip = np.zeros((len(t), 2))
    jittered = theta.copy()
    jittered[110] += .24
    shoulder = np.column_stack([100 * np.sin(jittered), -100 * np.cos(jittered)])
    result = rotation_evidence(t, shoulder, hip, 1, .08, False)
    assert result['somersault_quarters'] == 4
    assert result['direction'] == 'backward'
    jittered[110] = theta[110] + 1
    shoulder = np.column_stack([100 * np.sin(jittered), -100 * np.cos(jittered)])
    assert rotation_evidence(t, shoulder, hip, 1, .08, False)['somersault_quarters'] is None


def test_upright_rebound_gap_uses_local_height_and_asymmetric_endpoints():
    t = np.arange(0, 12, .01)
    hip = np.column_stack([np.zeros(len(t)), 300 - 200 * np.cos(2 * np.pi * t / 1.4) - 20 * t])
    shoulder = hip + [0, -100]
    # The descent is hidden earlier than the ascent, below a changing camera envelope.
    hip[888:913], shoulder[888:913] = np.nan, np.nan
    assert horizontal_rebound_gaps(t, shoulder, hip, .08, include_upright=True) == [(887, 913)]
    assert horizontal_rebound_gaps(t, shoulder, hip, .08) == []
    # An occluded apex reverses velocity in the other direction; it is not contact.
    hip = np.column_stack([np.zeros(len(t)), 300 - 200 * np.cos(2 * np.pi * t / 1.4) - 20 * t])
    shoulder = hip + [0, -100]
    missing = (t >= 8.28) & (t <= 8.52)
    hip[missing], shoulder[missing] = np.nan, np.nan
    assert horizontal_rebound_gaps(t, shoulder, hip, .08, include_upright=True) == []


def test_occluded_feet_contacts_survive_camera_height_drift():
    data = bouncing_export(60, cycles=8)
    for frame in data['frames']:
        for name, point in frame['joints'].items():
            point['final']['y'] -= 20 * frame['timestamp_seconds']
            if name.endswith('_ankle'):
                point['status'] = 'missing'
    result = analyze(data)
    assert len(result['contacts']) == 8
    assert all(c['method'] == 'occluded_feet_body_rebound_proxy' for c in result['contacts'])


def test_continuous_horizontal_bounces_use_body_contact_evidence():
    data = bouncing_export(30, cycles=2)
    for frame in data['frames']:
        for side in ('left', 'right'):
            hip = frame['joints'][side + '_hip']['final']
            frame['joints'][side + '_shoulder'] = joint(hip['x'] - 100, hip['y'])
    result = analyze(data)
    assert len(result['contacts']) == 2
    assert all(c['method'] == 'horizontal_hip_deceleration_proxy' for c in result['contacts'])


def test_occluded_feet_rebound_requires_supported_torso():
    data = bouncing_export(30, cycles=2)
    for frame in data['frames']:
        for side in ('left', 'right'):
            frame['joints'][side + '_ankle'] = joint(None, None, status='missing')
    result = analyze(data)
    assert len(result['contacts']) == 2
    assert all(c['method'] == 'occluded_feet_body_rebound_proxy' for c in result['contacts'])
    for frame in data['frames']:
        for side in ('left', 'right'):
            frame['joints'][side + '_shoulder'] = joint(None, None, status='missing')
    assert analyze(data)['contacts'] == []


def test_upright_tuck_shape_is_selected_in_flight_not_contact_flexion():
    t = np.arange(0, 1.01, .01)
    hip = np.column_stack([np.zeros(len(t)), 300 - 200 * np.sin(np.pi * t)])
    shoulder = hip + [0, -100]
    knee, ankle = hip + [0, 70], hip + [0, 140]
    tuck = (t >= .35) & (t <= .65)
    knee[tuck], ankle[tuck] = hip[tuck] + [60, -80], hip[tuck] + [0, 10]
    evidence = shape_evidence(t, shoulder, hip, knee, ankle, .12)
    assert evidence['execution_position'] == 'tuck_like'
    assert .35 <= evidence['shape_timestamp_seconds'] <= .65
    knee, ankle = hip + [0, 70], hip + [0, 140]
    contact = (t < .15) | (t > .85)
    knee[contact], ankle[contact] = hip[contact] + [60, -80], hip[contact] + [0, 10]
    evidence = shape_evidence(t, shoulder, hip, knee, ankle, .12)
    assert evidence['execution_position'] == 'straight_like'


def test_occluded_tighter_shape_offers_review_time_without_inventing_angles():
    t = np.arange(0, 1.01, .01)
    hip = np.zeros((len(t), 2))
    shoulder = hip + [0, -100]
    knee, ankle = hip + [100, 0], hip + [200, 0]
    knee[[40, 66]] = [70, -70]
    ankle[[40, 66]] = [140, -70]
    knee[41:66], ankle[41:66] = np.nan, np.nan
    r = shape_evidence(t, shoulder, hip, knee, ankle, .12)
    assert r['review_shape_timestamp_seconds'] == pytest.approx(.53)
    assert r['tightest_shape_review_interval_seconds'] == [.4, .66]
    assert r['shape_timestamp_seconds'] != r['review_shape_timestamp_seconds']
    assert r['sustained_hip_angle_degrees'] == 90
    assert 'tighter_shape_may_be_hidden_by_missing_joints' in r['reasons']


def test_short_gap_accepts_bounded_pose_wobble_but_not_reversing_motion():
    t = np.arange(0, 1.201, .01)
    theta = t / 1.2 * 2 * np.pi
    wobble = (t >= .66) & (t <= .8)
    theta[wobble] += .08 * np.sin((t[wobble] - .66) / .14 * 6 * np.pi)
    hip = np.zeros((len(t), 2))
    shoulder = np.column_stack([100 * np.sin(theta), -100 * np.cos(theta)])
    shoulder[46:66] = np.nan
    r = rotation_evidence(t, shoulder, hip, 1, .08, False)
    assert r['somersault_quarters'] == 4
    assert r['reconstructed_inversion_gaps']


def test_missing_uncertain_inferred_and_crop_edge_are_not_observations():
    frame = bouncing_export(30)["frames"][0]
    assert usable(frame, "left_hip") is not None
    for status in ("inferred", "missing"):
        f = copy.deepcopy(frame)
        f["joints"]["left_hip"]["status"] = status
        assert np.isnan(centre(f, "hip")).all()
    frame["track_state"] = "uncertain"
    assert usable(frame, "left_hip") is None
    frame["track_state"] = "tracked"
    frame["joints"]["left_hip"]["flags"] = ["near_crop_edge"]
    assert usable(frame, "left_hip") is None


def test_short_interpolation_never_fills_long_outages_or_endpoints():
    t = np.arange(0, 1, .01)
    v = t.copy()
    v[:5] = np.nan
    v[20:23] = np.nan
    v[40:60] = np.nan
    out = interpolate_short(t, v, t, .08)
    assert np.isnan(out[:5]).all()
    assert np.isfinite(out[20:23]).all()
    assert np.isnan(out[40:60]).all()


def test_anatomical_side_swaps_do_not_change_results():
    original = bouncing_export(30, cycles=2)
    swapped = copy.deepcopy(original)
    for f in swapped["frames"]:
        for part in ("hip", "shoulder", "knee", "ankle", "ear"):
            a, b = "left_" + part, "right_" + part
            f["joints"][a], f["joints"][b] = f["joints"][b], f["joints"][a]
    assert analyze(original) == analyze(swapped)


def test_projected_shape_separates_straight_pike_tuck_and_unknown():
    t = np.linspace(0, .5, 51)
    hip = np.zeros((len(t), 2))
    shoulder = np.tile([100, 0], (len(t), 1))  # inverted/sideways active rotation
    knee = np.tile([50, 86.6], (len(t), 1))
    ankle = 2 * knee
    pike = shape_evidence(t, shoulder, hip, knee, ankle, .12)
    assert pike["body_position"] == "pike"
    assert pike["sustained_knee_angle_degrees"] > 179
    tuck = shape_evidence(t, shoulder, hip, knee, np.tile([100, 0], (len(t), 1)), .12)
    assert tuck["body_position"] == "tuck"
    straight = shape_evidence(t, shoulder, hip, -shoulder, -2 * shoulder, .12)
    assert straight["body_position"] == "straight"
    knee[:] = np.nan
    assert shape_evidence(t, shoulder, hip, knee, ankle, .12)["body_position"] == "unknown"


def test_bad_timestamps_fail_and_no_evidence_returns_no_contacts():
    data = bouncing_export(30)
    data["frames"][1]["timestamp_seconds"] = data["frames"][0]["timestamp_seconds"]
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze(data)
    data = bouncing_export(30)
    for f in data["frames"]:
        f["track_state"] = "lost"
    assert analyze(data)["skills"] == []


def test_csv_allows_unknown_rows_with_different_evidence_fields(tmp_path):
    path = tmp_path / "skills.csv"
    write_csv(path, [dict(skill=1, shape="unknown"), dict(skill=2, shape="pike", angle=60)])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["angle"] == ""
    assert rows[1]["angle"] == "60"


def test_source_pts_mismatch_rejected():
    data = bouncing_export(30)
    for f in data["frames"]:
        f["source_pts"] = f["frame_index"]
        f["source_time_base"] = "1/30"
    with pytest.raises(ValueError, match="source PTS"):
        analyze(data)


def test_comparison_keeps_reference_identity_out_of_predictions(tmp_path):
    result = analyze(bouncing_export(30, cycles=2))
    before = copy.deepcopy(result)
    labels = tmp_path / "labels.csv"
    fields = ["skill_index", "end_seconds", "start_seconds", "end_frame_0based", "start_frame_0based",
              "somersault_quarters", "skill_name", "notes", "body_position", "somersault_direction", "review_status"]
    with labels.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(result["skills"], 1):
            writer.writerow(dict(skill_index=i, end_seconds=s["visible_end_seconds"],
                                 start_seconds="" if i == 1 else s["visible_start_seconds"],
                                 end_frame_0based=s["end_frame"], start_frame_0based=s["start_frame"],
                                 somersault_quarters=0, skill_name="Arbitrary reference identity",
                                 notes="supplied deviation", body_position="straight", somersault_direction="none",
                                 review_status="synthetic"))
    comparison = compare(result, labels)
    assert result == before
    assert all(r["supplied_skill_identity"] == "Arbitrary reference identity" for r in comparison["rows"])
    assert comparison["rows"][0]["rotation_match"] is None

@pytest.mark.parametrize('rate', [30, 60, 120])
@pytest.mark.parametrize('mirrored', [False, True])
@pytest.mark.parametrize('start,end,twist,name', [
    ('feet', 'seat', False, 'Seat drop'),
    ('seat', 'feet', True, 'Half twist from seat to feet'),
    ('seat', 'feet', False, 'Return from seat to feet'),
])
def test_seat_transition_generalizes_pose_scale_rate_and_facing(rate, mirrored, start, end, twist, name):
    from trampoline_pose.skill_identification import seat_skill_evidence
    frames = []
    for i, stamp in enumerate(np.linspace(0, 1.6, round(rate * 1.6) + 1)):
        position = start if stamp < .8 else end
        sign = (-1 if mirrored else 1) * (-1 if twist and stamp >= .8 else 1)
        scale = 1.7
        js = {}
        for side in ('left', 'right'):
            for part, xy in dict(hip=(0, 0), shoulder=(0, -100), ear=(0, -145),
                                knee=(65, 0) if position == 'seat' else (0, 60),
                                ankle=(130, 0) if position == 'seat' else (0, 130)).items():
                js[side + '_' + part] = joint(500 + sign * xy[0] * scale, 400 + xy[1] * scale)
        js['nose'] = joint(500 + sign * 20 * scale, 400 - 145 * scale)
        frames.append(dict(timestamp_seconds=float(stamp), track_state='tracked', joints=js))
    result = seat_skill_evidence(frames, .2, 1.4, {'somersault_quarters':0})
    assert result['name'] == name
    assert result['twist_count'] == (.5 if twist else 0)
    assert result['status'] == 'needs_review'
    assert seat_skill_evidence(frames, .2, 1.4, {'somersault_quarters':4})['name'] is None
    assert seat_skill_evidence(frames, .2, 1.4, {'somersault_quarters':None})['name'] is None
    assert seat_skill_evidence(frames, .2, 1.4, {'somersault_quarters':0}, clipped=True)['name'] is None
    for f in frames:
        f['joints']['nose']['status'] = 'missing'
    assert seat_skill_evidence(frames, .2, 1.4, {'somersault_quarters':0})['name'] is None


def test_contact_position_does_not_call_airborne_tuck_a_seat():
    from trampoline_pose.skill_identification import contact_position
    frames = bouncing_export(60)['frames']
    for f in frames:
        for side in ('left', 'right'):
            hip = f['joints'][side + '_hip']['final']
            f['joints'][side + '_knee'] = joint(hip['x'] + 30, hip['y'] - 30)
            f['joints'][side + '_ankle'] = joint(hip['x'] + 25, hip['y'] + 15)
    assert contact_position(frames, .7) == 'unknown'
