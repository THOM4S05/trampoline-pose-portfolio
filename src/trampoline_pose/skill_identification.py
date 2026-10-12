"""Conservative 2D skill evidence, independent of reference annotations.

Fixed-camera, predominantly sagittal, feet-bounce baseline. Scores are not
probabilities. Pair centres intentionally do not depend on anatomical side IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Settings:
    step_seconds: float = 0.01
    max_gap_seconds: float = 0.08
    smooth_seconds: float = 0.025
    min_bounce_seconds: float = 0.6
    shape_window_seconds: float = 0.12


def usable(frame, name):
    joint = frame.get("joints", {}).get(name, {})
    p = joint.get("final", {})
    if (frame.get("track_state") != "tracked"
            or joint.get("status") not in ("observed", "corrected")
            or joint.get("visibility") == "out_of_frame"
            or "near_crop_edge" in joint.get("flags", [])):
        return None
    xy = [p.get("x"), p.get("y")]
    if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xy):
        return None
    return np.asarray(xy, dtype=float)


def centre(frame, part):
    a, b = [usable(frame, side + "_" + part) for side in ("left", "right")]
    return (a + b) / 2 if a is not None and b is not None else np.full(2, np.nan)


def interpolate_short(t, values, grid, max_gap):
    """Never bridge long outages or extrapolate missing endpoints."""
    values = np.asarray(values)
    good = np.isfinite(values)
    out = np.full(len(grid), np.nan)
    ids = np.flatnonzero(good)
    if len(ids) < 2:
        return out
    for group in np.split(ids, np.flatnonzero(np.diff(t[ids]) > max_gap) + 1):
        if len(group) < 2:
            continue
        mask = (grid >= t[group[0]]) & (grid <= t[group[-1]])
        out[mask] = np.interp(grid[mask], t[group], values[group])
    return out


def smooth_blocks(values, sigma):
    out = values.copy()
    ids = np.flatnonzero(np.isfinite(values))
    for group in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
        if len(group) > 2:
            out[group] = gaussian_filter1d(values[group], sigma, mode="nearest")
    return out


def joint_angle(a, b, c):
    u, v = a - b, c - b
    lengths = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.degrees(np.arccos(np.clip(np.sum(u * v, axis=1) / lengths, -1, 1)))
    result[lengths < 25] = np.nan
    return result


def rotation_evidence(t, shoulder, hip, facing, max_gap, clipped):
    axis = shoulder - hip
    valid = np.isfinite(axis).all(axis=1) & (np.linalg.norm(axis, axis=1) > 10)
    ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return dict(somersault_count=None, somersault_quarters=None, direction="unknown",
                    projected_turns=None, coverage=float(valid.mean()) if valid.size else 0., reasons=["insufficient_torso"])
    th = np.arctan2(axis[ids, 0], -axis[ids, 1])
    raw_delta = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
    dt = np.diff(t[ids])
    continuous = dt <= max_gap + 1e-9
    # Estimate velocity over elapsed-time neighbourhoods, not a single noisy
    # frame difference. Never smooth across a tracking outage or hide big jumps.
    filtered = th.copy()
    for group in np.split(np.arange(len(ids)), np.flatnonzero(~continuous) + 1):
        unwrapped = np.unwrap(th[group])
        stamps = t[ids[group]]
        for local, index in enumerate(group):
            nearby = np.abs(stamps - stamps[local]) <= .02 + 1e-9
            filtered[index] = np.mean(unwrapped[nearby])
        filtered[group[0]], filtered[group[-1]] = unwrapped[0], unwrapped[-1]
    delta = (np.diff(filtered) + np.pi) % (2 * np.pi) - np.pi
    # Smoothing endpoints can concentrate several frames of motion into the
    # last derivative before an outage. Check speed over a supported 60 ms
    # neighbourhood within each continuous block, retaining the raw jump veto.
    velocity = np.abs(delta) / dt
    for group in np.split(np.arange(len(ids)), np.flatnonzero(~continuous) + 1):
        transitions = group[:-1]
        for i in transitions:
            midpoints = (t[ids[transitions]] + t[ids[transitions + 1]]) / 2
            nearby = transitions[np.abs(midpoints - (t[ids[i]] + t[ids[i + 1]]) / 2) <= .03 + 1e-9]
            velocity[i] = np.sum(np.abs(delta[nearby])) / np.sum(dt[nearby])
    plausible = (velocity < 20) & (np.abs(raw_delta) / dt < 40)
    accepted = continuous & plausible
    # Resolve a short occlusion near inversion only when observed motion on
    # both sides selects a unique angular arc. This estimates angle evidence;
    # it does not create joint observations or infer turns hidden in long gaps.
    reconstructed = []
    lengths = np.linalg.norm(axis[ids], axis=1)
    for i in np.flatnonzero(~continuous):
        if dt[i] > .35 or i < 3 or i + 4 >= len(ids):
            continue
        before = np.flatnonzero((np.arange(len(dt)) < i) & (t[ids[:-1]] >= t[ids[i]] - .15))
        after = np.flatnonzero((np.arange(len(dt)) > i) & (t[ids[1:]] <= t[ids[i + 1]] + .15))
        if min(len(before), len(after)) < 3:
            continue
        neighbours = np.r_[before, after]
        speeds_before, speeds_after = delta[before] / dt[before], delta[after] / dt[after]
        sign = np.sign(np.median(speeds_before))
        near_inversion = min(abs(th[i]), abs(th[i + 1])) > np.pi / 3
        coherent = accepted[neighbours].all() and all(
            np.median(speeds) * sign > .5
            and np.sum(np.maximum(-speeds * sign, 0) * dt[indices]) <= .2
            for speeds, indices in ((speeds_before, before), (speeds_after, after)))
        if not near_inversion or not coherent or max(lengths[i:i+2]) / min(lengths[i:i+2]) > 1.5:
            continue
        arcs = [raw_delta[i] + k * 2 * np.pi for k in (-1, 0, 1)
                if 0 < (raw_delta[i] + k * 2 * np.pi) * sign < 20 * dt[i]]
        if len(arcs) != 1:
            continue
        delta[i] = arcs[0]
        accepted[i] = True
        reconstructed.append(dict(start_seconds=round(float(t[ids[i]]), 6),
                                  end_seconds=round(float(t[ids[i + 1]]), 6),
                                  estimated_turns=round(float(arcs[0] / (2 * np.pi)), 4)))
    total = float(np.sum(delta[accepted]) / (2 * np.pi))
    reasons = []
    if clipped:
        reasons.append("clipped_skill")
    if not accepted.all() or (valid.mean() < .8 and not reconstructed):
        reasons.append("torso_gaps_or_jumps")
    if t[ids[0]] - t[0] > max_gap or t[-1] - t[ids[-1]] > max_gap:
        reasons.append("missing_rotation_endpoints")
    # Back/front/seat landings can end at a quarter turn, rather than upright.
    quarters = int(round(abs(total) * 4))
    n = quarters / 4
    if abs(abs(total) - n) > .10:
        reasons.append("non_quarter_projected_rotation")
    # A large amount of cancelling angle motion can hide tracking corruption.
    if np.sum(np.abs(delta[accepted])) / (2 * np.pi) > abs(total) + .5:
        reasons.append("excess_reversing_motion")
    direction = "none" if n == 0 else ("backward" if total * facing < 0 else "forward") if facing else "unknown"
    if n and not facing:
        reasons.append("facing_unresolved")
    count = None if any(r != "facing_unresolved" for r in reasons) else n
    if reconstructed:
        reasons.append("inversion_gap_arc_estimated_needs_review")
    return dict(somersault_count=count, candidate_somersault_count=n, candidate_direction=direction,
                somersault_quarters=None if count is None else quarters,
                candidate_somersault_quarters=quarters,
                direction=direction if count is not None else "unknown", projected_turns=round(total, 4),
                coverage=round(float(valid.mean()), 4), rejected_transition_count=int((~accepted).sum()),
                max_torso_gap_seconds=round(float(dt.max()), 6),
                reconstructed_inversion_gaps=reconstructed, reasons=reasons)


def horizontal_rebound_gaps(t, shoulder, hip, max_gap, include_upright=False):
    """Bracket a possible body contact across a short tracking outage.

    Require horizontal torso poses near the lower observed hip-height envelope,
    a reliable descent before the gap and ascent afterwards. This is evidence of
    a rebound, not independently verified first bed contact.
    """
    axis = shoulder - hip
    valid = np.isfinite(axis).all(axis=1) & np.isfinite(hip).all(axis=1)
    ids = np.flatnonzero(valid)
    if len(ids) < 10:
        return []
    candidates = []
    for a, b in zip(ids, ids[1:]):
        if not max_gap < t[b] - t[a] <= .65:
            continue
        lengths = np.linalg.norm(axis[[a, b]], axis=1)
        nearby = ids[(t[ids] >= t[a] - 3) & (t[ids] <= t[b] + 3)]
        low_envelope = np.percentile(hip[nearby, 1], 80)
        horizontal = np.all(np.abs(axis[[a, b], 0]) / np.maximum(lengths, 1) >= .8)
        low_upright = include_upright and np.max(hip[[a, b], 1]) >= low_envelope - lengths.mean() * .5
        if np.any(lengths < 10) or not (horizontal or low_upright):
            continue
        # Horizontal torso contacts place the hips higher than feet landings.
        required_height = low_envelope - lengths.mean() * (1.25 if horizontal else .5)
        height = np.min(hip[[a, b], 1]) if horizontal else np.max(hip[[a, b], 1])
        if height < required_height or abs(hip[a, 1] - hip[b, 1]) > lengths.mean() * (.8 if horizontal else 2):
            continue
        before = ids[(t[ids] >= t[a] - .12) & (ids <= a)]
        after = ids[(ids >= b) & (t[ids] <= t[b] + .12)]
        if min(len(before), len(after)) < 4:
            continue
        descent = np.polyfit(t[before] - t[a], hip[before, 1], 1)[0]
        ascent = np.polyfit(t[after] - t[b], hip[after, 1], 1)[0]
        if descent > lengths.mean() and ascent < -lengths.mean():
            candidates.append((int(a), int(b)))
    return candidates


def shape_evidence(t, shoulder, hip, knee, ankle, window):
    ha, ka = joint_angle(shoulder, hip, knee), joint_angle(hip, knee, ankle)
    axis = shoulder - hip
    tilt = np.abs(np.arctan2(axis[:, 0], -axis[:, 1]))
    # Upright tuck/pike jumps need shape evidence too. Restrict their windows
    # to the elevated central flight, avoiding knee flexion at bed contact.
    finite_y = hip[np.isfinite(hip[:, 1]), 1]
    central = (t >= t[0] + (t[-1] - t[0]) * .15) & (t <= t[-1] - (t[-1] - t[0]) * .15)
    elevated = np.zeros(len(t), dtype=bool)
    if len(finite_y):
        extent = np.ptp(finite_y)
        elevated = hip[:, 1] <= finite_y.min() + extent * .6 if extent >= 20 else np.isfinite(hip[:, 1])
    active = ((tilt > np.pi / 4) | (central & elevated)) & np.isfinite(ha) & np.isfinite(ka)
    candidates = []
    # Fully supported sustained windows; avoid classifying a single noisy frame.
    for i in np.flatnonzero(active):
        mask = np.abs(t - t[i]) <= window / 2 + 1e-9
        if t[i] - t[0] < window / 2 or t[-1] - t[i] < window / 2:
            continue
        if np.mean(active[mask]) >= .8:
            candidates.append((float(np.nanmedian(ha[mask & active])),
                               float(np.nanmedian(ka[mask & active])), int(i)))
    if not candidates:
        valid = np.isfinite(ha) & np.isfinite(ka)
        if valid.mean() >= .8 and np.nanmedian(ha) >= 150 and np.nanmedian(ka) >= 150:
            h, k, ix = float(np.nanmedian(ha)), float(np.nanmedian(ka)), None
        else:
            return dict(body_position="unknown", execution_position="unknown", reasons=["insufficient_shape_evidence"])
    else:
        h, k, ix = min(candidates)
    if h < 110 and k >= 150:
        body, execution = "pike", "pike_like"
    elif h < 110 and k < 130:
        body, execution = "tuck", "tuck_like"
    elif h >= 150 and k >= 150:
        body, execution = "straight", "straight_like"
    else:
        body, execution = "unknown", "flexed_or_open_shape"
    result = dict(body_position=body, execution_position=execution,
                sustained_hip_angle_degrees=round(h, 1), sustained_knee_angle_degrees=round(k, 1),
                shape_timestamp_seconds=None if ix is None else round(float(t[ix]), 6),
                bent_knees_evidence=k < 160, hip_flexion_evidence=h < 150,
                usable_angle_fraction=round(float(np.mean(np.isfinite(ha) & np.isfinite(ka))), 4),
                reasons=["projected_angles_do_not_establish_intended_skill"])
    # A stricter sustained window can miss a tighter pose hidden by occlusion.
    # Offer a review image at the bracket midpoint, never fabricated angles.
    ids = np.flatnonzero(active)
    gaps = []
    for a, b in zip(ids, ids[1:]):
        if window < t[b] - t[a] <= .35 and np.mean(ha[[a, b]]) < h - 10:
            gaps.append((float(np.mean(ha[[a, b]])), int(a), int(b)))
    if gaps:
        _, a, b = min(gaps)
        result['tightest_shape_review_interval_seconds'] = [round(float(t[a]), 6), round(float(t[b]), 6)]
        stamp = (t[a] + t[b]) / 2
        method = 'occluded_shape_review_midpoint'
        angles = np.arctan2(axis[[a, b], 0], -axis[[a, b], 1])
        arc = (angles[1] - angles[0] + np.pi) % (2 * np.pi) - np.pi
        # For a rotating folded shape, propose the inversion crossing inside
        # the outage as a review frame. Its time/angles are not measurements.
        if angles[0] * angles[1] < 0 and abs(angles[0]) > np.pi / 3:
            target = np.pi if arc > 0 else -np.pi
            fraction = (target - angles[0]) / arc if arc else -1
            if 0 < fraction < 1:
                stamp = t[a] + fraction * (t[b] - t[a])
                method = 'occluded_shape_review_inversion'
        result['review_shape_timestamp_seconds'] = round(float(stamp), 6)
        result['review_shape_selection_method'] = method
        result['reasons'].append('tighter_shape_may_be_hidden_by_missing_joints')
    return result


def contact_position(frames, stamp, radius=.15):
    """Estimate feet/seat support from several observed poses near a contact.

    Ratios use torso length so image scale and frame rate do not set the rule.
    Missing limbs, horizontal torsos and ambiguous bent-leg poses abstain.
    """
    votes = []
    for frame in frames:
        if abs(frame['timestamp_seconds'] - stamp) > radius:
            continue
        shoulder, hip, knee, ankle = [centre(frame, p) for p in ('shoulder', 'hip', 'knee', 'ankle')]
        length = np.linalg.norm(shoulder - hip)
        if not np.isfinite([*shoulder, *hip, *knee, *ankle]).all() or length < 10:
            continue
        if (hip[1] - shoulder[1]) / length < .65:
            votes.append('unknown')
        elif abs(knee[1] - hip[1]) / length < .4 and abs(ankle[1] - hip[1]) / length < .45 and abs(ankle[0] - hip[0]) / length > .65:
            votes.append('seat')
        elif (knee[1] - hip[1]) / length > .4 and (ankle[1] - hip[1]) / length > .9:
            votes.append('feet')
        else:
            votes.append('unknown')
    if len(votes) < 3:
        return 'unknown'
    position, count = Counter(votes).most_common(1)[0]
    return position if count / len(votes) >= .75 else 'unknown'


def seat_skill_evidence(frames, start, end, rotation, clipped=False):
    """Conservative, reviewable seat-transition candidates; no reference labels.

    A facing reversal supports a half-twist candidate, not a 3D twist count.
    More than one sustained reversal or missing endpoint facing abstains.
    """
    takeoff = contact_position(frames, start)
    landing = contact_position(frames, end)
    result = dict(takeoff_position=takeoff, landing_position=landing,
                  name=None, twist_count=None, status='needs_review',
                  method='contact_pose_and_sustained_facing', reasons=[])
    if clipped or (takeoff, landing) not in [('feet', 'seat'), ('seat', 'feet')]:
        return result
    quarters = rotation.get('somersault_quarters')
    if quarters != 0:
        # A candidate with missing tracking must not override measured rotation.
        result['reasons'].append('rotation_not_supported')
        return result
    samples = []
    for frame in frames:
        t = frame['timestamp_seconds']
        if not start - .15 <= t <= end + .15:
            continue
        sh, hip, ear = [centre(frame, p) for p in ('shoulder', 'hip', 'ear')]
        nose = usable(frame, 'nose')
        length = np.linalg.norm(sh - hip)
        if nose is None or not np.isfinite([*sh, *hip, *ear]).all() or length < 10 or (hip[1] - sh[1]) / length < .65:
            continue
        offset = (nose[0] - ear[0]) / length
        if abs(offset) >= .06:
            samples.append((t, 1 if offset > 0 else -1))
    def facing_at(stamp):
        votes = [v for t, v in samples if abs(t - stamp) <= .15]
        return (1 if sum(votes) > 0 else -1) if len(votes) >= 3 and abs(sum(votes)) / len(votes) >= .8 else None
    first, last = facing_at(start), facing_at(end)
    if first is None or last is None:
        result['reasons'].append('endpoint_facing_unknown')
        return result
    # Require sustained votes in time bins, not a count of noisy frame swaps.
    sequence = []
    for left in np.arange(start - .15, end + .15, .1):
        votes = [v for t, v in samples if left <= t < left + .1]
        if len(votes) >= 3 and abs(sum(votes)) / len(votes) >= .8:
            sign = 1 if sum(votes) > 0 else -1
            if not sequence or sequence[-1] != sign:
                sequence.append(sign)
    if not sequence or sequence[0] != first or sequence[-1] != last or len(sequence) > 2:
        result['reasons'].append('facing_sequence_ambiguous')
        return result
    count = .5 if first != last else 0
    result['twist_count'] = count
    result['name'] = ('Half twist to seat' if count else 'Seat drop') if landing == 'seat' else ('Half twist from seat to feet' if count else 'Return from seat to feet')
    result['reasons'].append('2d_facing_cannot_exclude_hidden_twists')
    return result


def analyze(data, settings=Settings(), facing_override=None):
    if data.get("schema_version") != "1.0.0":
        raise ValueError("Expected pose schema 1.0.0")
    frames = data.get("frames", [])
    if len(frames) < 3:
        raise ValueError("At least three frames required")
    t = np.asarray([f["timestamp_seconds"] for f in frames], dtype=float)
    fi = np.asarray([f["frame_index"] for f in frames])
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or np.any(np.diff(fi) <= 0):
        raise ValueError("Frame indices and finite timestamps must be strictly increasing")
    pts_checked = all("source_pts" in f and "source_time_base" in f for f in frames)
    if pts_checked:
        pts = np.asarray([float(f["source_pts"] * Fraction(f["source_time_base"])) for f in frames])
        if not np.allclose(t - t[0], pts - pts[0], rtol=0, atol=1e-6):
            raise ValueError("Timestamps disagree with source PTS")
    if settings.step_seconds <= 0 or settings.max_gap_seconds <= 0 or settings.min_bounce_seconds <= 0:
        raise ValueError("Timing settings must be positive")
    grid = np.arange(t[0], t[-1] + 1e-9, settings.step_seconds)
    native = {part: np.asarray([centre(f, part) for f in frames])
              for part in ("shoulder", "hip", "knee", "ankle", "ear")}
    points = {part: np.column_stack([interpolate_short(t, p[:, k], grid, settings.max_gap_seconds)
                                    for k in (0, 1)]) for part, p in native.items()}
    hy = smooth_blocks(points["hip"][:, 1], settings.smooth_seconds / settings.step_seconds)
    ay = smooth_blocks(points["ankle"][:, 1], settings.smooth_seconds / settings.step_seconds)
    finite = hy[np.isfinite(hy)]
    if len(finite) < 3 or np.ptp(finite) < 20:
        peaks = []
    else:
        # Missing samples cannot create artificial maxima; split into valid blocks.
        peaks = []
        ids = np.flatnonzero(np.isfinite(hy))
        for group in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
            if len(group) < 3:
                continue
            found, _ = find_peaks(hy[group], distance=max(1, round(settings.min_bounce_seconds / settings.step_seconds)),
                                  prominence=max(20, np.ptp(finite) * .12))
            peaks.extend(group[found].tolist())
    contacts = []
    velocity = np.gradient(ay, grid)
    hip_velocity = np.gradient(hy, grid)
    for peak in peaks:
        axis = points['shoulder'][peak] - points['hip'][peak]
        axis_length = np.linalg.norm(axis)
        horizontal = np.isfinite(axis).all() and axis_length > 10 and abs(axis[0]) / axis_length >= .8
        contact_velocity = hip_velocity if horizontal else velocity
        window = np.flatnonzero((grid >= grid[peak] - .4) & (grid <= grid[peak] - .04)
                                & np.isfinite(contact_velocity))
        if len(window) < 6:
            continue  # require supported deceleration, not just a height trough
        fast = window[int(np.argmax(contact_velocity[window]))]
        after = window[(window > fast) & (contact_velocity[window] < contact_velocity[fast] * .75)]
        if contact_velocity[fast] <= 0 or not len(after):
            continue
        ix = int(after[0])
        native_ix = int(np.argmin(np.abs(t - grid[ix])))
        if contacts and t[native_ix] - contacts[-1]["timestamp_seconds"] < settings.min_bounce_seconds:
            continue
        contacts.append(dict(frame_index=int(fi[native_ix]), timestamp_seconds=float(t[native_ix]),
                             bed_compression_peak_frame=int(fi[np.argmin(abs(t - grid[peak]))]),
                             rotation_start_seconds=float(grid[peak]), rotation_end_seconds=float(grid[peak]),
                             method="horizontal_hip_deceleration_proxy" if horizontal else "descending_ankle_deceleration_proxy",
                             review_interval_seconds=[max(float(t[0]), float(t[native_ix] - .12)),
                                                      min(float(t[-1]), float(t[native_ix] + .04))],
                             review_interval_frames=[int(fi[np.argmin(abs(t - max(t[0], t[native_ix] - .12)))]),
                                                     int(fi[np.argmin(abs(t - min(t[-1], t[native_ix] + .04)))])],
                             status="needs_review", note="First body/sole-bed contact precedes this proxy; interval is heuristic, not calibrated."))
    # When feet are occluded, require a supported low-body rebound rather than
    # an ankle observation. Short bridging is used only to locate candidates;
    # it never supplies rotation or shape evidence.
    candidate_y = smooth_blocks(interpolate_short(grid, points['hip'][:, 1], grid, .25), settings.smooth_seconds / settings.step_seconds)
    valid_candidates = np.flatnonzero(np.isfinite(candidate_y))
    for group in np.split(valid_candidates, np.flatnonzero(np.diff(valid_candidates) > 1) + 1):
        if len(group) < 3:
            continue
        found, _ = find_peaks(candidate_y[group], distance=max(1, round(settings.min_bounce_seconds / settings.step_seconds)), prominence=max(20, np.ptp(finite) * .12) if len(finite) else 20)
        for peak in group[found]:
            stamp = float(grid[peak])
            if any(abs(c['timestamp_seconds'] - stamp) < settings.min_bounce_seconds for c in contacts):
                continue
            axis = points['shoulder'][peak] - points['hip'][peak]
            if not np.isfinite(axis).all() or np.linalg.norm(axis) < 10:
                continue
            local_heights = hy[(grid >= stamp - 3) & (grid <= stamp + 3) & np.isfinite(hy)]
            if candidate_y[peak] < np.percentile(local_heights, 80):
                continue
            before = np.flatnonzero((grid >= stamp - .18) & (grid <= stamp - .04) & np.isfinite(hy))
            after = np.flatnonzero((grid >= stamp + .04) & (grid <= stamp + .18) & np.isfinite(hy))
            if min(len(before), len(after)) < 4:
                continue
            descent = np.polyfit(grid[before] - stamp, hy[before], 1)[0]
            ascent = np.polyfit(grid[after] - stamp, hy[after], 1)[0]
            if descent <= np.linalg.norm(axis) or ascent >= -np.linalg.norm(axis):
                continue
            index = int(np.argmin(abs(t - stamp)))
            contacts.append(dict(frame_index=int(fi[index]), timestamp_seconds=float(t[index]),
                                 method='occluded_feet_body_rebound_proxy', status='needs_review',
                                 review_interval_seconds=[max(float(t[0]), stamp - .18), min(float(t[-1]), stamp + .04)],
                                 note='Feet unavailable: low hip rebound supported by torso observations; compression peak is a late contact proxy.'))
    for a, b in horizontal_rebound_gaps(grid, points['shoulder'], points['hip'], settings.max_gap_seconds, include_upright=True):
        stamp = float((grid[a] + grid[b]) / 2)
        if any(abs(c['timestamp_seconds'] - stamp) < settings.min_bounce_seconds for c in contacts):
            continue
        native_ix = int(np.argmin(abs(t - stamp)))
        contacts.append(dict(frame_index=int(fi[native_ix]), timestamp_seconds=float(t[native_ix]),
                             method='body_rebound_tracking_gap_bracket', status='needs_review',
                             review_interval_seconds=[float(grid[a]), float(grid[b])],
                             review_interval_frames=[int(fi[np.argmin(abs(t - grid[a]))]), int(fi[np.argmin(abs(t - grid[b]))])],
                             rotation_end_seconds=float(grid[a]), rotation_start_seconds=float(grid[b]),
                             note='Possible body contact during tracking outage; midpoint is a proxy. Reliable landing/departure poses bound the rotation windows.'))
    contacts.sort(key=lambda c: c['timestamp_seconds'])
    # Estimate sagittal facing from nose relative to ear centre when upright.
    face_votes = []
    timed_face_votes = []
    for i, f in enumerate(frames):
        sh, h, ea = [native[p][i] for p in ("shoulder", "hip", "ear")]
        no = usable(f, "nose")
        if no is None or not np.isfinite([*sh, *h, *ea]).all():
            continue
        axis = sh - h
        if abs(math.atan2(axis[0], -axis[1])) < .35 and abs(no[0] - ea[0]) > np.linalg.norm(axis) * .06:
            vote = 1 if no[0] > ea[0] else -1
            face_votes.append(vote)
            timed_face_votes.append((float(t[i]), vote))
    agreement = abs(sum(face_votes)) / len(face_votes) if face_votes else 0
    facing = (1 if sum(face_votes) > 0 else -1) if len(face_votes) >= 10 and agreement >= .8 else 0
    if facing_override:
        facing = 1 if facing_override == "right" else -1
    skills = []
    edges = [None, *contacts, None] if contacts else []
    for a, b in zip(edges, edges[1:]):
        start = t[0] if a is None else a["timestamp_seconds"]
        end = t[-1] if b is None else b["timestamp_seconds"]
        mask = (grid >= start) & (grid <= end)
        if mask.sum() < 3:
            continue
        args = [points[p][mask] for p in ("shoulder", "hip", "knee", "ankle")]
        rotation_start = a.get('rotation_start_seconds', start) if a else start
        rotation_end = b.get('rotation_end_seconds', end) if b else end
        rotation_mask = (grid >= rotation_start) & (grid <= rotation_end)
        local_facing = facing
        facing_method = 'global_upright_votes'
        if not facing_override:
            recent = [(stamp, vote) for stamp, vote in timed_face_votes if start - .6 <= stamp <= start + .15]
            if len(recent) < 3:
                recent = [(stamp, vote) for stamp, vote in timed_face_votes if start - 2 <= stamp <= start + .15][-15:]
            votes = [vote for _, vote in recent]
            if len(votes) >= 3 and abs(sum(votes)) / len(votes) >= .8:
                local_facing = 1 if sum(votes) > 0 else -1
                facing_method = 'recent_upright_votes_at_takeoff'
            elif votes:
                local_facing = 0
                facing_method = 'conflicting_recent_facing_needs_review'
        rot = rotation_evidence(grid[rotation_mask], points['shoulder'][rotation_mask], points['hip'][rotation_mask], local_facing, settings.max_gap_seconds,
                                a is None or b is None)
        rot['facing_method'] = 'user_override' if facing_override else facing_method
        rot['window_start_seconds'] = float(rotation_start)
        rot['window_end_seconds'] = float(rotation_end)
        rot['contact_outage_excluded'] = bool(rotation_start != start or rotation_end != end)
        shape = shape_evidence(grid[mask], *args, settings.shape_window_seconds)
        selected = [f for f in frames if start <= f["timestamp_seconds"] <= end]
        seat = seat_skill_evidence(frames, rotation_start, rotation_end, rot, a is None or b is None)
        skills.append(dict(skill_index=len(skills) + 1,
                           start_frame=None if a is None else a["frame_index"],
                           end_frame=None if b is None else b["frame_index"],
                           start_seconds=None if a is None else start,
                           end_seconds=None if b is None else end,
                           visible_start_seconds=float(start), visible_end_seconds=float(end),
                           start_clipped=a is None, end_clipped=b is None,
                           span_kind="partial_clip_fragment" if a is None or b is None else "contact_to_contact",
                           supplied_skill_identity=None, predicted_skill_identity="unknown",
                           rotation=rot, execution=shape, twist="deferred", takeoff_position=seat["takeoff_position"],
                           landing_position=seat["landing_position"], seat_transition=seat, review_status="needs_review",
                           quality=dict(track_states=dict(Counter(f["track_state"] for f in selected)),
                                        joint_statuses=dict(Counter(j["status"] for f in selected for j in f["joints"].values())),
                                        joint_flags=dict(Counter(flag for f in selected for j in f["joints"].values() for flag in j["flags"])),
                                        side_policy="pair centres invariant to anatomical left/right swaps; local misassociation remains possible")))
    return dict(result_schema_version="0.1.0", source_sha256=data.get("source", {}).get("sha256"),
                timing=dict(source_pts_checked=pts_checked, first_timestamp_seconds=float(t[0]),
                            last_timestamp_seconds=float(t[-1]),
                            median_frame_period_seconds=float(np.median(np.diff(t))),
                            min_frame_period_seconds=float(np.min(np.diff(t))),
                            max_frame_period_seconds=float(np.max(np.diff(t)))),
                settings=asdict(settings), boundary_convention="contact-to-contact proxies; inclusive shared contacts",
                facing=dict(screen_direction={1: "right", -1: "left", 0: "unknown"}[facing],
                            vote_count=len(face_votes), vote_agreement=round(agreement, 4),
                            method="user_override" if facing_override else "upright_nose_relative_to_ears"),
                contacts=contacts, skills=skills,
                diagnostic_status="needs_review" if contacts else "no_supported_contacts_needs_review",
                limitations=["Development baseline, not validated on new routines.",
                             "Requires fixed camera and sagittal view; no bed-surface calibration.",
                             "2D torso turns and joint angles are projection evidence, not 3D measurements.",
                             "Boundary uncertainty intervals are heuristic, not confidence intervals.",
                             "Inferred, missing, uncertain-track and crop-edge joints excluded; only short gaps interpolated.",
                             "No twist or definitive intended-skill identification; horizontal rebound brackets are experimental and require review."])


def compare(result, labels_path, boundaries_path=None):
    with Path(labels_path).open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    annotation = None
    if boundaries_path:
        annotation = json.loads(Path(boundaries_path).read_text(encoding="utf-8"))
        if annotation.get("source_sha256") != result["source_sha256"]:
            raise ValueError("Reference boundary source hash does not match pose source")
    intervals = {c["after_skill"]: c["review_interval_frames"] for c in annotation["contacts"]} if annotation else {}
    rows, used, last_match = [], set(), -1
    # Order-preserving temporal overlap alignment permits extra/missing predictions.
    for label in labels:
        end = float(label["end_seconds"]) if label["end_seconds"] else math.inf
        start = float(label["start_seconds"]) if label["start_seconds"] else -math.inf
        candidates = [(max(0, min(end, s["visible_end_seconds"]) - max(start, s["visible_start_seconds"])), i)
                      for i, s in enumerate(result["skills"]) if i > last_match]
        overlap, ix = max(candidates, default=(0, -1))
        s = result["skills"][ix] if overlap > 0 else None
        if s:
            used.add(ix)
            last_match = ix
        index = int(label["skill_index"])
        ref_end = int(label["end_frame_0based"]) if label["end_frame_0based"] else None
        err = s["end_frame"] - ref_end if s and s["end_frame"] is not None and ref_end is not None else None
        ref_start = int(label["start_frame_0based"]) if label["start_frame_0based"] else None
        start_err = s["start_frame"] - ref_start if s and s["start_frame"] is not None and ref_start is not None else None
        interval = intervals.get(index)
        ref_quarters = int(label["somersault_quarters"]) if label["somersault_quarters"] else None
        rotation_eligible = bool(label["start_seconds"] and label["end_seconds"])
        rows.append(dict(reference_skill_index=index, supplied_skill_identity=label["skill_name"],
                         supplied_execution_notes=label["notes"], reference_body_position=label["body_position"],
                         predicted_skill_index=None if not s else s["skill_index"],
                         predicted_end_frame=None if not s else s["end_frame"], reference_end_frame=ref_end,
                         predicted_start_frame=None if not s else s["start_frame"], reference_start_frame=ref_start,
                         start_error_frames=start_err,
                         end_error_frames=err, reference_end_review_interval=interval,
                         within_reference_boundary_tolerance=None if err is None or not interval else interval[0] <= s["end_frame"] <= interval[1],
                         reference_somersault_quarters=ref_quarters,
                         predicted_somersault_quarters=None if not s else s["rotation"]["somersault_quarters"],
                         predicted_direction=None if not s else s["rotation"]["direction"],
                         candidate_somersault_count=None if not s else s["rotation"].get("candidate_somersault_count"),
                         candidate_direction=None if not s else s["rotation"].get("candidate_direction"),
                         rotation_evaluation_eligible=rotation_eligible,
                         rotation_match=None if not rotation_eligible or not s else s["rotation"]["somersault_quarters"] == ref_quarters and s["rotation"]["direction"] == label["somersault_direction"],
                         predicted_body_position=None if not s else s["execution"]["body_position"],
                         execution_position=None if not s else s["execution"]["execution_position"],
                         body_position_agreement=None if not s or s["execution"]["body_position"] == "unknown" else s["execution"]["body_position"] == label["body_position"],
                         shape_comparison_caveat="Reference is intended identity; prediction is observed projected execution.",
                         annotation_status=label["review_status"]))
    errors = [abs(r["end_error_frames"]) for r in rows if r["end_error_frames"] is not None]
    eligible = [r for r in rows if r["rotation_evaluation_eligible"]]
    return dict(rows=rows, summary=dict(reference_skills=len(labels), matched_skills=len(used),
                                      boundary_mean_absolute_error_frames=float(np.mean(errors)) if errors else None,
                                      boundary_within_tolerance=sum(r["within_reference_boundary_tolerance"] is True for r in rows),
                                      boundary_tolerance_evaluable=sum(r["within_reference_boundary_tolerance"] is not None for r in rows),
                                      rotation_eligible=len(eligible), rotation_matches=sum(r["rotation_match"] is True for r in eligible),
                                      rotation_unknown=sum(r["predicted_somersault_quarters"] is None for r in eligible)),
                unmatched_prediction_indices=[s["skill_index"] for i, s in enumerate(result["skills"]) if i not in used],
                reference_provenance=None if annotation is None else annotation["annotation_status"],
                note="Development comparison only. Opening rotation excluded; supplied identity never changes analyzer predictions.")


def write_csv(path, rows):
    if not rows:
        path.write_text("skill_index\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--facing", choices=["left", "right"])
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        result = analyze(data, facing_override=args.facing)
        comparison = compare(result, args.labels, args.boundaries) if args.labels else None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "skills.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        rows = [{k: s[k] for k in ("skill_index", "start_frame", "end_frame", "start_seconds", "end_seconds", "start_clipped", "end_clipped", "supplied_skill_identity", "predicted_skill_identity", "review_status")}
                | {"rotation_" + k if k == "reasons" else k: v for k, v in s["rotation"].items()}
                | {"shape_" + k if k == "reasons" else k: v for k, v in s["execution"].items()} for s in result["skills"]]
        write_csv(args.output_dir / "skills.csv", rows)
        if comparison:
            (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
            write_csv(args.output_dir / "comparison.csv", comparison["rows"])
        print(json.dumps(dict(contacts=len(result["contacts"]), spans=len(result["skills"]),
                              output_dir=str(args.output_dir), facing=result["facing"],
                              comparison_summary=None if comparison is None else comparison["summary"],
                              predictions=[dict(skill=s["skill_index"], end=s["end_frame"],
                                                rotation=s["rotation"]["somersault_count"],
                                                direction=s["rotation"]["direction"], shape=s["execution"]["body_position"])
                                           for s in result["skills"]]), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
