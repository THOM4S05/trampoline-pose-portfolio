"""Make a source-PTS-checked sheet of automatically selected shape evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.pose.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    with args.video.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != data["source"]["sha256"] or digest != results["source_sha256"]:
        raise ValueError("Source video/pose/results hashes disagree")
    times = np.array([f["timestamp_seconds"] for f in data["frames"]])
    selected = {}
    for s in results["skills"]:
        stamp = s["execution"].get("shape_timestamp_seconds")
        if stamp is None:
            stamp = (s["visible_start_seconds"] + s["visible_end_seconds"]) / 2
        f = data["frames"][int(np.argmin(abs(times - stamp)))]
        selected[f["frame_index"]] = (s, f)
    images, verified = [], []
    with av.open(str(args.video)) as container:
        first = None
        for i, frame in enumerate(container.decode(video=0)):
            if first is None:
                first = frame.pts * frame.time_base
            if i not in selected:
                continue
            s, pose = selected[i]
            stamp = float(frame.pts * frame.time_base - first)
            if abs(stamp - pose["timestamp_seconds"]) > 1e-6:
                raise ValueError("Decoded source PTS differs from pose timestamp")
            img = np.ascontiguousarray(np.rot90(frame.to_ndarray(format="bgr24"),
                                              round(data["source"]["rotation_degrees_ccw"] / 90) % 4))
            for j in pose["joints"].values():
                p = j["final"]
                if p["x"] is not None and p["y"] is not None:
                    colour = (0, 255, 0) if j["status"] == "observed" and not j["flags"] else (0, 165, 255)
                    cv2.circle(img, (round(p["x"]), round(p["y"])), 5, colour, -1)
            img = cv2.resize(img, (270, 480))
            panel = np.full((540, 270, 3), 255, np.uint8)
            panel[:480] = img
            text = [f"Skill {s['skill_index']} frame {i} {stamp:.3f}s", s["execution"]["execution_position"],
                    f"hip {s['execution'].get('sustained_hip_angle_degrees', '?')} knee {s['execution'].get('sustained_knee_angle_degrees', '?')}"]
            for k, line in enumerate(text):
                cv2.putText(panel, line, (5, 497 + k * 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            images.append(panel)
            verified.append(dict(skill_index=s["skill_index"], frame_index=i, timestamp_seconds=stamp))
            if len(images) == len(selected):
                break
    if len(images) != len(selected):
        raise ValueError("Not all selected native frames decoded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.concatenate(images, axis=1))
    args.output.with_suffix(".json").write_text(json.dumps(dict(source_sha256=digest, verified_frames=verified), indent=2), encoding="utf-8")
    print(f"Verified {len(images)} source frames; wrote {args.output}")


if __name__ == "__main__":
    main()
