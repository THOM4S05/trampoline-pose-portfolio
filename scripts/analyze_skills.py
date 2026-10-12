"""Compatibility entry point for conservative skill evidence.

Accepts a pose JSON export or its accompanying joints CSV. Twist is never
inferred from 2D left/right shoulder ordering.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trampoline_pose.skill_identification import analyze, write_csv


def load_pose(path):
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() != ".csv":
        raise ValueError("Input must be a pose .json or joints .csv export")
    frames = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            index = int(row["frame_index"])
            frame = frames.setdefault(index, dict(frame_index=index,
                timestamp_seconds=float(row["timestamp_seconds"]),
                track_state=row["track_state"], joints={}))
            x, y = row["final_x"], row["final_y"]
            frame["joints"][row["joint"]] = dict(
                final=dict(x=float(x) if x else None, y=float(y) if y else None),
                status=row["status"], visibility=row["visibility"],
                flags=[flag for flag in row["flags"].split(";") if flag])
    return dict(schema_version="1.0.0", source={},
                frames=[frames[index] for index in sorted(frames)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="Save full evidence as .json or flattened evidence as .csv")
    args = parser.parse_args(argv)
    try:
        result = analyze(load_pose(args.input))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() == ".json":
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            elif args.output.suffix.lower() == ".csv":
                rows = [dict(skill_index=s["skill_index"], start_frame=s["start_frame"],
                             end_frame=s["end_frame"], start_seconds=s["start_seconds"],
                             end_seconds=s["end_seconds"],
                             body_position=s["execution"]["body_position"],
                             projected_somersault_count=s["rotation"]["somersault_count"],
                             twist=s["twist"], review_status=s["review_status"])
                        for s in result["skills"]]
                write_csv(args.output, rows)
            else:
                raise ValueError("--output must end in .json or .csv")
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
