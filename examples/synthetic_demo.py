"""Exercise real tracking/export code with known synthetic poses, without a GPU.

The injected backend returns generated coordinates. This demonstrates software
behaviour and output formats, not computer-vision accuracy on a real athlete.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from trampoline_pose import AthleteSelection, PoseTracker, TrackerConfig
from trampoline_pose.geometry import crop_geometry
from trampoline_pose.output import validate
from trampoline_pose.skeleton import EDGES
from trampoline_pose.video import VideoWriter, decode


BOX = (150., 100., 490., 410.)
FRAME_COUNT = 80
FPS = 20


def pose(index):
    points = np.array([
        [0, -75], [-6, -80], [6, -80], [-13, -74], [13, -74],
        [-24, -48], [24, -48], [-40, -13], [40, -13],
        [-49, 18], [49, 18], [-17, 10], [17, 10],
        [-20, 52], [20, 52], [-22, 94], [22, 94],
    ], dtype=float)
    phase = index / FRAME_COUNT
    angle = 2 * np.pi * phase
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return points @ rotation.T + [320, 260 - 40 * np.sin(np.pi * phase)]


class SyntheticBackend:
    provenance = {'demo': 'generated COCO-17 coordinates; no model inference'}

    def __init__(self):
        self.frame = 0

    def detect(self, image):
        return np.array([BOX]), np.ones(1)

    def estimate(self, image, box):
        points = pose(self.frame)
        self.frame += 1
        return points, np.ones(17), crop_geometry(box)

    def metrics(self):
        return {'peak_gpu_allocated_bytes': None, 'peak_gpu_reserved_bytes': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/synthetic-demo'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / 'source.mp4'
    with VideoWriter(source, 640, 480, str(FPS)) as writer:
        for index in range(FRAME_COUNT):
            image = np.full((480, 640, 3), (30, 23, 18), dtype=np.uint8)
            points = np.rint(pose(index)).astype(int)
            cv2.line(image, (130, 416), (510, 416), (135, 113, 80), 4)
            for a, b in EDGES:
                cv2.line(image, tuple(points[a]), tuple(points[b]), (210, 195, 180), 7, cv2.LINE_AA)
            cv2.putText(image, 'SYNTHETIC FIXTURE / NO AI INFERENCE', (52, 454),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 226, 185), 1, cv2.LINE_AA)
            writer.write(image, index / FPS)

    tracker = PoseTracker(TrackerConfig(device='cpu'), backend=SyntheticBackend())
    result = tracker.process_video(source, AthleteSelection(0, BOX))
    validate(result.data)
    assert len(result.data['frames']) == FRAME_COUNT
    assert all(frame['track_state'] == 'tracked' for frame in result.data['frames'])
    result.save_json(args.output_dir / 'joints.json')
    result.save_csv(args.output_dir / 'joints.csv')
    overlay = args.output_dir / 'overlay.mp4'
    result.save_video(overlay)
    rendered = list(decode(overlay))
    np.testing.assert_allclose(
        [timing['timestamp_seconds'] for timing, _ in rendered],
        np.arange(FRAME_COUNT) / FPS, atol=1 / 90000,
    )
    images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
              for _, image in rendered[::2]]
    images[0].save(args.output_dir / 'preview.gif', save_all=True,
                   append_images=images[1:], duration=100, loop=0)
    print(f'Synthetic demo complete: {FRAME_COUNT} frames; timestamps verified.')
    print(f'Outputs: {args.output_dir}')
    print('Generated coordinates only; no GPU or pretrained models were used.')


if __name__ == '__main__':
    main()
