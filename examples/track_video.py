"""Run with VIDEO SEED_FRAME X1 Y1 X2 Y2; coordinates refer to display orientation."""
import sys
from trampoline_pose import PoseTracker,TrackerConfig,AthleteSelection
video,seed,*box=sys.argv[1:]
result=PoseTracker(TrackerConfig.from_yaml('configs/default.yaml')).process_video(
    video,AthleteSelection(int(seed),tuple(map(float,box))))
result.save_json('outputs/joints.json')
result.save_csv('outputs/joints.csv')
result.save_video('outputs/overlay.mp4')
