from dataclasses import dataclass, asdict
from pathlib import Path
import math
import yaml

def validate_box(box):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('Box must contain four finite display-image pixel coordinates')
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Box must have positive width and height (x1 y1 x2 y2)')

@dataclass(frozen=True)
class AthleteSelection:
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    flight_region_xyxy: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.frame_index < 0:
            raise ValueError('Seed frame must be nonnegative')
        validate_box(self.bbox_xyxy)
        object.__setattr__(self,'bbox_xyxy',tuple(float(v) for v in self.bbox_xyxy))
        if self.flight_region_xyxy is not None:
            validate_box(self.flight_region_xyxy)
            object.__setattr__(self,'flight_region_xyxy',tuple(float(v) for v in self.flight_region_xyxy))

@dataclass(frozen=True)
class TrackerConfig:
    model_directory: str = 'models'
    device: str = 'cuda:0'
    detection_score: float = .3
    detector_rotations: tuple[int,...] = (0,)
    joint_score: float = .3
    association_distance: float = 1.5
    association_margin: float = .12
    max_association_cost: float = .65
    minimum_seed_scale: float = .4
    maximum_seed_scale: float = 2.5
    reacquisition_distance: float = 1.5
    reacquisition_cost: float = .9
    automatic_reacquisition: bool = True
    late_reacquisition_appearance: float = .35
    late_reacquisition_confirmations: int = 3
    uncertain_ms: float = 200.
    reacquisition_ms: float = 500.
    crop_padding: float = 1.25
    temporal_refinement: bool = False
    max_gap_ms: float = 100.
    max_gap_frames: int = 3

    def __post_init__(self):
        rotations=tuple(int(v) for v in self.detector_rotations)
        if not rotations or any(v not in (0,90,180,270) for v in rotations) or len(set(rotations))!=len(rotations):
            raise ValueError('Detector rotations must be unique values from 0,90,180,270')
        object.__setattr__(self,'detector_rotations',rotations)
        if self.device not in ('cpu', 'cuda', 'cuda:0'):
            raise ValueError('Device must be cpu, cuda, or cuda:0')
        if not 0 <= self.detection_score <= 1 or not 0 <= self.joint_score <= 1:
            raise ValueError('Scores must be in [0,1]')
        if self.crop_padding < 1 or self.association_distance <= 0 or self.reacquisition_distance <= 0:
            raise ValueError('Invalid crop padding or association distance')
        if self.max_association_cost<=0:
            raise ValueError('Maximum association cost must be positive')
        if not math.isfinite(self.reacquisition_cost) or self.reacquisition_cost<=0:
            raise ValueError('Reacquisition cost must be finite and positive')
        if not 0<=self.late_reacquisition_appearance<=1:
            raise ValueError('Late reacquisition appearance distance must be in [0,1]')
        if (isinstance(self.late_reacquisition_confirmations,bool) or
                not isinstance(self.late_reacquisition_confirmations,int) or
                self.late_reacquisition_confirmations<3):
            raise ValueError('Late reacquisition requires at least three detections')
        if not isinstance(self.automatic_reacquisition,bool):
            raise ValueError('Automatic reacquisition must be a boolean')
        if not 0 < self.minimum_seed_scale < self.maximum_seed_scale:
            raise ValueError('Invalid seed-relative detection scale limits')
        if min(self.uncertain_ms, self.reacquisition_ms, self.max_gap_ms, self.max_gap_frames) < 0:
            raise ValueError('Time and frame limits must be nonnegative')

    @classmethod
    def from_yaml(cls, filename):
        path = Path(filename).resolve()
        values = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if 'model_directory' in values:
            values['model_directory'] = str((path.parent/values['model_directory']).resolve())
        return cls(**values)

    def to_dict(self):
        return asdict(self)
