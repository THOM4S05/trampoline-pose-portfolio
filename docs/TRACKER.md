# Tracker setup and technical reference

Offline Python package and CLI, with an optional private browser inbox. Select one athlete with a box in
a display-oriented video frame, then export 17 COCO joint records for every
decoded frame, JSON, optional CSV, and a skeleton review video.

**Browser upload and selection:** the [personal video inbox](VIDEO-INBOX.md)
lets you upload from a Mac, PC or phone, slide through exact frames, draw the
gymnast box and queue processing. Runs locally on the processing PC; Tailscale
Serve provides private remote access after setup. No public hosting is required.

**Verified on this PC:** actual GTX 1070 CUDA execution, RTMDet-tiny detection,
ViTPose-S pose inference, a 60-frame VFR GPU integration test, and full exports
for a private 779-frame trampoline development sample. Accuracy and anatomical left/right
identity need independent labels. See [validation](VALIDATION.md).

Development recordings and raw reports are intentionally excluded from this repository.
Replace all sample paths and selections with your own video.

## Environment

Tested 8 October 2026: Windows 10 Pro 19045, i7-9700K, 16 GB RAM, GTX 1070 with
8 GB VRAM, driver 582.28. Project-local Python 3.11.9; PyTorch 2.0.1+cu118;
torchvision 0.15.2+cu118; MMCV 2.0.1; MMEngine 0.10.7; MMDetection 3.2.0;
MMPose 1.3.2; MMPretrain 1.2.0; NumPy 1.26.4. FP32, batch size 1, 192×256 pose
input with the checkpoint's flip testing enabled. RTMDet uses a 640×640 canvas.

The CUDA runtime comes with PyTorch; no separate CUDA toolkit or driver change
was needed. The installed NVIDIA driver advertises CUDA 13.0 compatibility;
inference actually uses CUDA 11.8. Existing Python installations and PATH are
preserved. The official Python installer registers project Python with the
existing `py` launcher; it does not install a new launcher or create shortcuts.
Dependencies live in `.venv`.

FFmpeg/ffprobe 8.0.1 were already installed. They must be available on PATH.
On another PC without them, install the same build with
`winget install --id Gyan.FFmpeg --exact --version 8.0.1`, then open a fresh
PowerShell. The package uses PyAV for native decoding/encoding and ffprobe for
display/timing metadata. `curl.exe` is required during setup only.

## Reproduce setup

From this folder in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Setup installs Python under `.tools/python311`, makes `.venv`, downloads the
pinned official CUDA wheel (approximately 2.6 GB), verifies its published
SHA256, downloads/verifies the Windows MMCV wheel, installs all exact versions
from `requirements-lock.txt`, installs this package, obtains the official model
assets, and runs the CUDA smoke test and automated tests. Leave several GB free
for wheels and installed libraries. Network is required only for setup.

The CUDA download is cached/resumable in `.tools/wheels`. Small parallel HTTP
ranges work around the slow large transfers observed on this PC; no model
content is modified. If an existing cached wheel fails its SHA256, remove that
specific cached file and rerun setup. Setup intentionally refuses changed weights.

Use an existing Python 3.11 executable instead of installing one with
`-Python 'C:\path\to\python.exe'`. Python 3.12/3.14 are not supported by this lock.
The environment lock is specific to Windows x64/Python 3.11.

`requirements-base.txt` documents the principal compatible versions;
`requirements-support.txt` documents support libraries; `requirements-lock.txt`
pins the actual complete tested environment. It contains no local absolute
wheel/install paths. Model identifiers, upstream URLs, full checkpoint SHA256,
and config hashes are in `models/manifest.json`. Setup snapshots inherited
configuration trees locally; each inference export additionally hashes the tree
and effective configuration. Runtime never silently downloads model assets.

Useful checks:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/gpu-smoke.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/video-smoke.py
```

Reports go under `reports/`. Integration artifacts go under
`outputs/synthetic-smoke/`. The public test image comes from MMPose's COCO test
data; its dataset usage terms apply. Model/framework provenance:
[ViTPose authors](https://github.com/ViTAE-Transformer/ViTPose),
[MMPose compatibility](https://github.com/open-mmlab/mmpose/blob/v1.3.2/docs/en/installation.md),
[PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/),
[Windows MMCV wheels](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html).

## Process a video

Inspect actual metadata and extract a clear seed frame as an image:

```powershell
.\.venv\Scripts\trampoline-pose.exe inspect 'C:\path\gymnast.mp4' --frame 150 --image outputs\seed.jpg
```

Use a tight full-body `x1 y1 x2 y2` athlete box in that extracted display image.
The following numbers are examples; replace them with the selected athlete's
actual coordinates and frame number.

```powershell
.\.venv\Scripts\trampoline-pose.exe track 'C:\path\gymnast.mp4' --config configs\default.yaml --seed-frame 150 --bbox 450 610 590 920 --json outputs\joints.json --csv outputs\joints.csv --overlay outputs\overlay.mp4 --cache outputs\detections.json
.\.venv\Scripts\trampoline-pose.exe validate outputs\joints.json
```

The local development sample is `inputExamples/sample.mp4`. Its encoded
1920×1080 image has a -90° display rotation, producing 1080×1920 coordinates.
Frame 150 selects the airborne male gymnast in a white shirt and black shorts.
Reproduce its exports with:

```powershell
.\.venv\Scripts\trampoline-pose.exe track 'inputExamples/sample.mp4' --config configs\sample.yaml --seed-frame 150 --bbox 472 710 603 1266 --flight-region 50 200 1000 1510 --json outputs\sample\joints.json --csv outputs\sample\joints.csv --overlay outputs\sample\overlay.mp4 --cache outputs\sample\rotated-detections.json
.\.venv\Scripts\python.exe scripts/verify-video-output.py --json outputs\sample\joints.json --overlay outputs\sample\overlay.mp4 --csv outputs\sample\joints.csv --report reports\sample-verification.json --review outputs\sample\review.jpg
```

`configs/sample.yaml` detects people at 0/90/180/270° and maps boxes back before
association. It was developed on this routine: the ordinary upright detector
missed horizontal poses and selected spectators. Four orientations cost more
time and sometimes produce oversized or competing boxes. The default config
uses only 0°. This experiment rotates detector inputs; pose inference still
uses the original upright display image. No held-out accuracy gain is claimed.
Detection caches include source hash, model/config provenance and timestamps;
association settings can change without rerunning detection. Remove or choose
a new cache path to benchmark a full fresh run.

The final sample reports 754 tracked, 17 uncertain and 8 lost frames, with
94.3% accepted joint-record coverage. Lost spans are frames 338–340 and
577–581. All 779 overlay frame times match the source exactly. Review known
failures in [validation limitations](VALIDATION.md); these are unlabelled
model outputs, not validated anatomical measurements. First fresh detection
and pose processing took 319.2 seconds before the final association fixes;
the final cached-detection run took 25.4 seconds. Both exclude model loading
and export/encoding; the cached number omits detector inference.

Optional `--flight-region X1 Y1 X2 Y2` excludes person detections whose centres
are outside the apparatus **and full aerial flight path**. Keep the whole
gymnast flight visible. `--max-frames 60` restricts a diagnostic run; the seed
must occur within those frames. GPU is the default and fails explicitly if
unavailable. `--device cpu` is an explicit diagnostic mode; CPU performance
has not been benchmarked here.

The default profile now automatically searches for the athlete after an
occlusion, including gaps longer than 500 ms. For the sample rerun seeded at
frame 150 with box `467 702 613 1270`, it recovers at frames 343 and 581 using
the existing upright detection cache: 749 tracked, 22 uncertain, 8 lost frames.
The previous default permanently lost tracking from frame 337. The sample-only
`configs/sample-upright.yaml` remains available, but a separate sample profile
is no longer required to recover this run. Use the default for new videos:

```powershell
.\.venv\Scripts\trampoline-pose.exe track 'inputExamples/sample.mp4' --config configs\default.yaml --seed-frame 150 --bbox 467 702 613 1270 --json outputs\20250405_102511-auto\joints.json --csv outputs\20250405_102511-auto\joints.csv --overlay outputs\20250405_102511-auto\overlay.mp4 --cache outputs\20250405_102511\detections.json
```

`--raw-overlay` displays raw model predictions in magenta. Normal overlays
draw accepted image predictions in green, uncertain tracking in orange,
inferred points as orange rings, and
omit missing joints and edges with missing endpoints. Frame, timestamp, and
tracking state are drawn on the video. Rendering is independent of inference.
Overlays contain no audio and preserve source presentation times to a 1/90000
second time base. The final frame's duration is encoder-derived. Source VFR
timestamps are never fabricated from FPS. Unsupported non-square pixels or
non-right-angle display transforms fail explicitly.

## Import from Python

```python
from trampoline_pose import PoseTracker, TrackerConfig, AthleteSelection

tracker = PoseTracker(TrackerConfig.from_yaml('configs/default.yaml'))
result = tracker.process_video(
    'gymnast.mp4',
    selection=AthleteSelection(150, (450, 610, 590, 920)),
    cache_path='outputs/detections.json',
    # Optional progress(stage, completed, total) and cancel() callbacks.
)
result.save_json('outputs/joints.json')
result.save_csv('outputs/joints.csv')
result.save_video('outputs/overlay.mp4')
```

## Behaviour and output contract

The first streaming pass detects people on every native frame, caches compact
boxes, scores, appearance histograms, and timestamps. Association proceeds
forward and backward from the explicit seed using motion, overlap, scale,
appearance, and competing candidates. A seed-relative size gate prevents
progressive shrinkage towards small spectators; absolute association cost
rejects weak matches even when only one candidate exists. Unobserved velocity
decays during gaps. A second streaming pass runs ViTPose on
the selected source-resolution ROI. OpenMMLab performs the padded affine crop
before resizing; the whole portrait image is not shrunk into the pose input.
Its inverse-mapped full-frame predictions are saved without double-transforming
them. Crop centre, scale, padded bounds, affine matrix, and inverse matrix are
recorded. Rendering streams the video a third time. Full-resolution frames are
never retained as a whole clip; compact records grow with clip length.

Association is deliberately conservative. Short gaps become `uncertain`;
ambiguous candidates are recorded instead of accepted; longer gaps become
`lost`. Automatic reacquisition is enabled by default. Recovery has separate
distance and cost limits (1.5 body scales and 0.9), so the ordinary tracked
association threshold stays at 0.65. Short occlusions need two consecutive
consistent detections. Beyond `reacquisition_ms` (500 ms by default), the
tracker keeps searching near the last athlete box with a frozen motion estimate,
and requires at least three consecutive detections matching both the seed and
recent athlete appearance (Bhattacharyya histogram distance at most 0.35).
Missing or ambiguous detections reset confirmation; pending recovery produces
no skeleton. Search remains bounded by distance, seed-relative scale, and any
supplied flight region. A scene cut is a hard identity boundary. Setting
`automatic_reacquisition: false` restores expiry at the configured time limit.
Background officials can
still confuse this unvalidated baseline; the states and heuristics do not
guarantee recovery or zero wrong-athlete switches on other videos; colour
histograms cannot reliably distinguish similarly dressed people. Large camera
moves, appearance changes, a flight region that excludes the gymnast, or absent
person detections can still prevent recovery. Recovery is tested with simulated
occlusions and this sample; other routines need review. No pose-based crop adaptation, learned
appearance embeddings, or gymnastics-specific detector has been validated.

JSON schema `1.0.0` is published in `src/trampoline_pose/schema.json`. Every
decoded frame includes all 17 COCO joints, anatomical-side labels, source PTS,
relative timestamp, track state, box, transform, raw coordinates/scores, final
coordinates, status, visibility, correction displacement, and diagnostic flags.
Missing coordinates are JSON `null`. Raw values remain available even when
low scores or out-of-frame predictions cause final values to be missing.
`observed` means an accepted image-model prediction, not guaranteed truth or
visibility. Scores are uncalibrated; confidence remains null; visibility is
unknown except out-of-frame predictions. Pelvis/shoulder centres are explicitly
derived and missing if their source joints are unusable. These image landmarks
are not calibrated anatomical joint centres.

No automatic anatomical-side relabelling is enabled. The pretrained model's
left/right assignment may be wrong during inversion, twists or occlusion;
every frame records that it is not independently verified. No calibrated
confidence, left/right swap detector, fixed projected bone-length constraints,
or claimed fine-tuned model is supplied.

Temporal refinement is **disabled by default**. An experimental cached
short-gap interpolation command exists for development:

```powershell
.\.venv\Scripts\trampoline-pose.exe refine outputs\joints.json --json outputs\experimental.json --max-gap-ms 100 --max-gap-frames 3
```

It uses actual timestamps, requires reliable tracked endpoints, refuses long,
unbounded, or identity-ambiguous gaps, preserves raw predictions, and marks
filled joints `inferred`. It does not smooth trajectories or fix side swaps.
There is no labelled accuracy evidence for it; do not enable it as an accepted
accuracy improvement. Full raw and refined records can be re-exported without
rerunning models. Fine tuning and learned correction remain later milestones.

## Remaining validation

For measured accuracy, obtain independent manual COCO-17 labels with anatomical
side/visibility conventions and reference athlete boxes. Use athlete/routine
disjoint train/validation/test splits; do not tune on the held-out set. Report
coverage and missing/inferred rates alongside median and tail pixel error,
normalised error using `sqrt(reference_bbox_width * reference_bbox_height)`,
and missing-as-failure PCK. Evaluate side swaps, identity switches, upright vs
inverted poses, tucks/pikes, twists, blur, and occlusion separately. No 2%/6%
target, swap rate, or refinement improvement is claimed by this release.

The sample's frame count, rotation, codec, timestamps, CUDA inference, and
sampled overlay alignment have been checked. There are short uncertain/lost
spans and visible errors in difficult poses. Model-score coverage is not
localisation accuracy; hidden and inverted limbs can have convincing but wrong
predictions. No exhaustive identity audit or manual benchmark has been done.
Windows setup has been rerun in this existing environment; a clean second-PC
installation remains unverified.
# Skill and rotation estimates

After exporting a pose JSON or the accompanying joints CSV, you can create a
per-skill summary with:

```powershell
python scripts/analyze_skills.py path\to\pose.json --output skills.csv
```

Use the exported CSV instead by passing its path in place of the JSON file.
The report gives contact-to-contact spans, projected body-shape evidence, and
supported somersault counts. Twist remains `deferred`: shoulder-side crossings
in a 2D pose export do not measure twist. Omit `--output` to print full JSON,
or use a `.json` output path to save it. A `.csv` output path saves a compact
table. All skill labels and contact proxies need video review.
