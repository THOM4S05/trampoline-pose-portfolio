# Validation and limits

## Automated checks

On 12 October 2026, the local checkout passed **114 Python tests and 6
JavaScript tests**. Tests use synthetic video and injected model outputs; they
exercise application behaviour without downloading weights or requiring a GPU.
The GitHub Actions workflow runs these checks on Windows with Python 3.11,
Node.js and FFmpeg. Its badge reports the remote workflow's actual status.

Coverage includes native variable-frame-rate timestamps, display rotation,
coordinate transforms, bidirectional association, ambiguity and reacquisition,
scene cuts, missing joints, JSON/schema invariants, cache invalidation,
cancellation, authenticated uploads, request boundaries, restart recovery,
codec fixtures, skill estimates and browser navigation.

## Local GPU development evidence

Earlier local runs used a GTX 1070, Python 3.11, PyTorch 2.0.1 with CUDA 11.8,
RTMDet-tiny and ViTPose-S. A 60-frame synthetic video integration run exercised
actual GPU detection and pose inference. A private 779-frame trampoline sample
was also processed with matching source and overlay presentation times.

One development configuration produced 754 tracked, 17 uncertain and 8 lost
frames, with 94.3% accepted joint-record coverage. **Coverage is not accuracy.**
These are historical development results, not a held-out benchmark or a claim
about every routine. Raw reports, computer paths and athlete footage remain
local; the public synthetic demo is independently reproducible.

## What remains unverified

- Localisation error and anatomical left/right correctness against manual labels.
- Identity-switch rates across athletes, routines and camera positions.
- Reliable skill classification, twist estimation and calibrated confidence.
- A clean installation of the full GPU stack on a second computer.
- Production deployment or public multi-user upload hosting.

Inverted poses, occlusion, blur and similarly dressed spectators can produce
missing or confidently wrong predictions. Skill boundaries and body shapes are
heuristic evidence requiring video review. Twist is deferred. The browser inbox
is designed for local/private-network use.

## Next evaluation steps

Create independently labelled routines with athlete-disjoint evaluation splits.
Report missing predictions alongside median/p95 joint error, PCK, side swaps
and identity switches. Keep development clips separate from held-out evaluation.
Benchmark full fresh inference separately from cached detection runs.
