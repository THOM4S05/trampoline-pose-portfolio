# Experimental skill analysis

The analysis layer reads exported joint JSON or CSV without modifying pose
tracking. It estimates contact-to-contact spans, projected torso rotation and
supported body-shape windows. It does not establish intended skill identity.

```powershell
.\.venv\Scripts\python.exe scripts/analyze_skills.py outputs/joints.json --output outputs/skills.csv
.\.venv\Scripts\python.exe scripts/identify_skills.py outputs/joints.json --output-dir outputs/analysis
```

## Method

Contact proposals use vertical motion, low-height evidence and posture.
Feet-contact and horizontal torso-rebound proxies are distinguished. Brief
outages may bracket a possible contact, but do not reconstruct rotation through
an unsupported internal flight interval.

Rotation uses supported torso-angle segments and upright facing evidence.
Quarter-turn counts require sufficient continuous evidence and proximity to a
quarter multiple. Clipped flights, large jumps, conflicting facing evidence and
tracking outages can leave the count unknown.

Shape evidence compares supported hip and knee angles over sustained windows.
Elevated central-flight windows help avoid treating bed compression as a tuck.
Missing joints can leave shape or its tightest time unverified.

Twist remains deferred: shoulder crossings in a 2D pose sequence do not measure
3D twist. Displayed best guesses and user references remain separate from the
measured evidence. References are used for comparison after analysis, not as
inputs to predicted boundaries or counts.

## Review workflow

Use the portal to run analysis after tracking. Jump from a skill card to its
start time, inspect the review video and any shape overlay, and check uncertain
contact boundaries. Outputs include JSON/CSV and review reasons. Rerunning
analysis uses saved joints without rerunning the pose models.

Synthetic tests exercise rate invariance, gaps, direction, rotation, clipping,
shape evidence and reference separation. Development recordings informed the
heuristics, so they are not independent evaluation data. Real-routine accuracy
requires a held-out labelled dataset; see [validation](VALIDATION.md).
