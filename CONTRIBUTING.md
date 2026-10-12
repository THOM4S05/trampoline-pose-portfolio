# Continuing development

The project is a working prototype. Small commits that describe a behaviour
change, its reason and its verification make progress easy to review.

## Local checks

Set up Python 3.11 with `requirements-dev.txt` and FFmpeg/ffprobe as described in
the README. The lightweight suite does not require downloaded model weights.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/skill_navigation.test.cjs
```

For a change involving actual inference, also run the GPU smoke checks from
the [technical guide](docs/TRACKER.md) in the full inference environment.
Report whether results use synthetic fixtures, cached detections or fresh GPU
inference. Keep accuracy claims separate from software test results.

## Before committing

Inspect `git status` and stage only intended changes. Then inspect the staged
diff and run the repository's publication guard:

```powershell
git diff --cached
.\.venv\Scripts\python.exe scripts/check-publication.py
git commit -m "Describe the change"
git push
```

The guard checks staged files for local-only folders, unreviewed media, personal
paths and several credential patterns. It is a useful check, not a guarantee
that arbitrary new data is safe to publish. Review new media before adding it.

Keep recordings, upload keys, raw reports, model weights and personal exports
in the ignored local directories. The public demo media is an explicit exception.
Use a GitHub no-reply commit email if you want to keep your personal email private.
