# Personal video inbox

Upload a video from your Mac, PC or phone; slide to an exact decoded frame;
draw a full-body gymnast box; then queue processing on the Windows PC.
The sender needs only a browser (and Tailscale for access away from the PC).

## Start on the processing PC

From the project folder in PowerShell, after setting up the tracker:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt --index-url https://pypi.org/simple
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-portal.ps1
```

Open <http://127.0.0.1:8765>. Paste the generated key from
`.portal/access-key.txt` into the login form. Keep this key private. The key is
not put in a URL or browser local storage. Sessions expire after 12 hours.
The server listens only on 127.0.0.1. It does not open a campus-facing port.
Keep the launcher running; closing it stops the portal. Ctrl+C stops it after
any active worker has finished. Start only one portal process per data directory.

## The selection flow

1. **Upload:** Choose or drop one clip. Transfers use 8 MiB chunks, with safe
   retries for a lost response. Stay on the page until the upload completes.
   If the page closes or the connection fails permanently during upload, choose
   the file again to restart and delete the unfinished entry. Cross-session
   upload resumption is not implemented in this first version.
2. **Prepare:** The PC decodes native frames into JPEG previews. This runs in
   the same queue as processing, so a new video may wait behind an active job.
3. **Choose:** Move the slider, use Previous / Next, or type a frame number.
   The displayed frame numbers start at **1**; the tracker API indices start at
   **0**. For the existing sample's API seed 150, choose frame **151** in the UI.
   The timestamp is the source presentation time, not a calculation from FPS.
4. **Outline:** Drag around the entire gymnast, including hands and feet.
   Drag inside to move; drag the corner handles to resize. On a phone, the
   **Tap two corners instead** control can be easier. The page scrolls outside
   the image. Exact coordinate inputs are also available. Redraw clears the box.
   **Changing frames always clears the box** to prevent stale selections.
5. **Process:** The button enables only with a loaded frame and a valid box.
   The PC follows the selected athlete forwards and backwards through the
   video. You can close the browser after queuing. Return to Your videos to see
   status and download the review MP4, joint JSON or CSV.

## Skill Analysis

After pose tracking finishes, open **Skill Analysis** in the portal header (or
`/skills` on the same local or private Tailscale address). Choose a tracked video
and press **Run skill analysis on my PC**. The page shows whether analysis is
queued, running, complete or needs attention. You can close it and return later;
the result is saved with that video in `.portal/jobs/<job-id>/skills.json` and
`skills.csv`. When it finishes, the per-flight table appears automatically and
both files can be downloaded from the page. A failed analysis can be retried.

Each flight with a supported shape time also has a saved skeleton-overlay image
from the nearest original source frame. Its caption shows the actual frame time;
the displayed angles describe the sustained window rather than that single frame.
Flights without a selected shape time show an explanation instead. For older
analysis results, press **Add tightest-shape overlay images** to generate them.

The table shows contact proxy boundaries, supported backward somersault counts,
estimated execution shape, sustained tightest-shape time and angles, and review
reasons. An unknown count remains unknown even if there is a tentative candidate.
The opening or trailing partial flight is marked as such. Intended skill identity
is unconfirmed; the pike-back development labels are not used for portal runs.
Review the pose overlay and contacts before using these estimates.

Analysis runs in the portal's existing single worker queue on the Windows PC.
Its status and outputs survive a browser refresh and a normal PC restart. An
analysis interrupted by a restart becomes retryable; partial output is not
offered as complete. To start or restart the portal, run the same
`scripts/start-portal.ps1` command shown above, with `-PublicUrl` again if you
use the private Tailscale address. Keep the PC awake and the launcher running.

Previews use the tracker's own decoder and rotation handling. Preview pixel
coordinates are mapped to full-resolution display-oriented source coordinates;
browser video seeking and nominal frame rate are never used for seed indices.
Selection drafts stay in this browser; a submitted job stores its selection on
the PC. The current UI exposes the gymnast box; the CLI's optional flight-region
restriction is not exposed in this first version.

## Connect privately across campus

Tailscale must be installed and signed in on the processing PC and the sending
Mac, PC or phone. Use your own personal network and account; if your Tailscale
network includes other people, restrict access to this PC's HTTPS service using
Tailscale access rules. The portal key is an additional access gate.

1. Install Tailscale from <https://tailscale.com/download> and sign in on both
   devices. This requires your account interaction. Keep the PC awake and online.
2. On the processing PC run:

   ```powershell
   & 'C:\Program Files\Tailscale\tailscale.exe' serve --bg http://127.0.0.1:8765
   ```

   Follow Tailscale's HTTPS enablement instructions if prompted. Note the exact
   private `https://your-pc.your-network.ts.net` address it prints.
3. Stop and restart the portal with that address (replace the example):

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-portal.ps1 -PublicUrl 'https://your-pc.your-network.ts.net'
   ```

4. With Tailscale connected on the sender, open that HTTPS address and unlock
   with the portal key. Test a short clip from another part of campus.

Use **Serve**, not Funnel: Serve is for the private Tailscale network; Funnel
would make the endpoint public. Do not forward router ports or bind the Python
server to `0.0.0.0`. See the [official Serve documentation](https://tailscale.com/docs/reference/tailscale-cli/serve).
The app explicitly allows only localhost and the configured HTTPS origin,
rejects cross-origin writes, and sets secure cookies for the HTTPS hostname.
Forwarded identity headers are not used as authentication.

Check sharing with `tailscale serve status`; disable this HTTPS Serve endpoint
with `tailscale serve --https=443 off`. Tailscale background Serve persists across
reboots, but the Python portal currently must be started again manually. A PC
that is asleep, offline, or awaiting network login cannot receive uploads or
process jobs. Actual eduroam connectivity still needs a two-device test.

## Storage and operations

- Up to 2 GiB and 20,000 decoded frames per video; 30 jobs / 20 GiB of original
  videos per inbox. Previews, detection caches and results take additional disk
  space. The app checks available space during upload and preparation.
- Originals, previews and outputs live under `.portal/jobs/<job-id>/`. They are
  never automatically deleted or sent to cloud storage. Use **Delete video** to
  remove that inbox copy and its results; it does not affect the sender's file.
- A single worker prepares videos and runs the existing tracker, keeping GPU
  processing sequential. Submitted settings are locked during processing.
- A PC/app restart marks interrupted processing ready to submit again; it does
  not claim that a partial export is complete. Interrupted preparation can be
  retried. Completed jobs remain available across restarts.
- Frame previews use JPEGs up to 960 px on their longest side. All processing
  uses the original uploaded video. Nothing is transcoded before tracking.
- Pose accuracy limitations remain those of the existing tracker. Review the
  overlay; a completed job does not guarantee correct anatomical identity.
- This is a personal private-network service, not a hardened public upload
  platform. It uses Waitress, explicit host/origin checks, authenticated media
  routes, bounded requests and generated file paths. Keep the PC account and
  `.portal` directory private; local disk encryption is managed by Windows.

To invalidate all sessions and rotate the key, stop the server, remove only
`.portal/access-key.txt`, then restart. Do not remove the jobs directory to rotate
the key. Detailed processing failures are written to the server's stderr log.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
node --check src/trampoline_pose/web/app.js
```

Portal tests cover unauthenticated access, cookies, hosts, cross-origin writes,
size and file constraints, incomplete/chunk-retry uploads, seed validation,
rotated/VFR preview consistency, exports and interrupted-job recovery. Tests
inject a known-coordinate backend; they are not GPU accuracy evidence.

The codec tests send actual encoded fixtures through upload, preparation,
selection, tracking and overlay export: H.264 in MP4/MOV/M4V, HEVC in MP4/MOV,
10-bit HEVC in MOV, VP9 in WebM/MKV, MPEG-4 in AVI, AV1 in MP4, and ProRes in
MOV. All are returned as H.264 MP4 review videos. This verifies those fixture
combinations, not every codec profile or damaged file. HDR tone mapping and
Apple Photos slow-motion editing metadata have not been independently validated.
The source file's decoded presentation timestamps are preserved in the overlay;
the review export currently omits audio, matching the existing tracker.

Container-declared frame counts may differ from the playable frames, especially
for edited MOV/MP4 files. The slider uses the actual decoded frame count and
timestamps. The verification script compares every source and overlay frame
without treating the container's declared count as authoritative.

Skill analysis version 7 improves contact proposals when camera height changes and feet disappear near the bed. Existing completed jobs offer **Update skill analysis** after refreshing the Skill Analysis page. Routine windows may be proposed from video review; the page marks their provenance, and **Show preparation and outbounces** restores all full-clip spans. These windows only filter display; they do not supply skill identities or analyzer boundaries.

**Redo analysis:** completed skill results now always offer this button. It reruns skill analysis and regenerates JSON, CSV and overlay previews from the saved joints.json on the PC; it does not rerun pose tracking. Concurrent requests remain blocked. A preview marked as a review candidate comes from a potentially tighter shape interval with missing joints; its exact tightest time is unverified and displayed angles remain measurements from a supported held shape.
