# RedTuckBacks demonstration

The README preview comes from an existing tracking run of **RedTuckBacks.mov**,
the clip selected by the project owner for this public demo. It shows actual
predicted skeletons, rather than generated poses or hand-corrected landmarks.

The exported source run contains 1,925 frames over approximately 8.02 seconds.
The public MP4 is a presentation copy: cropped to the athlete and trampoline,
resized to 540 by 600 pixels and sampled at 30 fps. The looping GIF is a smaller
12 fps preview. Neither contains audio or the original recording metadata.
The underlying tracker export retains native frame timestamps; the demo's
resampling is only for browser-friendly playback.

This routine informed development, so it is not held-out validation. Skeletons
may be inaccurate during inversion or occlusion. User-supplied skill labels
remain separate from the predictions. See [validation limits](VALIDATION.md).

Only the prepared MP4 and GIF are included. Original recordings, job metadata,
joint exports, access keys and other athletes' recordings remain local. The
demo footage is excluded from the software's MIT license.
