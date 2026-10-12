"""Stream native frames with source presentation timestamps and explicit rotation."""
from fractions import Fraction
from pathlib import Path
import json
import shutil
import subprocess
import av
import numpy as np

def probe(path):
    path=Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Video not found: {path}')
    if not shutil.which('ffprobe'):
        raise RuntimeError('ffprobe is required on PATH; see README setup')
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_format','-of','json',str(path)],text=True))
    if not data['streams']:
        raise ValueError('Input has no video stream')
    stream=data['streams'][0]
    rotation=0.
    for side in stream.get('side_data_list',[]):
        if 'rotation' in side:
            rotation=float(side['rotation'])
    if not stream.get('side_data_list') and 'rotate' in stream.get('tags',{}):
        rotation=float(stream['tags']['rotate'])
    if abs(rotation/90-round(rotation/90)) > 1e-4:
        raise ValueError('Only right-angle display rotations are supported')
    sar=stream.get('sample_aspect_ratio','1:1')
    if sar not in ('1:1','N/A','0:1'):
        raise ValueError('Non-square video pixels are not supported by this first version')
    w,h=stream['width'],stream['height']
    if round(rotation/90)%2:
        w,h=h,w
    return dict(path=str(path),encoded_width=stream['width'],encoded_height=stream['height'],
        display_width=w,display_height=h,rotation_degrees_ccw=rotation,
        rotation_handling='PyAV decodes encoded pixels; numpy applies display rotation once',
        stream_index=stream['index'],source_time_base=stream.get('time_base'),
        avg_frame_rate=stream.get('avg_frame_rate'),nominal_frame_rate=stream.get('r_frame_rate'),
        declared_frame_count=int(stream['nb_frames']) if stream.get('nb_frames','N/A').isdigit() else None,
        duration_seconds=float(stream.get('duration',data['format'].get('duration',0))),
        codec=stream.get('codec_name'),pixel_format=stream.get('pix_fmt'),
        sample_aspect_ratio=sar)

def orient(image,rotation):
    return np.ascontiguousarray(np.rot90(image,round(rotation/90)%4))

def decode(path,metadata=None,max_frames=None):
    meta=metadata or probe(path)
    first=None
    previous=None
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO'
        for i,frame in enumerate(container.decode(stream)):
            if max_frames is not None and i >= max_frames:
                break
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f'Frame {i} lacks source PTS; refusing fabricated timing')
            absolute=frame.pts*frame.time_base
            if first is None:
                first=absolute
            stamp=absolute-first
            if previous is not None and stamp <= previous:
                raise ValueError(f'Non-increasing source PTS at frame {i}')
            previous=stamp
            image=orient(frame.to_ndarray(format='bgr24'),meta['rotation_degrees_ccw'])
            if image.shape[:2] != (meta['display_height'],meta['display_width']):
                raise ValueError('Video dimensions changed midstream')
            yield dict(frame_index=i,timestamp_seconds=float(stamp),source_pts=frame.pts,
                       source_time_base=str(frame.time_base)),image

class VideoWriter:
    """VFR review video: one video frame per input frame; no source audio."""
    def __init__(self,path,width,height,rate='30/1'):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.container=av.open(str(self.path),mode='w')
        try:
            fps=Fraction(rate)
            if fps <= 0:
                fps=Fraction(30)
        except (ValueError,ZeroDivisionError):
            fps=Fraction(30)
        self.stream=self.container.add_stream('libx264',rate=fps)
        self.stream.width=width
        self.stream.height=height
        self.stream.pix_fmt='yuv420p' if width%2==0 and height%2==0 else 'yuv444p'
        self.stream.time_base=Fraction(1,90000)
        self.stream.codec_context.time_base=Fraction(1,90000)
        self.stream.options={'crf':'18','preset':'fast','bf':'0'}

    def write(self,image,timestamp):
        frame=av.VideoFrame.from_ndarray(image,format='bgr24')
        frame.pts=round(timestamp*90000)
        frame.time_base=Fraction(1,90000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container=None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
