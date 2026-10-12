import csv
import hashlib
import json
import math
from pathlib import Path
import jsonschema

def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def validate(data):
    schema=json.loads(Path(__file__).with_name('schema.json').read_text(encoding='utf-8'))
    jsonschema.Draft202012Validator(schema).validate(data)
    previous=-1.
    for i,frame in enumerate(data['frames']):
        if frame['frame_index'] != i or frame['timestamp_seconds'] <= previous:
            raise ValueError('Frame indices/timestamps do not align')
        previous=frame['timestamp_seconds']
        for joint in frame['joints'].values():
            for field in ('raw','final'):
                if any(joint[field][axis] is not None and not math.isfinite(joint[field][axis]) for axis in ('x','y')):
                    raise ValueError('Coordinates must be finite or null')
                if (joint[field]['x'] is None) != (joint[field]['y'] is None):
                    raise ValueError('Joint x/y must both be present or both be null')
            if (joint['status']=='missing') != (joint['final']['x'] is None):
                raise ValueError('Missing status and final coordinates disagree')
            if joint['status']=='observed' and (joint['raw']['x'],joint['raw']['y']) != (joint['final']['x'],joint['final']['y']):
                raise ValueError('Observed final coordinates must match raw coordinates')
            if frame['track_state']=='lost' and joint['final']['x'] is not None:
                raise ValueError('Lost athlete must have missing final joints')
            if joint['raw']['model_score'] is not None and not math.isfinite(joint['raw']['model_score']):
                raise ValueError('Model scores must be finite or null')
    if data['processed_frame_count'] != len(data['frames']):
        raise ValueError('Processed frame count mismatch')

def save_json(data,path):
    # Roundtrip normalises tuples and rejects NaN/infinity before schema checking.
    text=json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)
    validate(json.loads(text))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(text,encoding='utf-8')
    temporary.replace(path)

def save_csv(data,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file)
        writer.writerow(['frame_index','timestamp_seconds','track_state','joint','raw_x','raw_y','model_score','final_x','final_y','status','visibility','flags'])
        for frame in data['frames']:
            for name,joint in frame['joints'].items():
                writer.writerow([frame['frame_index'],frame['timestamp_seconds'],frame['track_state'],name,
                    joint['raw']['x'],joint['raw']['y'],joint['raw']['model_score'],joint['final']['x'],joint['final']['y'],
                    joint['status'],joint['visibility'],';'.join(joint['flags'])])
