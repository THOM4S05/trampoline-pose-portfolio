NAMES = ['nose','left_eye','right_eye','left_ear','right_ear',
         'left_shoulder','right_shoulder','left_elbow','right_elbow',
         'left_wrist','right_wrist','left_hip','right_hip','left_knee',
         'right_knee','left_ankle','right_ankle']
EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
FLIP = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]

def metadata():
    return dict(name='COCO-17',version='1.0',keypoint_names=NAMES,edges=EDGES,
                side_convention='athlete anatomical left/right; never screen position',
                derived_landmarks=['pelvis_center','shoulder_center'])

def derived(joints):
    result={}
    for name, pair in [('pelvis_center',('left_hip','right_hip')),('shoulder_center',('left_shoulder','right_shoulder'))]:
        a,b=[joints[n] for n in pair]
        usable = a['final']['x'] is not None and b['final']['x'] is not None
        result[name]=dict(x=(a['final']['x']+b['final']['x'])/2 if usable else None,
                          y=(a['final']['y']+b['final']['y'])/2 if usable else None,
                          status='missing' if not usable else ('inferred' if 'inferred' in (a['status'],b['status']) else 'observed'),
                          confidence=None,source_joints=list(pair))
    return result
