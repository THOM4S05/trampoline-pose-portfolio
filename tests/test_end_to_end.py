import json
import numpy as np
import pytest
from trampoline_pose.api import PoseTracker
from trampoline_pose.geometry import crop_geometry
from trampoline_pose.output import validate
from trampoline_pose.types import TrackerConfig,AthleteSelection
from trampoline_pose.video import VideoWriter,decode


def test_injected_backend_exports_without_inference_packages(tmp_path, monkeypatch):
    from trampoline_pose import backend
    real_version = backend.importlib.metadata.version
    def version_without_torch(name):
        if name == 'torch':
            raise backend.importlib.metadata.PackageNotFoundError(name)
        return real_version(name)
    monkeypatch.setattr(backend.importlib.metadata, 'version', version_without_torch)
    path = tmp_path / 'no-model-packages.mp4'
    video(path)
    result = PoseTracker(TrackerConfig(device='cpu'), KnownPoseBackend()).process_video(
        path, AthleteSelection(0, (10, 10, 50, 70)))
    result.save_json(tmp_path / 'joints.json')
    validate(result.data)
    assert result.data['dependencies']['torch'] is None
    assert result.data['dependencies']['numpy'] == np.__version__

class KnownPoseBackend:
    provenance={'test_backend':'known source coordinates; never GPU evidence'}
    def detect(self,image):
        return np.array([[10.,10.,50.,70.]]),np.array([.95])
    def estimate(self,image,box):
        points=np.array([[20.+i,30.+i] for i in range(17)])
        scores=np.ones(17)*.9
        points[0]=[-4,1]
        scores[1]=.1
        return points,scores,crop_geometry(box)
    def metrics(self):
        return dict(peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None)

def video(path):
    with VideoWriter(path,96,80,'25/1') as writer:
        for t in [0,.04,.10,.15]:
            writer.write(np.full((80,96,3),100,dtype=np.uint8),t)

def test_export_overlay_cache_and_all_joints(tmp_path):
    path=tmp_path/'input.mp4'
    video(path)
    tracker=PoseTracker(TrackerConfig(device='cpu'),KnownPoseBackend())
    result=tracker.process_video(path,AthleteSelection(2,(10,10,50,70)),cache_path=tmp_path/'cache.json')
    result.save_json(tmp_path/'joints.json')
    result.save_csv(tmp_path/'joints.csv')
    result.save_video(tmp_path/'overlay.mp4')
    data=json.loads((tmp_path/'joints.json').read_text())
    validate(data)
    assert data['processed_frame_count']==4
    for f in data['frames']:
        assert len(f['joints'])==17
        assert f['joints']['nose']['raw']['x']==-4
        assert f['joints']['nose']['visibility']=='out_of_frame'
        assert f['joints']['nose']['final']['x'] is None
        assert f['joints']['left_eye']['status']=='missing'
        assert f['joints']['left_knee']['raw']['x']==33
    decoded=list(decode(tmp_path/'overlay.mp4'))
    np.testing.assert_allclose([f['timestamp_seconds'] for f,_ in decoded],[0,.04,.10,.15],atol=1/90000)
    rerun=tracker.process_video(path,AthleteSelection(2,(10,10,50,70)),cache_path=tmp_path/'cache.json')
    assert rerun.data['diagnostics']['detection_cache_used']
    assert rerun.data['frames']==result.data['frames']

def test_cancellation_is_explicit(tmp_path):
    path=tmp_path/'input.mp4'
    video(path)
    with pytest.raises(InterruptedError):
        PoseTracker(TrackerConfig(device='cpu'),KnownPoseBackend()).process_video(path,AthleteSelection(0,(10,10,50,70)),cancel=lambda:True)

def test_api_cannot_overwrite_source_and_rejects_changed_video(tmp_path):
    path=tmp_path/'input.mp4'
    video(path)
    tracker=PoseTracker(TrackerConfig(device='cpu'),KnownPoseBackend())
    with pytest.raises(ValueError):
        tracker.process_video(path,AthleteSelection(0,(10,10,50,70)),cache_path=path)
    result=tracker.process_video(path,AthleteSelection(0,(10,10,50,70)))
    for action in [result.save_json,result.save_csv,result.save_video]:
        with pytest.raises(ValueError):
            action(path)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash changed'):
        result.save_video(tmp_path/'overlay.mp4')

def test_schema_rejects_missing_joint_and_status_mismatch(tmp_path):
    path=tmp_path/'input.mp4'
    video(path)
    result=PoseTracker(TrackerConfig(device='cpu'),KnownPoseBackend()).process_video(path,AthleteSelection(0,(10,10,50,70)))
    data=json.loads(json.dumps(result.data))
    data['frames'][0]['joints']['left_knee']['status']='missing'
    with pytest.raises(ValueError):
        validate(data)
    del data['frames'][0]['joints']['left_knee']
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        validate(data)

def test_validator_rejects_nonstandard_json_nan(tmp_path):
    path=tmp_path/'input.mp4'
    video(path)
    result=PoseTracker(TrackerConfig(device='cpu'),KnownPoseBackend()).process_video(path,AthleteSelection(0,(10,10,50,70)))
    data=json.loads(json.dumps(result.data))
    data['frames'][0]['joints']['left_knee']['raw']['x']=float('nan')
    with pytest.raises(ValueError,match='finite'):
        validate(data)


def test_pipeline_recovers_after_occlusion_without_inventing_hidden_joints(tmp_path):
    class OccludedBackend(KnownPoseBackend):
        frame=0
        def detect(self,image):
            i=self.frame
            self.frame+=1
            if i in (1,2,3):
                return np.empty((0,4)),np.empty(0)
            return super().detect(image)

    path=tmp_path/'occlusion.mp4'
    with VideoWriter(path,96,80,'25/1') as writer:
        for t in (0.,.3,.6,1.,1.04,1.08,1.12):
            writer.write(np.full((80,96,3),100,dtype=np.uint8),t)
    result=PoseTracker(TrackerConfig(device='cpu'),OccludedBackend()).process_video(
        path,AthleteSelection(0,(10,10,50,70)))
    result.save_json(tmp_path/'recovered.json')
    result.save_video(tmp_path/'recovered.mp4')
    frames=result.data['frames']
    assert [f['track_state'] for f in frames]==['tracked']+['lost']*5+['tracked']
    assert all(j['final']['x'] is None for f in frames[1:6] for j in f['joints'].values())
    assert frames[-1]['joints']['left_knee']['status']=='observed'
    assert frames[-1]['diagnostics']['reason']=='reacquired_after_occlusion'
    np.testing.assert_allclose([r['timestamp_seconds'] for r,_ in decode(tmp_path/'recovered.mp4')],
                               [f['timestamp_seconds'] for f in frames],atol=1/90000)
