"""Exercise the private inbox at its HTTP boundary, with real VFR media."""
import json
import time

import av
import cv2
import numpy as np
import pytest

pytest.importorskip('flask')
from trampoline_pose.portal import create_app, CHUNK_SIZE, JobStore
from trampoline_pose.api import PoseTracker
from trampoline_pose.geometry import crop_geometry
from trampoline_pose.output import validate
from trampoline_pose.types import TrackerConfig
from trampoline_pose.video import VideoWriter, decode

ORIGIN = {'Origin': 'http://localhost:8765', 'X-Portal-Request': '1'}
BASE = 'http://localhost:8765'


def test_progress_write_tolerates_transient_windows_file_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from trampoline_pose.portal import atomic_json
    original = Path.replace
    attempts = []
    def transient_lock(path, target):
        attempts.append(target)
        if len(attempts) == 1:
            raise PermissionError('temporary Windows read handle')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', transient_lock)
    atomic_json(tmp_path/'job.json', {'state':'processing'})
    assert json.loads((tmp_path/'job.json').read_text()) == {'state':'processing'}
    assert len(attempts) == 2


def test_reference_alignment_survives_renumbering_and_preserves_labels():
    from trampoline_pose.portal import align_reference
    reference = dict(skills=[dict(skill_index=2, start_seconds=2, end_seconds=4,
                                 skill_identity='User reference', twist_count=.5)])
    predictions = [dict(skill_index=7, visible_start_seconds=2.1, visible_end_seconds=4.1)]
    aligned = align_reference(reference, predictions)
    assert aligned['skills'][0]['skill_index'] == 7
    assert aligned['skills'][0]['reference_flight_index'] == 2
    assert aligned['skills'][0]['twist_count'] == .5
    assert reference['skills'][0]['skill_index'] == 2
    assert predictions[0]['skill_index'] == 7


class TestBackend:
    __test__ = False
    provenance = {'test_backend': 'known coordinates, not GPU evidence'}
    def detect(self, image):
        return np.array([[10., 10., 40., 60.]]), np.array([.95])
    def estimate(self, image, box):
        return np.array([[20.+i, 30.+i] for i in range(17)]), np.ones(17)*.9, crop_geometry(box)
    def metrics(self):
        return dict(peak_gpu_allocated_bytes=None, peak_gpu_reserved_bytes=None)


@pytest.fixture
def portal(tmp_path):
    config = TrackerConfig(device='cpu')
    app = create_app(tmp_path / 'inbox', config, tracker_factory=lambda c: PoseTracker(c, TestBackend()))
    app.testing = True
    client = app.test_client()
    key = (tmp_path / 'inbox' / 'access-key.txt').read_text()
    assert client.post('/api/login', json={'key': key}, headers=ORIGIN, base_url=BASE).status_code == 200
    yield app, client, tmp_path
    app.extensions['job_store'].worker.shutdown(wait=True)


def request(client, method, path, **kwargs):
    return client.open(path, method=method, headers=ORIGIN, base_url=BASE, **kwargs)


def video(path):
    with VideoWriter(path, 96, 80, '25/1') as writer:
        for t in [0, .04, .1, .15]:
            image = np.full((80, 96, 3), 100, dtype=np.uint8)
            image[:20, :20] = [0, 0, 255]
            writer.write(image, t)


def wait_state(app, job_id, expected):
    limit = time.monotonic() + 20
    store = app.extensions['job_store']
    while time.monotonic() < limit:
        with store.lock:
            job = dict(store.jobs[job_id])
        if job['state'] == expected:
            return job
        assert job['state'] != 'error', job
        time.sleep(.03)
    pytest.fail(f'Job did not reach {expected}: {job}')


def uploaded(portal, path=None):
    app, client, tmp = portal
    if path is None:
        path = tmp / 'vfr.mp4'
        video(path)
    data = path.read_bytes()
    response = request(client, 'POST', '/api/jobs', json={'name': path.name, 'size': len(data)})
    assert response.status_code == 201
    job_id = response.json['id']
    assert request(client, 'PUT', f'/api/jobs/{job_id}/upload?offset=0', data=data, content_type='application/octet-stream').status_code == 200
    assert request(client, 'POST', f'/api/jobs/{job_id}/finish').status_code == 202
    wait_state(app, job_id, 'ready')
    return job_id


def test_auth_host_csrf_and_security_headers(portal):
    app, client, tmp = portal
    stranger = app.test_client()
    for path in ['/api/jobs', '/api/jobs/abc', '/api/jobs/abc/frames/0.jpg', '/api/jobs/abc/results/joints.json']:
        assert stranger.get(path, base_url=BASE).status_code == 401
    assert client.get('/api/jobs', base_url='http://evil.example').status_code == 400
    assert client.post('/api/jobs', base_url=BASE, json={'name':'x.mp4', 'size':3}).status_code == 403
    assert client.post('/api/jobs', base_url=BASE, headers={**ORIGIN, 'Origin':'https://evil.example'}, json={}).status_code == 403
    result = client.get('/', base_url=BASE)
    assert "frame-ancestors 'none'" in result.headers['Content-Security-Policy']
    assert result.headers['X-Content-Type-Options'] == 'nosniff'
    for _ in range(5):
        assert request(stranger, 'POST', '/api/login', json={'key':'wrong'}).status_code == 401
    assert request(stranger, 'POST', '/api/login', json={'key':'wrong'}).status_code == 429
    assert request(client, 'POST', '/api/logout').status_code == 200
    assert client.get('/api/jobs', base_url=BASE).status_code == 401


def test_private_https_cookie_and_origin(tmp_path):
    app = create_app(tmp_path, public_url='https://dorm.example.ts.net')
    client = app.test_client()
    result = client.post('/api/login', base_url='https://dorm.example.ts.net', headers={'Origin':'https://dorm.example.ts.net','X-Portal-Request':'1'}, json={'key':(tmp_path/'access-key.txt').read_text()})
    assert result.status_code == 200
    assert 'Secure' in result.headers['Set-Cookie'] and 'HttpOnly' in result.headers['Set-Cookie'] and 'SameSite=Strict' in result.headers['Set-Cookie']
    app.extensions['job_store'].worker.shutdown()


def test_upload_limits_partial_and_safe_retry(portal):
    app, client, _ = portal
    for payload in [{'name':'bad.exe','size':3}, {'name':'a.mp4','size':True}, {'name':'a.mp4','size':-1}]:
        assert request(client, 'POST', '/api/jobs', json=payload).status_code == 400
    assert request(client, 'POST', '/api/jobs', json={'name':'a.mp4','size':3*1024**3}).status_code == 413
    job = request(client, 'POST', '/api/jobs', json={'name':'../../escape.mp4','size':6}).json
    job_id=job['id']
    assert job['name']=='escape.mp4'
    assert request(client, 'POST', f'/api/jobs/{job_id}/finish').status_code == 409
    for _ in range(2):
        response=request(client, 'PUT', f'/api/jobs/{job_id}/upload?offset=0', data=b'abc', content_type='application/octet-stream')
        assert response.status_code==200 and response.json['offset']==3
    assert request(client, 'PUT', f'/api/jobs/{job_id}/upload?offset=0', data=b'xyz', content_type='application/octet-stream').status_code==409
    assert request(client, 'PUT', f'/api/jobs/{job_id}/upload?offset=3', data=b'defg', content_type='application/octet-stream').status_code==409
    assert request(client, 'PUT', f'/api/jobs/{job_id}/upload?offset=3', data=b'x'*(CHUNK_SIZE+1), content_type='application/octet-stream').status_code==413
    assert request(client, 'POST', f'/api/jobs/{job_id}/run', json={'frame_index':0,'bbox_xyxy':[1,1,5,5]}).status_code==409
    assert request(client, 'GET', '/api/jobs/not-a-job').status_code==404
    assert request(client, 'DELETE', f'/api/jobs/{job_id}').status_code==200
    assert not app.extensions['job_store'].directory(job).exists()


def test_vfr_preview_selection_and_outputs(portal):
    app, client, tmp = portal
    job_id = uploaded(portal)
    detail = request(client,'GET',f'/api/jobs/{job_id}').json
    assert detail['frame_count']==4
    np.testing.assert_allclose(detail['timestamps'],[0,.04,.1,.15],atol=1/90000)
    assert (detail['width'], detail['height'])==(96,80)
    frame = request(client,'GET',f'/api/jobs/{job_id}/frames/2.jpg')
    assert frame.status_code==200
    assert cv2.imdecode(np.frombuffer(frame.data,np.uint8),cv2.IMREAD_COLOR).shape[:2]==(80,96)
    assert request(client,'GET',f'/api/jobs/{job_id}/frames/4.jpg').status_code==404
    assert request(client,'GET',f'/api/jobs/{job_id}/results/joints.json').status_code==404
    selection={'frame_index':2,'bbox_xyxy':[10,10,40,60]}
    for bad in [{**selection,'frame_index':4}, {**selection,'frame_index':True}, {**selection,'bbox_xyxy':[-1,0,40,60]}, {**selection,'bbox_xyxy':[1,1,float('nan'),60]}, {**selection,'bbox_xyxy':[1,1,2,2]}]:
        assert request(client,'POST',f'/api/jobs/{job_id}/run',json=bad).status_code==400
    assert request(client,'POST',f'/api/jobs/{job_id}/selection',json=selection).status_code==200
    assert request(client,'POST',f'/api/jobs/{job_id}/run',json=selection).status_code==202
    assert request(client,'POST',f'/api/jobs/{job_id}/run',json=selection).status_code==409
    wait_state(app,job_id,'done')
    result=request(client,'GET',f'/api/jobs/{job_id}/results/joints.json')
    assert result.status_code==200
    exported=json.loads(result.data)
    validate(exported)
    assert exported['selection']['frame_index']==2 and exported['selection']['bbox_xyxy']==selection['bbox_xyxy']
    assert request(client,'GET',f'/api/jobs/{job_id}/results/joints.csv').status_code==200
    response=request(client,'GET',f'/api/jobs/{job_id}/results/overlay.mp4')
    assert response.status_code==200
    response.close()
    assert request(client,'GET',f'/api/jobs/{job_id}/results/source.mp4').status_code==404


def test_skill_analysis_page_runs_locally_and_persists(portal):
    app, client, _ = portal
    stranger = app.test_client()
    assert client.get('/skills', base_url=BASE).status_code == 200
    assert b'Skill Analysis' in client.get('/skills', base_url=BASE).data
    job_id = uploaded(portal)
    route = f'/api/jobs/{job_id}/analysis'
    assert request(client, 'POST', route).status_code == 409
    assert stranger.get(route, base_url=BASE).status_code == 401
    assert stranger.post(route, base_url=BASE, headers=ORIGIN).status_code == 401
    assert request(client, 'GET', f'/api/jobs/{job_id}/results/skills.json').status_code == 404
    assert request(client, 'POST', f'/api/jobs/{job_id}/run', json={
        'frame_index': 2, 'bbox_xyxy': [10, 10, 40, 60]}).status_code == 202
    wait_state(app, job_id, 'done')
    response = request(client, 'POST', route)
    assert response.status_code == 202
    assert request(client, 'POST', route).status_code == 409
    store = app.extensions['job_store']
    store.worker.shutdown(wait=True)
    detail = request(client, 'GET', route).json
    assert detail['analysis']['state'] == 'done'
    assert isinstance(detail['skills'], list)
    assert request(client, 'GET', f'/api/jobs/{job_id}/results/skills.csv').status_code == 200
    saved = request(client, 'GET', f'/api/jobs/{job_id}/results/skills.json').json
    assert saved['result_schema_version'] == '0.1.0'
    assert request(client, 'GET', f'/api/jobs/{job_id}/results/comparison.json').status_code == 404
    recovered = JobStore(store.root, store.config)
    assert recovered.jobs[job_id]['analysis']['state'] == 'done'
    recovered.worker.shutdown()


def test_skill_analysis_rejects_bad_pose_and_can_retry(portal):
    app, client, _ = portal
    job_id = uploaded(portal)
    assert request(client, 'POST', f'/api/jobs/{job_id}/run', json={
        'frame_index': 2, 'bbox_xyxy': [10, 10, 40, 60]}).status_code == 202
    wait_state(app, job_id, 'done')
    store = app.extensions['job_store']
    source = store.directory(store.jobs[job_id]) / 'joints.json'
    original = source.read_bytes()
    source.write_text('{bad', encoding='utf-8')
    route = f'/api/jobs/{job_id}/analysis'
    assert request(client, 'POST', route).status_code == 202
    store.worker.shutdown(wait=True)
    assert request(client, 'GET', route).json['analysis']['state'] == 'error'
    assert request(client, 'GET', f'/api/jobs/{job_id}/results/skills.json').status_code == 404
    source.write_bytes(original)
    # A fresh app loads the saved error and can queue a retry.
    again = create_app(store.root.parent)
    again.testing = True
    retry = again.test_client()
    key = (store.root.parent / 'access-key.txt').read_text()
    assert retry.post('/api/login', json={'key': key}, headers=ORIGIN, base_url=BASE).status_code == 200
    assert request(retry, 'POST', route).status_code == 202
    again.extensions['job_store'].worker.shutdown(wait=True)
    assert request(retry, 'GET', route).json['analysis']['state'] == 'done'


def test_explicit_redo_reuses_saved_pose_and_rejects_concurrent_runs(portal, monkeypatch):
    import threading
    app, client, _ = portal
    jid = uploaded(portal)
    assert request(client, 'POST', f'/api/jobs/{jid}/run', json={
        'frame_index': 2, 'bbox_xyxy': [10, 10, 40, 60]}).status_code == 202
    wait_state(app, jid, 'done')
    route = f'/api/jobs/{jid}/analysis'
    store = app.extensions['job_store']
    assert request(client, 'POST', route).status_code == 202
    store.worker.submit(lambda: None).result(timeout=20)
    assert request(client, 'POST', route).status_code == 409
    assert request(client, 'POST', route, json={'redo':'yes'}).status_code == 400
    pose = (store.directory(store.jobs[jid]) / 'joints.json').read_bytes()
    release = threading.Event()
    original = store.analyze_skills
    def paused(job):
        assert release.wait(10)
        original(job)
    monkeypatch.setattr(store, 'analyze_skills', paused)
    try:
        assert request(client, 'POST', route, json={'redo':True}).status_code == 202
        assert request(client, 'POST', route, json={'redo':True}).status_code == 409
    finally:
        release.set()
    store.worker.submit(lambda: None).result(timeout=20)
    assert request(client, 'GET', route).json['analysis']['state'] == 'done'
    assert (store.directory(store.jobs[jid]) / 'joints.json').read_bytes() == pose
    assert request(client, 'GET', f'/api/jobs/{jid}/results/skills.csv').status_code == 200


def test_interrupted_skill_analysis_recovers_without_exposing_partial_results(portal):
    app, client, _ = portal
    job_id = uploaded(portal)
    store = app.extensions['job_store']
    job = store.jobs[job_id]
    store.update(job, state='done', analysis=dict(state='running', message='Estimating'))
    assert request(client, 'DELETE', f'/api/jobs/{job_id}').status_code == 409
    recovered = JobStore(store.root, store.config)
    assert recovered.jobs[job_id]['analysis']['state'] == 'error'
    assert request(client, 'GET', f'/api/jobs/{job_id}/results/skills.csv').status_code == 404
    recovered.worker.shutdown()


def test_skill_images_use_nearest_native_pts_and_authenticated_route(portal):
    from trampoline_pose.skill_images import save_shape_images
    app, client, _ = portal
    job_id = uploaded(portal)
    request(client, 'POST', f'/api/jobs/{job_id}/run', json={
        'frame_index': 2, 'bbox_xyxy': [10, 10, 40, 60]})
    wait_state(app, job_id, 'done')
    store = app.extensions['job_store']
    job = store.jobs[job_id]
    directory = store.directory(job)
    data = json.loads((directory / 'joints.json').read_text())
    result = dict(skills=[dict(skill_index=1, visible_start_seconds=0, visible_end_seconds=.15,
                              execution=dict(shape_timestamp_seconds=.09)),
                         dict(skill_index=2, execution=dict(shape_timestamp_seconds=None))])
    manifest = save_shape_images(data, result, store.source(job), directory / 'skill-images')
    assert manifest['1']['frame_index'] == 2
    assert manifest['1']['timestamp_seconds'] == pytest.approx(.1, abs=1/90000)
    assert manifest['2']['available'] is False
    (directory / 'skill-images.json').write_text(json.dumps(manifest), encoding='utf-8')
    store.update(job, analysis=dict(state='done', images_version=1))
    route = f'/api/jobs/{job_id}/skill-images/1.jpg'
    response = request(client, 'GET', route)
    assert response.status_code == 200 and response.mimetype == 'image/jpeg'
    assert cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (62, 42)
    assert app.test_client().get(route, base_url=BASE).status_code == 401
    assert request(client, 'GET', f'/api/jobs/{job_id}/skill-images/2.jpg').status_code == 404
    data['frames'][2]['timestamp_seconds'] = .101
    with pytest.raises(ValueError, match='timestamp'):
        save_shape_images(data, result, store.source(job), directory / 'skill-images')


def test_rotated_preview_matches_tracker_coordinates(portal):
    import subprocess
    _,client,tmp=portal
    source=tmp/'unrotated.mp4';rotated=tmp/'rotated.mp4'
    video(source)
    subprocess.run(['ffmpeg','-v','error','-y','-display_rotation:v:0','90','-i',str(source),'-c','copy',str(rotated)],check=True)
    job_id=uploaded(portal,rotated)
    detail=request(client,'GET',f'/api/jobs/{job_id}').json
    assert (detail['width'],detail['height'])==(80,96)
    response=request(client,'GET',f'/api/jobs/{job_id}/frames/0.jpg')
    preview=cv2.imdecode(np.frombuffer(response.data,np.uint8),cv2.IMREAD_COLOR)
    tracker_frame=next(decode(rotated))[1]
    assert preview.shape==tracker_frame.shape
    assert np.abs(preview.astype(float)-tracker_frame).mean()<3


def test_restart_preserves_upload_and_requires_resubmit(portal):
    app,client,_=portal
    job_id=uploaded(portal)
    store=app.extensions['job_store']
    job=store.jobs[job_id]
    store.update(job,state='processing',selection={'frame_index':2,'bbox_xyxy':[10,10,40,60]})
    recovered=JobStore(store.root,store.config)
    assert recovered.jobs[job_id]['state']=='ready'
    assert recovered.jobs[job_id]['selection']['frame_index']==2
    assert 'restarted' in recovered.jobs[job_id]['message']
    recovered.worker.shutdown()


def test_invalid_video_can_be_deleted(portal):
    app,client,_=portal
    job=request(client,'POST','/api/jobs',json={'name':'broken.mp4','size':3}).json
    request(client,'PUT',f'/api/jobs/{job["id"]}/upload?offset=0',data=b'bad',content_type='application/octet-stream')
    request(client,'POST',f'/api/jobs/{job["id"]}/finish')
    limit=time.monotonic()+10
    while app.extensions['job_store'].jobs[job['id']]['state']!='error' and time.monotonic()<limit:
        time.sleep(.03)
    assert app.extensions['job_store'].jobs[job['id']]['state']=='error'
    assert request(client,'DELETE',f'/api/jobs/{job["id"]}').status_code==200


def test_full_chunk_over_real_http(portal):
    """Flask's test client cannot catch Waitress's exclusive body-size limit."""
    import requests
    import threading
    from trampoline_pose.portal import http_server
    app, _, tmp = portal
    server = http_server(app, 0)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    session = requests.Session()
    session.trust_env = False
    # Connect to the ephemeral socket but use the configured, allowed Host.
    base = f'http://127.0.0.1:{server.effective_port}'
    headers = {**ORIGIN, 'Host': 'localhost:8765'}
    try:
        response = session.post(base+'/api/login', headers=headers,
                                json={'key':(tmp/'inbox'/'access-key.txt').read_text()}, timeout=5)
        assert response.status_code == 200
        response = session.post(base+'/api/jobs', headers=headers,
                                json={'name':'large.mp4','size':CHUNK_SIZE}, timeout=5)
        assert response.status_code == 201
        job_id = response.json()['id']
        response = session.put(base+f'/api/jobs/{job_id}/upload?offset=0',
                               headers={**headers,'Content-Type':'application/octet-stream'},
                               data=b'v'*CHUNK_SIZE, timeout=10)
        assert response.status_code == 200 and response.json()['offset'] == CHUNK_SIZE
    finally:
        session.close()
        server.task_dispatcher.shutdown()
        server.close()
        thread.join(timeout=3)


@pytest.mark.parametrize('extension,encoder,extra', [
    ('.mp4', 'libx264', []),
    ('.mov', 'libx264', []),
    ('.mov', 'libx265', ['-tag:v','hvc1','-x265-params','pools=1:log-level=error']),
    ('.mp4', 'libx265', ['-tag:v','hvc1','-x265-params','pools=1:log-level=error']),
    ('.mov', 'libx265', ['-pix_fmt','yuv420p10le','-tag:v','hvc1','-x265-params','pools=1:log-level=error']),
    ('.m4v', 'libx264', ['-f','mp4']),
    ('.webm', 'libvpx-vp9', []),
    ('.mkv', 'libvpx-vp9', []),
    ('.avi', 'mpeg4', []),
    ('.mp4', 'libaom-av1', ['-cpu-used','8']),
    ('.mov', 'prores_ks', ['-profile:v','1']),
])
def test_common_codecs_upload_select_process_and_export(portal, extension, encoder, extra):
    import subprocess
    app, client, tmp = portal
    source = tmp / 'original.mp4'
    video(source)
    encoded = tmp / ('codec-test' + extension)
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-an','-c:v',encoder,
                    *extra,str(encoded)],check=True,capture_output=True)
    job_id = uploaded(portal, encoded)
    detail = request(client,'GET',f'/api/jobs/{job_id}').json
    assert detail['frame_count'] >= 4
    assert all(b>a for a,b in zip(detail['timestamps'],detail['timestamps'][1:]))
    assert request(client,'POST',f'/api/jobs/{job_id}/run',json={
        'frame_index':2,'bbox_xyxy':[10,10,40,60]}).status_code==202
    wait_state(app,job_id,'done')
    result = request(client,'GET',f'/api/jobs/{job_id}/results/joints.json').json
    assert result['processed_frame_count'] == detail['frame_count']
    overlay = app.extensions['job_store'].directory(detail) / 'overlay.mp4'
    np.testing.assert_allclose([f['timestamp_seconds'] for f,_ in decode(overlay)],
                               detail['timestamps'], atol=1/90000)
