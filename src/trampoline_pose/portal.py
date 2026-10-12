"""Personal video inbox. Bind only to loopback; use Tailscale Serve for HTTPS.

The browser selects JPEGs decoded with the same rotation, timing and frame
indices as inference. No browser seek-time -> nominal-FPS conversion is used.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hmac
import json
import logging
import math
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from urllib.parse import urlsplit
import uuid

from flask import Flask, abort, jsonify, request, send_file
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.exceptions import HTTPException

from .types import AthleteSelection, TrackerConfig

CHUNK_SIZE = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 2 * 1024**3
MAX_FRAMES = 20000
COOKIE = 'trampoline_session'
SESSION_SECONDS = 12 * 60 * 60
BUSY = {'preparing', 'queued', 'processing'}
LOG = logging.getLogger(__name__)


class StorageFull(HTTPException):
    code = 507
    description = 'Not enough free disk space'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
    # Windows readers / virus scanners can briefly hold a file without delete
    # sharing. Progress persistence should tolerate that transient lock.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}')
    return value


def align_reference(reference, skills):
    """Match saved reference time spans without feeding labels to analysis."""
    rows = []
    for row in reference.get('skills', []):
        item = dict(row)
        if row.get('start_seconds') is not None and row.get('end_seconds') is not None:
            overlap = lambda skill: max(0., min(row['end_seconds'], skill['visible_end_seconds']) - max(row['start_seconds'], skill['visible_start_seconds']))
            best = max(skills, key=overlap) if skills else None
            item['reference_flight_index'] = row.get('reference_flight_index', row.get('skill_index'))
            item['skill_index'] = best['skill_index'] if best and overlap(best) > 0 else None
        rows.append(item)
    return {**reference, 'skills': rows}


class JobStore:
    def __init__(self, root, config, tracker_factory=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.tracker_factory = tracker_factory
        self.lock = threading.RLock()
        # One worker owns video preparation and GPU work: no concurrent models.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='video-worker')
        self.jobs = {}
        self.tracker = None
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if not re.fullmatch(r'[a-f0-9]{32}', job['id']) or path.parent.name != job['id']:
                    continue
                self.jobs[job['id']] = job
                if job['state'] in BUSY:
                    job['state'] = 'ready' if job.get('frame_count') else 'error'
                    job['message'] = 'PC restarted. Review your selection and start again.'
                    self.save(job)
                if job.get('analysis', {}).get('state') in {'queued', 'running'}:
                    job['analysis'] = dict(state='error', message='PC restarted during skill analysis. Run it again.')
                    self.save(job)
            except (ValueError, KeyError):
                LOG.warning('Ignoring invalid job metadata: %s', path)

    def directory(self, job):
        return self.root / job['id']

    def source(self, job):
        return self.directory(job) / ('source' + job['extension'])

    def save(self, job):
        job['updated_at'] = now()
        atomic_json(self.directory(job) / 'job.json', job)

    def update(self, job, **values):
        with self.lock:
            job.update(values)
            self.save(job)

    def get(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id) or job_id not in self.jobs:
            abort(404, 'Video not found')
        return self.jobs[job_id]

    def create(self, data):
        size = integer(data.get('size'), 'Video size', 1)
        if size > MAX_VIDEO_BYTES:
            abort(413, 'Videos must be no larger than 2 GiB')
        name = data.get('name')
        if not isinstance(name, str) or not name.strip() or len(name) > 240:
            raise ValueError('Choose a video with a filename under 240 characters')
        # Display name is never used as a filesystem path.
        name = name.replace('\\', '/').rsplit('/', 1)[-1]
        extension = Path(name).suffix.lower()
        if extension not in {'.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'}:
            raise ValueError('Choose an MP4, MOV, M4V, MKV, AVI or WebM video')
        with self.lock:
            if len(self.jobs) >= 30 or sum(j['size'] for j in self.jobs.values()) + size > 20 * 1024**3:
                abort(409, 'Inbox is full. Delete old videos before uploading more.')
            reserved = sum(j['size'] - self.offset(j) for j in self.jobs.values() if j['state'] == 'uploading')
            if shutil.disk_usage(self.root).free < reserved + size + 2 * 1024**3:
                raise StorageFull('Not enough free disk space; keep at least 2 GiB free')
            job = dict(id=uuid.uuid4().hex, name=name, extension=extension, size=size,
                       created_at=now(), state='uploading', message='Waiting for upload',
                       frame_count=0, selection=None, progress=None)
            self.directory(job).mkdir()
            (self.directory(job) / 'upload.part').touch()
            self.jobs[job['id']] = job
            self.save(job)
            return job

    def offset(self, job):
        path = self.directory(job) / 'upload.part'
        return path.stat().st_size if path.exists() else 0

    def summary(self, job):
        return {**job, 'offset': self.offset(job) if job['state'] == 'uploading' else job['size']}

    def prepare(self, job):
        try:
            import cv2
            from .video import probe, decode
            meta = probe(self.source(job))
            if max(meta['display_width'], meta['display_height']) > 8192 or meta['display_width'] * meta['display_height'] > 34000000:
                raise ValueError('Video resolution exceeds the supported 8K limit')
            frames = self.directory(job) / 'frames'
            frames.mkdir(exist_ok=True)
            timestamps = []
            for timing, image in decode(self.source(job), meta, max_frames=MAX_FRAMES + 1):
                i = timing['frame_index']
                if i >= MAX_FRAMES:
                    raise ValueError('Video exceeds 20,000 frames. Trim it into shorter clips.')
                if i % 100 == 0 and shutil.disk_usage(self.root).free < 1024**3:
                    raise ValueError('Less than 1 GiB disk space remains. Free space and retry.')
                scale = min(1., 960 / max(image.shape[:2]))
                preview = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
                if not cv2.imwrite(str(frames / f'{i}.jpg'), preview, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    raise ValueError('Unable to save preview frame')
                timestamps.append(timing['timestamp_seconds'])
                if i % 30 == 0:
                    self.update(job, progress=dict(stage='Preparing frames', completed=i+1, total=meta['declared_frame_count']))
            if not timestamps:
                raise ValueError('Video decoded no frames')
            atomic_json(self.directory(job) / 'frames.json', timestamps)
            # Do not send server source paths to the browser.
            self.update(job, state='ready', message='Choose a clear frame, draw a box around the gymnast, then select Start tracking.',
                        frame_count=len(timestamps), width=meta['display_width'], height=meta['display_height'],
                        duration=timestamps[-1], progress=None)
        except Exception as exc:
            self.fail(job, exc, 'Could not prepare this video')

    def fail(self, job, exc, message):
        LOG.exception('%s: %s', message, job['id'])
        # Keep detailed paths / decoder / model internals in local logs only.
        detail = str(exc) if isinstance(exc, (ValueError, InterruptedError)) else 'See the portal log on the processing PC for details.'
        self.update(job, state='error', message=f'{message}. {detail}', progress=None)

    def selection(self, job, data):
        frame = integer(data.get('frame_index'), 'Seed frame')
        if not job['frame_count'] or frame >= job['frame_count']:
            raise ValueError('Choose an available frame from this video')
        box = data.get('bbox_xyxy')
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError('Draw a valid box around the gymnast')
        selection = AthleteSelection(frame, tuple(box))
        x1, y1, x2, y2 = selection.bbox_xyxy
        if not (0 <= x1 < x2 <= job['width'] and 0 <= y1 < y2 <= job['height']):
            raise ValueError('Box must lie inside the video frame')
        if x2-x1 < 4 or y2-y1 < 4:
            raise ValueError('Box is too small. Include the gymnast’s entire body.')
        return dict(frame_index=frame, bbox_xyxy=list(selection.bbox_xyxy))

    def process(self, job):
        try:
            self.update(job, state='processing', message='Loading pose models', progress=None)
            if self.tracker is None:
                from .api import PoseTracker
                self.tracker = (self.tracker_factory or PoseTracker)(self.config)
            selection = AthleteSelection(**job['selection'])
            def progress(stage, count, total):
                if count == 1 or count % 15 == 0 or count == total:
                    self.update(job, message='Tracking your gymnast', progress=dict(
                        stage={'detect': 'Finding the gymnast', 'pose': 'Estimating joints'}.get(stage, stage),
                        completed=count, total=job['frame_count']))
            result = self.tracker.process_video(self.source(job), selection, progress=progress,
                                               cache_path=self.directory(job) / 'detections.json')
            self.update(job, message='Writing results and review video', progress=None)
            result.save_json(self.directory(job) / 'joints.json')
            result.save_csv(self.directory(job) / 'joints.csv')
            result.save_video(self.directory(job) / 'overlay.mp4')
            self.update(job, state='done', message='Results ready. Review the overlay for tracking errors.', progress=None)
        except Exception as exc:
            self.fail(job, exc, 'Processing stopped')

    def analyze_skills(self, job):
        """Run the pose-only baseline on this job's finished tracker export."""
        try:
            from .skill_identification import analyze, write_csv
            self.update(job, analysis=dict(state='running', message='Estimating skills on this PC'))
            source = self.directory(job) / 'joints.json'
            data = json.loads(source.read_text(encoding='utf-8'))
            result = analyze(data)
            self.update(job, analysis=dict(state='running', message='Creating body shape previews'))
            from .skill_images import save_shape_images
            images = save_shape_images(data, result, self.source(job), self.directory(job) / 'skill-images')
            rows = [{k: s[k] for k in ('skill_index', 'start_frame', 'end_frame', 'start_seconds',
                                       'end_seconds', 'start_clipped', 'end_clipped',
                                       'supplied_skill_identity', 'predicted_skill_identity', 'review_status')}
                    | {'rotation_' + k if k == 'reasons' else k: v for k, v in s['rotation'].items()}
                    | {'shape_' + k if k == 'reasons' else k: v for k, v in s['execution'].items()}
                    for s in result['skills']]
            directory = self.directory(job)
            json_tmp = directory / 'skills.pending.json'
            csv_tmp = directory / 'skills.pending.csv'
            json_tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            write_csv(csv_tmp, rows)
            json_tmp.replace(directory / 'skills.json')
            csv_tmp.replace(directory / 'skills.csv')
            atomic_json(directory / 'skill-images.json', images)
            self.update(job, analysis=dict(state='done', message='Skill estimates ready for review',
                                           skill_count=len(result['skills']), images_version=2, analysis_version=10))
        except Exception as exc:
            LOG.exception('Skill analysis stopped: %s', job['id'])
            detail = str(exc) if isinstance(exc, (ValueError, KeyError, TypeError)) else 'See the portal log on the processing PC.'
            self.update(job, analysis=dict(state='error', message=f'Skill analysis stopped. {detail}'))


def create_app(root, config=None, public_url=None, port=8765, tracker_factory=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    public = urlsplit(public_url) if public_url else None
    if public and (public.scheme != 'https' or not public.hostname or public.path not in ('', '/') or public.query or public.fragment or public.username or public.password):
        raise ValueError('--public-url must be your private HTTPS origin, without a path')
    origins = {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}
    if public:
        origins.add(f'https://{public.netloc}')
    app = Flask(__name__, static_folder='web', static_url_path='/assets')
    app.config.update(MAX_CONTENT_LENGTH=CHUNK_SIZE, TRUSTED_HOSTS=['127.0.0.1', 'localhost'] + ([public.hostname] if public else []))
    key_path = root / 'access-key.txt'
    if not key_path.exists():
        key_path.write_text(secrets.token_urlsafe(32), encoding='utf-8')
    key = key_path.read_text(encoding='utf-8').strip()
    if len(key) < 32:
        raise ValueError('Access key must have at least 32 characters')
    signer = URLSafeTimedSerializer(key, salt='trampoline-portal-session-v1')
    store = JobStore(root / 'jobs', config or TrackerConfig(), tracker_factory)
    app.extensions['job_store'] = store
    attempts = deque()
    login_lock = threading.Lock()

    @app.before_request
    def guard():
        if request.host not in {urlsplit(origin).netloc for origin in origins}:
            abort(400, 'Unrecognised portal address')
        # Cross-site forms cannot send this header. No CORS is enabled.
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            if request.headers.get('X-Portal-Request') != '1' or request.headers.get('Origin') not in origins:
                abort(403, 'Open the portal directly to make changes')
        if request.path.startswith('/api/') and request.path != '/api/login':
            try:
                if signer.loads(request.cookies.get(COOKIE, ''), max_age=SESSION_SECONDS) != 'owner':
                    abort(401, 'Unlock the inbox first')
            except BadSignature:
                abort(401, 'Unlock the inbox first')

    @app.after_request
    def security_headers(response):
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(ValueError)
    def value_error(error):
        return jsonify(error=str(error)), 400

    def body():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object')
        return data

    @app.get('/')
    def index():
        return send_file(Path(app.static_folder) / 'index.html')

    @app.get('/skills')
    def skills_page():
        return send_file(Path(app.static_folder) / 'skills.html')

    @app.post('/api/login')
    def login():
        with login_lock:
            stamp = time.monotonic()
            while attempts and stamp - attempts[0] > 60:
                attempts.popleft()
            if len(attempts) >= 5:
                abort(429, 'Too many attempts. Wait a minute and try again.')
            supplied = body().get('key', '')
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(), key.encode()):
                attempts.append(stamp)
                abort(401, 'That access key did not match')
        response = jsonify(ok=True)
        response.set_cookie(COOKIE, signer.dumps('owner'), httponly=True,
                            secure=bool(public and request.host.split(':')[0] == public.hostname),
                            samesite='Strict', max_age=SESSION_SECONDS)
        return response

    @app.post('/api/logout')
    def logout():
        response = jsonify(ok=True)
        response.delete_cookie(COOKIE)
        return response

    @app.get('/api/jobs')
    def jobs():
        with store.lock:
            return jsonify(jobs=[store.summary(j) for j in sorted(store.jobs.values(), key=lambda j: j['created_at'], reverse=True)], chunk_size=CHUNK_SIZE)

    @app.post('/api/jobs')
    def new_job():
        return jsonify(store.summary(store.create(body()))), 201

    @app.get('/api/jobs/<job_id>')
    def job_detail(job_id):
        with store.lock:
            job = store.get(job_id)
            data = store.summary(job)
            if job['frame_count']:
                data['timestamps'] = json.loads((store.directory(job) / 'frames.json').read_text(encoding='utf-8'))
            return jsonify(data)

    @app.put('/api/jobs/<job_id>/upload')
    def upload(job_id):
        try:
            offset = int(request.args.get('offset', '-1'))
        except ValueError:
            raise ValueError('Invalid upload offset')
        if request.mimetype != 'application/octet-stream':
            abort(415, 'Expected video bytes')
        chunk = request.get_data()
        if not chunk:
            raise ValueError('Empty upload chunk')
        with store.lock:
            job = store.get(job_id)
            if job['state'] != 'uploading':
                abort(409, 'This upload is already finished')
            path = store.directory(job) / 'upload.part'
            current = store.offset(job)
            # Retry after a lost response is safe only if the bytes agree.
            if 0 <= offset < current and offset + len(chunk) <= current:
                with path.open('rb') as f:
                    f.seek(offset)
                    if f.read(len(chunk)) == chunk:
                        return jsonify(offset=current)
            if offset != current or current + len(chunk) > job['size']:
                abort(409, 'Upload offset changed. Retry this upload.')
            if shutil.disk_usage(store.root).free < len(chunk) + 1024**3:
                raise StorageFull()
            with path.open('ab') as f:
                f.write(chunk)
            return jsonify(offset=store.offset(job))

    @app.post('/api/jobs/<job_id>/finish')
    def finish(job_id):
        with store.lock:
            job = store.get(job_id)
            if job['state'] != 'uploading':
                abort(409, 'This upload is already finished')
            if store.offset(job) != job['size']:
                abort(409, 'Upload is incomplete')
            (store.directory(job) / 'upload.part').replace(store.source(job))
            store.update(job, state='preparing', message='Preparing the frame slider')
            store.worker.submit(store.prepare, job)
            return jsonify(store.summary(job)), 202

    @app.post('/api/jobs/<job_id>/prepare')
    def retry_prepare(job_id):
        with store.lock:
            job = store.get(job_id)
            if job['state'] != 'error' or job['frame_count'] or not store.source(job).exists():
                abort(409, 'This video cannot be prepared again')
            store.update(job, state='preparing', message='Preparing the frame slider')
            store.worker.submit(store.prepare, job)
            return jsonify(store.summary(job)), 202

    @app.get('/api/jobs/<job_id>/frames/<int:frame>.jpg')
    def frame_image(job_id, frame):
        with store.lock:
            job = store.get(job_id)
            if not 0 <= frame < job['frame_count']:
                abort(404, 'Frame not found')
            return send_file(store.directory(job) / 'frames' / f'{frame}.jpg', mimetype='image/jpeg')

    @app.post('/api/jobs/<job_id>/selection')
    def save_selection(job_id):
        with store.lock:
            job = store.get(job_id)
            if job['state'] not in {'ready', 'error'}:
                abort(409, 'Selection is locked while queued, processing or finished')
            selection = store.selection(job, body())
            store.update(job, selection=selection)
            return jsonify(store.summary(job))

    @app.post('/api/jobs/<job_id>/run')
    def run(job_id):
        with store.lock:
            job = store.get(job_id)
            if job['state'] not in {'ready', 'error'}:
                abort(409, 'Video is already queued, processing or finished')
            selection = store.selection(job, body())
            store.update(job, state='queued', selection=selection, message='Waiting for the processing PC', progress=None)
            store.worker.submit(store.process, job)
            return jsonify(store.summary(job)), 202

    @app.get('/api/jobs/<job_id>/results/<filename>')
    def result_file(job_id, filename):
        with store.lock:
            job = store.get(job_id)
            pose_file = filename in {'joints.json', 'joints.csv', 'overlay.mp4'} and job['state'] == 'done'
            skill_file = filename in {'skills.json', 'skills.csv'} and job.get('analysis', {}).get('state') == 'done'
            if not (pose_file or skill_file):
                abort(404, 'Result is not ready')
            return send_file(store.directory(job) / filename, as_attachment=request.args.get('download') == '1' or filename != 'overlay.mp4', conditional=True)

    @app.post('/api/jobs/<job_id>/analysis')
    def start_analysis(job_id):
        with store.lock:
            job = store.get(job_id)
            if job['state'] != 'done' or not (store.directory(job) / 'joints.json').is_file():
                abort(409, 'Track this video before running skill analysis')
            previous = job.get('analysis', {})
            redo = body().get('redo', False) if request.is_json else False
            if not isinstance(redo, bool):
                abort(400, 'redo must be true or false')
            if previous.get('state') in {'queued', 'running'} or (not redo and previous.get('state') == 'done' and previous.get('images_version', 0) >= 2 and previous.get('analysis_version', 0) >= 9):
                abort(409, 'Skill analysis is already queued or complete')
            store.update(job, analysis=dict(state='queued', message='Waiting for the processing PC'))
            store.worker.submit(store.analyze_skills, job)
            return jsonify(analysis=job['analysis']), 202

    @app.get('/api/jobs/<job_id>/analysis')
    def analysis_detail(job_id):
        with store.lock:
            job = store.get(job_id)
            status = job.get('analysis', dict(state='not_started', message='Select Run skill analysis to estimate rotations and body shape.'))
            if status['state'] != 'done':
                return jsonify(analysis=status)
            result = json.loads((store.directory(job) / 'skills.json').read_text(encoding='utf-8'))
            image_path = store.directory(job) / 'skill-images.json'
            images = json.loads(image_path.read_text(encoding='utf-8')) if image_path.is_file() else {}
            reference_path = store.directory(job) / 'skill-reference.json'
            reference = json.loads(reference_path.read_text(encoding='utf-8')) if reference_path.is_file() else {}
            reference = align_reference(reference, result['skills'])
            return jsonify(analysis=status, skills=result['skills'], contacts=result['contacts'], shape_images=images, reference=reference)

    @app.get('/api/jobs/<job_id>/skill-images/<int:skill_index>.jpg')
    def skill_image(job_id, skill_index):
        with store.lock:
            job = store.get(job_id)
            if job.get('analysis', {}).get('state') != 'done':
                abort(404, 'Skill image is not ready')
            manifest_path = store.directory(job) / 'skill-images.json'
            if not manifest_path.is_file() or not json.loads(manifest_path.read_text(encoding='utf-8')).get(str(skill_index), {}).get('available'):
                abort(404, 'No tightest-shape image for this skill')
            path = store.directory(job) / 'skill-images' / f'{skill_index}.jpg'
            if not path.is_file():
                abort(404, 'No tightest-shape image for this skill')
            return send_file(path, mimetype='image/jpeg', conditional=True)

    @app.delete('/api/jobs/<job_id>')
    def delete(job_id):
        with store.lock:
            job = store.get(job_id)
            if job['state'] in BUSY or job.get('analysis', {}).get('state') in {'queued', 'running'}:
                abort(409, 'Wait for this job to finish before deleting it')
            directory = store.directory(job).resolve()
            if directory.parent != store.root:
                abort(400, 'Invalid job directory')
            shutil.rmtree(directory)
            del store.jobs[job_id]
            return jsonify(ok=True)

    return app


def http_server(app, port):
    from waitress import create_server
    # Waitress rejects bodies equal to its limit; Flask accepts equality.
    return create_server(app, host='127.0.0.1', port=port, threads=6,
                         max_request_body_size=CHUNK_SIZE + 1024,
                         channel_timeout=120, clear_untrusted_proxy_headers=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Private trampoline video inbox')
    parser.add_argument('--data-dir', default='.portal')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--public-url', help='Exact private Tailscale HTTPS origin')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    app = create_app(args.data_dir, TrackerConfig.from_yaml(args.config), args.public_url, args.port)
    print(f'Video inbox: http://127.0.0.1:{args.port}', flush=True)
    print(f'Access key: open {Path(args.data_dir).resolve() / "access-key.txt"}', flush=True)
    server = http_server(app, args.port)
    try:
        server.run()
    finally:
        server.close()
        app.extensions['job_store'].worker.shutdown(wait=True)


if __name__ == '__main__':
    main()
