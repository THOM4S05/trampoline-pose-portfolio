"""Fail closed unless CUDA, detector and ViTPose execute on the actual GPU."""
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import time
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT/'reports').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.tools/matplotlib'))

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    import cv2
    import numpy as np
    import psutil
    import torch
    import mmdet
    import mmpose
    from mmcv.ops import nms
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import init_model, inference_topdown
    from mmengine import Config
    from mmengine.registry import DefaultScope
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback permitted'
    assert torch.cuda.get_device_capability() == (6, 1), 'Expected GTX 1070 capability 6.1'
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(512, 512, device='cuda')
    product = x @ x.T
    torch.cuda.synchronize()
    assert product.is_cuda and torch.isfinite(product).all()
    boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]], device='cuda')
    _, keep = nms(boxes, torch.tensor([.9,.8],device='cuda'), .5)
    assert keep.is_cuda and len(keep) == 1
    detector_cfg = pathlib.Path(mmdet.__file__).parent / '.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    pose_cfg = pathlib.Path(mmpose.__file__).parent / '.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py'
    dc = Config.fromfile(str(detector_cfg))
    dc.model.backbone.init_cfg = None
    pc = Config.fromfile(str(pose_cfg))
    pc.model.backbone.init_cfg = None
    detector = init_detector(dc, str(ROOT/'models/checkpoints/rtmdet-tiny.pth'), device='cuda:0')
    pose = init_model(pc, str(ROOT/'models/checkpoints/vitpose-small.pth'), device='cuda:0')
    assert next(detector.parameters()).is_cuda and next(pose.parameters()).is_cuda
    audits={}
    for name,model,filename,allowed in [
        ('detector',detector,'rtmdet-tiny.pth',{'data_preprocessor.mean','data_preprocessor.std'}),
        ('pose',pose,'vitpose-small.pth',{'backbone.cls_token'})]:
        weights=torch.load(ROOT/'models/checkpoints'/filename,map_location='cpu')['state_dict']
        positional_adaptation=False
        if name=='pose':
            before=tuple(weights['backbone.pos_embed'].shape)
            model.backbone._prepare_pos_embed(weights,'backbone.')
            positional_adaptation=before!=tuple(weights['backbone.pos_embed'].shape)
        state=model.state_dict()
        missing=sorted(set(state)-set(weights))
        unexpected=sorted(set(weights)-set(state))
        mismatched=[key for key in set(weights)&set(state) if weights[key].shape!=state[key].shape]
        assert not missing and not mismatched, f'{name} missing {missing}; mismatched {mismatched}'
        assert set(unexpected)<=allowed, f'{name} unexpected parameters outside known unused fields'
        audits[name]=dict(missing_keys=missing,shape_mismatches=mismatched,known_unused_keys=unexpected,
                          framework_removed_unused_cls_position_embedding=positional_adaptation)
        del weights
    image = cv2.imread(str(ROOT/'models/smoke.jpg'))
    assert image is not None
    with torch.inference_mode():
        with DefaultScope.overwrite_default_scope('mmdet'):
            pred = inference_detector(detector, image).pred_instances
        mask = (pred.labels == 0) & (pred.scores > .3)
        people = pred.bboxes[mask].cpu().numpy()
        assert len(people), 'No person in smoke image'
        t = time.perf_counter()
        with DefaultScope.overwrite_default_scope('mmpose'):
            result = inference_topdown(pose, image, people[:1])[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
    points = result.pred_instances.keypoints[0]
    scores = result.pred_instances.keypoint_scores[0]
    assert points.shape == (17, 2) and np.isfinite(points).all()
    assert np.count_nonzero(scores > .3) >= 8, 'Insufficient useful keypoints'
    cv2.rectangle(image, tuple(people[0,:2].astype(int)),tuple(people[0,2:].astype(int)),(0,255,255),2)
    for point, score in zip(points,scores):
        if score > .3:
            cv2.circle(image,tuple(point.astype(int)),4,(0,255,0),-1)
    cv2.imwrite(str(ROOT/'reports/gpu-smoke.jpg'), image)
    report = dict(passed=True, python=sys.version, platform=platform.platform(),
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        compiled_architectures=torch.cuda.get_arch_list(), cuda_runtime=torch.version.cuda,
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        torch_cuda_operation=True, mmcv_cuda_nms=True, detector_device=str(next(detector.parameters()).device),
        pose_device=str(next(pose.parameters()).device), detected_people=len(people),
        checkpoint_parameter_audits=audits,
        pose_keypoints=points.tolist(), pose_scores=scores.tolist(), pose_inference_seconds=seconds,
        total_seconds=time.perf_counter()-start, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        host_rss_bytes=psutil.Process().memory_info().rss,
        pose_config_sha256=sha(pose_cfg), detector_config_sha256=sha(detector_cfg),
        checkpoints={p.name:sha(p) for p in (ROOT/'models/checkpoints').glob('*.pth')},
        dependencies={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','mmcv','mmengine','mmdet','mmpose','mmpretrain']},
        test_image='Official MMPose COCO test image 000000000785.jpg; trampoline sample verification is separate')
    (ROOT/'reports/gpu-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
