"""Small OpenMMLab adapter; all weights are local and integrity checked."""
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
from .geometry import crop_geometry,udp_affine,inverse_affine,rotated_boxes_to_source,iou
from .output import sha256

class OpenMMLabBackend:
    def __init__(self,config):
        model_dir=Path(config.model_directory).resolve()
        manifest=model_dir/'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'Model manifest missing: {manifest}. Run scripts/setup-assets.py explicitly.')
        self.provenance=json.loads(manifest.read_text(encoding='utf-8'))
        os.environ.setdefault('MPLCONFIGDIR',str(model_dir.parent/'.tools/matplotlib'))
        import torch
        from mmengine import Config
        from mmdet.apis import init_detector
        from mmpose.apis import init_model
        self.torch=torch
        self.config=config
        if config.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback must be explicit')
        torch.set_num_threads(4)
        if config.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        resolved={}
        for role,record in self.provenance.items():
            cfg=model_dir/f"{record['package']}-configs"/Path(record['config_relative']).relative_to('configs')
            checkpoint=model_dir/'checkpoints'/Path(record['checkpoint']).name
            for p,key in [(cfg,'config_sha256'),(checkpoint,'checkpoint_sha256')]:
                if not p.is_file() or sha256(p)!=record[key]:
                    raise ValueError(f'Missing or changed model asset: {p}. Run explicit model setup.')
            # Hash inherited configs too, so the export identifies the actual model.
            tree=model_dir/f"{record['package']}-configs"
            record['configuration_tree_sha256']=sha256_tree(tree)
            c=Config.fromfile(str(cfg))
            c.model.backbone.init_cfg=None
            if role=='pose':
                for step in c.test_dataloader.dataset.pipeline:
                    if step['type']=='GetBBoxCenterScale':
                        step['padding']=config.crop_padding
            record['effective_config_sha256']=__import__('hashlib').sha256(c.pretty_text.encode()).hexdigest()
            resolved[role]=(c,checkpoint)
        self.detector=init_detector(resolved['detector'][0],str(resolved['detector'][1]),device=config.device)
        self.pose=init_model(resolved['pose'][0],str(resolved['pose'][1]),device=config.device)
        self.input_size=tuple(self.pose.cfg.codec.input_size)
        self.provenance['runtime']=dict(device=config.device,precision='FP32',batch_size=1,
            pose_flip_test=bool(self.pose.cfg.model.test_cfg.get('flip_test',False)),
            detector_resolution=[640,640],init_backbone_pretraining_disabled=True,
            detector_rotations=list(config.detector_rotations),rotation_duplicate_nms_iou=.4,
            load_notes=['Detector checkpoint contains unused normalization buffers',
                        'ViTPose checkpoint contains unused cls_token; backbone uses with_cls_token=False'])

    def detect(self,image):
        from mmdet.apis import inference_detector
        from mmengine.registry import DefaultScope
        all_boxes=[]
        all_scores=[]
        h,w=image.shape[:2]
        with self.torch.inference_mode(), DefaultScope.overwrite_default_scope('mmdet'):
            for rotation in self.config.detector_rotations:
                rotated=np.ascontiguousarray(np.rot90(image,rotation//90)) if rotation else image
                result=inference_detector(self.detector,rotated).pred_instances
                mask=(result.labels==0)&(result.scores>=self.config.detection_score)
                boxes=rotated_boxes_to_source(result.bboxes[mask].cpu().numpy(),rotation,w,h)
                scores=result.scores[mask].cpu().numpy()
                all_boxes.extend(boxes)
                all_scores.extend(scores.tolist())
        boxes=np.asarray(all_boxes,dtype=float).reshape(-1,4)
        scores=np.asarray(all_scores,dtype=float)
        keep=[]
        for i in np.argsort(-scores):
            if all(iou(boxes[i],boxes[j])<=.4 for j in keep):
                keep.append(i)
        return boxes[keep],scores[keep]

    def estimate(self,image,box):
        from mmpose.apis import inference_topdown
        from mmengine.registry import DefaultScope
        with self.torch.inference_mode(), DefaultScope.overwrite_default_scope('mmpose'):
            result=inference_topdown(self.pose,image,np.array([box],dtype=np.float32))[0]
        # MMPose returns full-image coordinates, including its inverse UDP transform.
        # They must not be multiplied by another inverse transform.
        meta=result.metainfo
        crop=crop_geometry(box,self.config.crop_padding,self.input_size)
        center=meta.get('input_center')
        scale=meta.get('input_scale')
        if center is not None and scale is not None:
            center=np.asarray(center).reshape(-1,2)[0]
            scale=np.asarray(scale).reshape(-1,2)[0]
            matrix=udp_affine(center,scale,self.input_size)
            crop.update(center=center.tolist(),scale=scale.tolist(),source_to_input_affine=matrix.tolist(),
                        input_to_source_affine=inverse_affine(matrix).tolist(),
                        padded_bbox_xyxy=np.r_[center-scale/2,center+scale/2].tolist())
        crop['backend_coordinates']='MMPose predicts and returns inverse-mapped full-frame pixels'
        return result.pred_instances.keypoints[0],result.pred_instances.keypoint_scores[0],crop

    def metrics(self):
        if self.config.device.startswith('cuda'):
            self.torch.cuda.synchronize()
            return dict(peak_gpu_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                        peak_gpu_reserved_bytes=self.torch.cuda.max_memory_reserved())
        return dict(peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None)

def sha256_tree(root):
    import hashlib
    digest=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).replace('\\','/').encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def versions():
    """Record absent inference packages explicitly for injected CPU backends."""
    optional = {'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmpose', 'mmpretrain'}
    result = {}
    for name in ['torch','torchvision','mmcv','mmengine','mmdet','mmpose','mmpretrain','numpy','scipy','opencv-python','av','jsonschema','psutil']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name not in optional:
                raise
            result[name] = None
    return result
