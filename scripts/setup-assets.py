"""Explicit network setup; runtime inference never downloads weights."""
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    (ROOT/'models/checkpoints').mkdir(parents=True,exist_ok=True)
    assets = {}
    choices = [('pose','mmpose','configs/body_2d_keypoint/topdown_heatmap/coco/vitpose_coco.yml',
                'td-hm_ViTPose-small_8xb64-210e_coco-256x192','vitpose-small.pth'),
               ('detector','mmdet','configs/rtmdet/metafile.yml',
                'rtmdet_tiny_8xb32-300e_coco','rtmdet-tiny.pth')]
    for role,package,manifest,name,filename in choices:
        root = pathlib.Path(importlib.util.find_spec(package).origin).parent / '.mim'
        metadata = yaml.safe_load((root/manifest).read_text(encoding='utf-8'))
        model = next(m for m in metadata['Models'] if m['Name'] == name)
        url = model['Weights']
        checkpoint = ROOT/'models/checkpoints'/filename
        if not checkpoint.exists() or checkpoint.stat().st_size < 1024:
            temp = checkpoint.with_suffix('.download')
            subprocess.run(['curl.exe','--fail','--location','--retry','3','--connect-timeout','30','--output',str(temp),url],check=True)
            temp.replace(checkpoint)
        expected = {'pose':'62d7a7128f13a63ec63d7319ea705c9822883ebf75130832cccdcb54313ddb13',
                    'detector':'78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb'}[role]
        if sha(checkpoint)!=expected:
            raise RuntimeError(f'Checkpoint integrity failed: {checkpoint}; replace it through explicit setup')
        config = root/model['Config']
        assets[role] = dict(package=package,name=name,config_relative=model['Config'],
                            checkpoint=str(checkpoint.relative_to(ROOT)).replace('\\','/'),
                            checkpoint_sha256=sha(checkpoint),upstream_url=url,
                            config_sha256=sha(config))
        shutil.copytree(root/'configs',ROOT/'models'/f'{package}-configs',dirs_exist_ok=True)
    image = ROOT/'models/smoke.jpg'
    if not image.exists():
        urllib.request.urlretrieve('https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/tests/data/coco/000000000785.jpg',image)
    (ROOT/'models/manifest.json').write_text(json.dumps(assets,indent=2),encoding='utf-8')
    print(json.dumps(assets,indent=2))

if __name__ == '__main__':
    main()
