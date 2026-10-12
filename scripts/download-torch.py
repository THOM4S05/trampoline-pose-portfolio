"""Fetch the pinned Windows CUDA wheel with resumable ranges and upstream SHA256."""
import concurrent.futures
import hashlib
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = 'torch-2.0.1+cu118-cp311-cp311-win_amd64.whl'
DEST = ROOT / '.tools/wheels' / NAME

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    html = urllib.request.urlopen('https://download.pytorch.org/whl/cu118/torch/', timeout=30).read().decode()
    match = re.search(r'href="([^"]*torch-2\.0\.1(?:%2B|\+)cu118-cp311-cp311-win_amd64\.whl)#sha256=([a-f0-9]+)"', html)
    if not match:
        raise RuntimeError('Pinned official wheel/hash not found')
    url, expected = match.groups()
    url = url.replace('https://download-r2.pytorch.org/', 'https://download.pytorch.org/')
    if url.startswith('/'):
        url = 'https://download.pytorch.org' + url
    probe = subprocess.check_output(['curl.exe','-sS','--fail','--location','--range','0-0','--max-time','30','--dump-header','-','--output',str(ROOT/'.tools/torch-byte.bin'),url],text=True)
    size = int(re.search(r'content-range: bytes 0-0/(\d+)',probe,re.I).group(1))
    if not DEST.exists():
        old = list(pathlib.Path(tempfile.gettempdir()).glob('pip-unpack-*/'+NAME))
        if old:
            shutil.copyfile(max(old,key=lambda p:p.stat().st_size),DEST)
    offset = DEST.stat().st_size if DEST.exists() else 0
    print(f'Official wheel {size} bytes; preserving {offset} bytes; expected SHA256 {expected}', flush=True)
    ranges = [(start,min(start+4*1024*1024,size)-1) for start in range(offset,size,4*1024*1024)]
    def fetch(bounds):
        start,end = bounds
        part = DEST.with_suffix(f'.{start}.part')
        if part.exists() and part.stat().st_size == end-start+1:
            return part
        for attempt in range(4):
            try:
                subprocess.run(['curl.exe','--fail','--location','--range',f'{start}-{end}','--connect-timeout','20','--max-time','90','--output',str(part),url],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                if part.stat().st_size != end-start+1:
                    raise RuntimeError('Truncated range')
                print(f'Completed range {start}-{end}',flush=True)
                return part
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(fetch,ranges))
    with DEST.open('ab') as output:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source,output,1024*1024)
    h=hashlib.sha256()
    with DEST.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise RuntimeError('Official SHA256 mismatch; do not install wheel')
    for part in parts:
        part.unlink()
    print(f'Verified {DEST}',flush=True)

if __name__ == '__main__':
    main()
