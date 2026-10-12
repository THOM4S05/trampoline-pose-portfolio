/* No third-party scripts, uploads or analytics. Coordinates stay in source pixels. */
'use strict';
const $ = id => document.getElementById(id);
const state = {jobs: [], job: null, image: null, frame: 0, box: null, ready: false,
  request: 0, uploading: false, corners: false, anchor: null, drag: null, locked: true};
const labels = {uploading:'Uploading', preparing:'Preparing frames', ready:'Select gymnast', queued:'Queued', processing:'Tracking', done:'Ready to review', error:'Needs attention'};
let noticeTimer, frameTimer, pollBusy = false;
function notice(text) { $('notice').textContent = text; $('notice').hidden = false; clearTimeout(noticeTimer); noticeTimer = setTimeout(() => $('notice').hidden = true, 8000); }
async function api(path, options={}) {
  const response = await fetch('/api' + path, {...options, headers:{'X-Portal-Request':'1', ...(options.body && typeof options.body === 'string' ? {'Content-Type':'application/json'} : {}), ...options.headers}});
  const data = await response.json();
  if (!response.ok) { if(response.status === 401) lock(); throw new Error(data.error || 'Request failed'); }
  return data;
}
function lock() { window.skillAnalysis.clear(); state.locked = true; state.request++; state.ready=false; state.job=null; state.jobs=[]; $('job-list').replaceChildren(); $('result-video').removeAttribute('src'); $('result-video').load(); $('workspace').hidden=true; $('logout').hidden=true; $('login-panel').hidden=false; }
function unlock() { state.locked=false; $('login-panel').hidden=true; $('workspace').hidden=false; $('logout').hidden=false; }
function editable() { return state.job && ['ready','error'].includes(state.job.state); }
function validBox() { const b=state.box, j=state.job; return b && j && b.every(Number.isFinite) && b[0]>=0 && b[1]>=0 && b[2]<=j.width && b[3]<=j.height && b[2]-b[0]>=4 && b[3]-b[1]>=4; }
function selection() { return {frame_index:state.frame,bbox_xyxy:state.box}; }
function draftKey() { return 'flightroom-selection-' + state.job.id; }
function saveDraft() { if(state.job) try { localStorage.setItem(draftKey(),JSON.stringify(selection())); } catch {} }
function updateControls(syncCoordinates=true) {
  const can=editable();
  $('start-processing').disabled=!(can && state.ready && validBox());
  for(const id of ['clear-box','corner-mode','apply-box']) $(id).disabled=!can || !state.ready;
  $('frame-slider').disabled=!can; $('frame-number').disabled=!can;
  $('previous-frame').disabled=!can || state.frame<=0;
  $('next-frame').disabled=!can || state.frame>=state.job.frame_count-1;
  $('box-status').textContent=validBox() ? `Box selected · ${Math.round(state.box[2]-state.box[0])} × ${Math.round(state.box[3]-state.box[1])} px` : (state.anchor ? 'Now tap the opposite corner' : 'No box selected');
  ['box-x1','box-y1','box-x2','box-y2'].forEach((id,i)=>{ if(syncCoordinates) $(id).value=state.box ? Math.round(state.box[i]) : ''; $(id).disabled=!can; });
}
function draw() {
  const canvas=$('frame-canvas'),ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!state.image || !state.job) return;
  ctx.drawImage(state.image,0,0,canvas.width,canvas.height);
  const sx=canvas.width/state.job.width, sy=canvas.height/state.job.height;
  if(state.box) {
    const [x1,y1,x2,y2]=state.box, x=x1*sx,y=y1*sy,w=(x2-x1)*sx,h=(y2-y1)*sy;
    ctx.fillStyle='#d9f59922';ctx.fillRect(x,y,w,h);ctx.strokeStyle='#e1ff91';ctx.lineWidth=3;ctx.strokeRect(x,y,w,h);
    const handle=Math.max(9,canvas.width/canvas.getBoundingClientRect().width*9);
    for(const [hx,hy] of [[x,y],[x+w,y],[x,y+h],[x+w,y+h]]) { ctx.fillStyle='#e1ff91';ctx.fillRect(hx-handle/2,hy-handle/2,handle,handle);ctx.strokeStyle='#24382b';ctx.lineWidth=1;ctx.strokeRect(hx-handle/2,hy-handle/2,handle,handle); }
  }
  if(state.anchor) { ctx.beginPath();ctx.arc(state.anchor[0]*sx,state.anchor[1]*sy,7,0,Math.PI*2);ctx.fillStyle='#e1ff91';ctx.fill(); }
}
async function loadFrame(frame, keepBox=false) {
  if(!state.job || !state.job.frame_count) return;
  frame=Math.max(0,Math.min(state.job.frame_count-1,Math.round(frame)));
  const jobId=state.job.id, version=++state.request;
  state.frame=frame;state.ready=false;state.drag=null;state.anchor=null;
  if(!keepBox) {state.box=null;saveDraft();}
  $('frame-slider').value=frame;$('frame-number').value=frame+1;
  $('frame-time').textContent=`${(state.job.timestamps[frame] || 0).toFixed(3)} s`;
  $('frame-loading').hidden=false;updateControls();
  try {
    const img=new Image();img.src=`/api/jobs/${jobId}/frames/${frame}.jpg`;
    await img.decode();
    if(version!==state.request || state.job?.id!==jobId) return;
    state.image=img;state.ready=true;
    const canvas=$('frame-canvas');canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
    // Fix both rendered dimensions to avoid CSS max-height introducing letterboxing.
    fitCanvas();draw();updateControls();
  } catch { if(version===state.request) notice('Could not load this frame. Try another frame or unlock the inbox again.'); }
  finally { if(version===state.request) $('frame-loading').hidden=true; }
}
function fitCanvas() {
  if(!state.image) return;
  const canvas=$('frame-canvas'),parent=canvas.parentElement;
  const maxHeight=window.innerWidth<=680 ? window.innerHeight*.65 : 540;
  const scale=Math.min(1,(parent.clientWidth-16)/state.image.naturalWidth,maxHeight/state.image.naturalHeight);
  canvas.style.width=`${state.image.naturalWidth*scale}px`;canvas.style.height=`${state.image.naturalHeight*scale}px`;
}
function position(event) {
  const r=$('frame-canvas').getBoundingClientRect(),j=state.job;
  return [Math.max(0,Math.min(j.width,(event.clientX-r.left)/r.width*j.width)),Math.max(0,Math.min(j.height,(event.clientY-r.top)/r.height*j.height))];
}
function sortedBox(a,b) {return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
function boxDone() { if(!validBox()) state.box=null;draw();updateControls();saveDraft(); }
const canvas=$('frame-canvas');
canvas.addEventListener('pointerdown',event=>{
  if(!editable() || !state.ready || !event.isPrimary || event.button!==0) return;
  event.preventDefault();const p=position(event);
  if(state.corners) {if(!state.anchor) {state.anchor=p;state.box=null;} else {state.box=sortedBox(state.anchor,p);state.anchor=null;boxDone();}draw();updateControls();return;}
  canvas.setPointerCapture(event.pointerId);
  const b=state.box, r=canvas.getBoundingClientRect();
  let opposite=null;
  if(b) {
    const corners=[[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]];
    const hit=corners.findIndex(c=>Math.hypot((p[0]-c[0])/state.job.width*r.width,(p[1]-c[1])/state.job.height*r.height)<22);
    if(hit>=0) opposite=corners[3-hit];
    else if(p[0]>b[0] && p[0]<b[2] && p[1]>b[1] && p[1]<b[3]) {state.drag={mode:'move',start:p,box:[...b],pointer:event.pointerId};return;}
  }
  state.drag={mode:'draw',start:opposite || p,original:b ? [...b] : null,pointer:event.pointerId};
  state.box=sortedBox(state.drag.start,p);draw();updateControls();
});
canvas.addEventListener('pointermove',event=>{
  if(!state.drag || event.pointerId!==state.drag.pointer || !state.ready) return;
  const p=position(event),d=state.drag;
  if(d.mode==='move') {const b=d.box,dx=Math.max(-b[0],Math.min(state.job.width-b[2],p[0]-d.start[0])),dy=Math.max(-b[1],Math.min(state.job.height-b[3],p[1]-d.start[1]));state.box=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy];}
  else state.box=sortedBox(d.start,p);
  draw();updateControls();
});
canvas.addEventListener('pointerup',event=>{if(state.drag?.pointer===event.pointerId) {state.drag=null;boxDone();}});
canvas.addEventListener('pointercancel',()=>{if(state.drag) {state.box=state.drag.box || state.drag.original || null;state.drag=null;boxDone();}});
window.addEventListener('resize',()=>{fitCanvas();draw();});
function frameChanged(value) {
  if(!editable() || !Number.isFinite(value)) return;
  // Invalidate in-flight images and the box immediately, before slider debounce.
  state.request++;state.ready=false;state.box=null;state.anchor=null;state.drag=null;updateControls();draw();
  clearTimeout(frameTimer);frameTimer=setTimeout(()=>loadFrame(value),100);
}
$('frame-slider').addEventListener('input',e=>frameChanged(Number(e.target.value)));
$('frame-number').addEventListener('input',e=>{if(e.target.value!=='')frameChanged(Number(e.target.value)-1);});
$('frame-number').addEventListener('change',e=>frameChanged(Number(e.target.value)-1));
$('previous-frame').onclick=()=>{clearTimeout(frameTimer);loadFrame(state.frame-1);};
$('next-frame').onclick=()=>{clearTimeout(frameTimer);loadFrame(state.frame+1);};
$('clear-box').onclick=()=>{state.box=null;state.anchor=null;boxDone();};
$('corner-mode').onclick=()=>{state.corners=!state.corners;state.anchor=null;$('corner-mode').setAttribute('aria-pressed',String(state.corners));$('corner-mode').textContent=state.corners?'Draw by dragging':'Tap two corners instead';draw();updateControls();};
$('apply-box').onclick=()=>{const previous=state.box;state.box=['box-x1','box-y1','box-x2','box-y2'].map(id=>$(id).value===''?NaN:Number($(id).value));if(!validBox()){state.box=previous;notice('Enter a box inside the image, with right greater than left and bottom greater than top.');}boxDone();};

function renderList() {
  $('job-count').textContent=state.jobs.length;$('empty-list').hidden=!!state.jobs.length;
  $('job-list').replaceChildren(...state.jobs.map(job=>{
    const button=document.createElement('button');button.className='job-card'+(job.id===state.job?.id?' selected':'');
    const name=document.createElement('strong');name.textContent=job.name;
    const status=document.createElement('span');status.textContent=`${labels[job.state]} · ${(job.size/1024**2).toFixed(1)} MB`;
    button.append(name,status);button.onclick=()=>openJob(job.id).catch(e=>notice(e.message));return button;
  }));
}
function renderStatus() {
  const j=state.job;if(!j)return;
  $('video-name').textContent=j.name;$('job-state').textContent=labels[j.state];
  $('job-status').textContent=j.progress ? `${j.progress.stage} · ${j.progress.completed}${j.progress.total ? ' / '+j.progress.total : ''} frames` : j.message;
  $('job-progress').hidden=!['preparing','queued','processing'].includes(j.state);
  if(j.progress?.total) {$('job-progress').max=j.progress.total;$('job-progress').value=j.progress.completed;} else $('job-progress').removeAttribute('value');
  $('resume-panel').hidden=j.state!=='uploading' || state.uploading;$('retry-prepare').hidden=!(j.state==='error' && !j.frame_count);
  $('selection-panel').hidden=!j.frame_count || j.state==='done';$('results-panel').hidden=j.state!=='done';
  $('delete-job').disabled=['preparing','queued','processing'].includes(j.state) || state.uploading;
  if(j.state==='done') {
    const base=`/api/jobs/${j.id}/results/`;
    if($('result-video').getAttribute('src')!==base+'overlay.mp4') $('result-video').src=base+'overlay.mp4';
    $('download-video').href=base+'overlay.mp4?download=1';$('download-json').href=base+'joints.json';$('download-csv').href=base+'joints.csv';
  }
  updateControls(false);
}
async function openJob(id) {
  window.skillAnalysis.clear();
  clearTimeout(frameTimer);const version=++state.request;state.ready=false;
  const job=await api('/jobs/'+id);
  if(version!==state.request)return;
  state.job=job;state.box=null;state.image=null;state.frame=0;state.anchor=null;
  $('empty-editor').hidden=true;$('job-editor').hidden=false;
  $('result-video').removeAttribute('src');$('result-video').load();
  renderStatus();renderList();
  await window.skillAnalysis.select(job);
  if(job.frame_count && job.state!=='done') {
    $('frame-slider').max=job.frame_count-1;$('frame-number').max=job.frame_count;$('frame-total').textContent=`/ ${job.frame_count}`;
    let selected=job.selection;
    if(editable()) {try {selected=JSON.parse(localStorage.getItem(draftKey())) || selected;} catch {}}
    if(selected && Number.isInteger(selected.frame_index) && selected.frame_index>=0 && selected.frame_index<job.frame_count) {state.frame=selected.frame_index;state.box=selected.bbox_xyxy; if(!Array.isArray(state.box)||state.box.length!==4||!validBox())state.box=null;}
    await loadFrame(state.frame,true);
  }
}
async function refresh() {
  if(pollBusy)return;pollBusy=true;
  try {
    const data=await api('/jobs');if(state.locked)unlock();state.jobs=data.jobs;renderList();
    if(state.job) {
      const latest=state.jobs.find(j=>j.id===state.job.id);
      if(!latest) {window.skillAnalysis.clear();$('result-video').removeAttribute('src');$('result-video').load();state.job=null;state.request++;$('job-editor').hidden=true;$('empty-editor').hidden=false;}
      else if(latest.state!==state.job.state) await openJob(latest.id);
      else {Object.assign(state.job,latest);renderStatus();await window.skillAnalysis.select(state.job);}
    }
  } finally {pollBusy=false;}
}
$('login-form').onsubmit=async event=>{event.preventDefault();try{await api('/login',{method:'POST',body:JSON.stringify({key:$('access-key').value.trim()})});$('access-key').value='';unlock();await refresh();}catch(e){notice(e.message);}};
$('logout').onclick=async()=>{try{await api('/logout',{method:'POST'});lock();}catch(e){notice(e.message);}};

async function uploadFile(file) {
  if(!file || state.uploading)return;
  state.uploading=true;$('file-input').disabled=true;$('upload-progress').hidden=false;$('upload-bar').value=0;$('upload-message').textContent=`Starting upload of ${file.name}…`;
  let job;
  try {
    job=await api('/jobs',{method:'POST',body:JSON.stringify({name:file.name,size:file.size})});
    await refresh();await openJob(job.id);
    const size=8*1024*1024;
    for(let offset=0;offset<file.size;) {
      const chunk=file.slice(offset,offset+size);let data;
      for(let attempt=0;attempt<3;attempt++) {
        try{data=await api(`/jobs/${job.id}/upload?offset=${offset}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:chunk});break;}
        catch(e){if(attempt===2||state.locked)throw e;await new Promise(resolve=>setTimeout(resolve,1000*(attempt+1)));}
      }
      offset=data.offset;const percent=Math.round(offset/file.size*100);$('upload-bar').value=percent;$('upload-message').textContent=`Uploading ${file.name} · ${percent}%`;
    }
    await api(`/jobs/${job.id}/finish`,{method:'POST'});$('upload-message').textContent='Upload complete. Wait for frames to load, then select the gymnast.';await refresh();
  } catch(e) {notice(e.message);$('upload-message').textContent='Upload stopped. Choose the same file to start a new upload, then delete the unfinished entry.';}
  finally {state.uploading=false;$('file-input').disabled=false;$('file-input').value='';if(state.job)renderStatus();}
}
$('file-input').onchange=e=>uploadFile(e.target.files[0]);
for(const event of ['dragenter','dragover']) $('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.add('dragging');});
for(const event of ['dragleave','drop']) $('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.remove('dragging');});
$('drop-zone').addEventListener('drop',e=>{if(e.dataTransfer.files.length>1)notice('Upload one video at a time.');uploadFile(e.dataTransfer.files[0]);});
$('start-processing').onclick=async()=>{
  if(!state.ready||!validBox()||!editable())return;
  const jobId=state.job.id,chosen=selection();$('start-processing').disabled=true;
  try {await api(`/jobs/${jobId}/run`,{method:'POST',body:JSON.stringify(chosen)});await refresh();notice('Video queued. You can close this page and return for the results.');}
  catch(e){notice(e.message);updateControls();}
};
$('retry-prepare').onclick=async()=>{try{await api(`/jobs/${state.job.id}/prepare`,{method:'POST'});await refresh();}catch(e){notice(e.message);}};
$('delete-job').onclick=async()=>{
  const job=state.job;
  if(!confirm(`Delete “${job.name}” and its results from the processing PC? Your original file on the sending device is not affected.`))return;
  try {await api(`/jobs/${job.id}`,{method:'DELETE'});localStorage.removeItem('flightroom-selection-'+job.id);await refresh();}catch(e){notice(e.message);}
};
refresh().catch(()=>{});
setInterval(()=>{if(!state.locked)refresh().catch(e=>notice(e.message));},2500);
