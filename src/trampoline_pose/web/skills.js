(() => {
'use strict';
const $ = id => document.getElementById(id);
const embedded = !!$('embedded-skills');
let pendingSeek = null, renderedResult = null;
let jobs = [], selectedId = null, requestVersion = 0, polling = false, showContext = false, submitting = false, noticeTimer;
const stateNames = {not_started:'Not started', queued:'Queued', running:'Analysing', done:'Ready to review', error:'Needs attention'};
function notice(text) { $('notice').textContent = text; $('notice').hidden = false; clearTimeout(noticeTimer); noticeTimer = setTimeout(() => $('notice').hidden = true, 8000); }
function lock() { clearSelection(); $('result-video').removeAttribute('src'); $('result-video').load(); requestVersion++; selectedId = null; jobs = []; $('job-list').replaceChildren(); $('skill-list').replaceChildren(); $('workspace').hidden = true; $('logout').hidden = true; $('login-panel').hidden = false; }
function unlock() { $('login-panel').hidden = true; $('workspace').hidden = false; $('logout').hidden = false; }
async function api(path, options = {}) {
  const response = await fetch('/api' + path, {...options, headers:{'X-Portal-Request':'1', ...options.headers}});
  const data = await response.json();
  if (!response.ok) { if (response.status === 401) lock(); throw new Error(data.error || 'Request failed'); }
  return data;
}
function clearSelection() {
  requestVersion++; selectedId = null; pendingSeek = null; renderedResult = null;
  $('analysis-panel').hidden = true; $('skill-list').replaceChildren();
  if ($('empty-selection')) $('empty-selection').hidden = false;
}
function applySeek() {
  const video = $('result-video');
  if (!pendingSeek || video.readyState < 1) return;
  const {id, seconds} = pendingSeek;
  pendingSeek = null;
  if (id !== selectedId) return;
  video.currentTime = Math.max(0, Math.min(seconds, Number.isFinite(video.duration) ? video.duration : seconds));
}
function jumpToSkill(seconds) {
  if (!selectedId || !Number.isFinite(seconds)) return;
  pendingSeek = {id: selectedId, seconds};
  applySeek();
  $('result-video').scrollIntoView({behavior:'smooth', block:'center'});
  $('result-video').focus({preventScroll:true});
}
$('result-video').addEventListener('loadedmetadata', applySeek);
function field(dl, label, value, review = false) {
  const container = document.createElement('div'), dt = document.createElement('dt'), dd = document.createElement('dd');
  if (review) container.className = 'review';
  dt.textContent = label; dd.textContent = value; container.append(dt, dd); dl.append(container);
}
function time(value) { return value == null ? 'Outside clip / unknown' : `${Number(value).toFixed(3)} s`; }
function angle(value) { return value == null ? 'Unknown' : `${Number(value).toFixed(1)}°`; }
function frame(value) { return value == null ? 'Outside clip / unknown' : String(value + 1); }
// Names use measured fields only; reference labels never supply missing evidence.
// Single-somersault terminology: FIG Trampoline Code of Points 2025–2028.
function bestGuess(skill) {
  if (skill.seat_transition?.name) return {name:skill.seat_transition.name, note:'Estimated from contact poses and facing direction. Review the video; hidden twists cannot be excluded.'};
  const rotation = skill.rotation || {};
  const quarters = rotation.somersault_quarters ?? (Number.isFinite(rotation.somersault_count) ? rotation.somersault_count * 4 : null);
  const twist = typeof skill.twist === 'number' ? skill.twist : skill.twist?.count;
  const twists = Number.isFinite(twist) ? twist : null;
  if ((Number.isFinite(quarters) && quarters > 4) || (twists != null && twists > 1)) return {name:'Outside naming range', note:'Names cover up to 1 somersault and 1 twist.'};
  if (!Number.isInteger(quarters) || quarters < 0 || (twists != null && ![0, .5, 1].includes(twists))) return {name:'Unknown', note:'Not enough supported rotation or twist data to name this skill.'};
  if (skill.start_clipped || skill.end_clipped) return {name:'Unknown — partial skill', note:'The complete skill is not visible.'};
  const positions = {feet:'feet', standing:'feet', upright:'feet', seat:'seat', seated:'seat', front:'front', back:'back'};
  const start = positions[skill.takeoff_position], end = positions[skill.landing_position];
  const shape = {straight_like:'Straight', pike_like:'Pike', tuck_like:'Tuck', straddle_like:'Straddle'}[skill.execution?.execution_position];
  const direction = {forward:'Front', backward:'Back'}[rotation.direction];
  const twistLabel = twists === .5 ? 'Half twist' : twists === 1 ? 'Full twist' : '';
  let name;
  if (quarters === 0) {
    if (end === 'seat' && start === 'feet') name = 'Seat drop';
    else if (start && start !== 'feet' && end === 'feet') name = 'Return to feet';
    else if (start && start !== 'feet') name = `Rebound from ${start}${end ? ' to ' + end : ''}`;
    else name = shape ? `${shape} jump` : 'Jump';
    if (twistLabel) name = `${twistLabel}${name === 'Return to feet' ? ' to feet' : ' · ' + name.toLowerCase()}`;
  } else if (quarters === 4 && start === 'feet' && end === 'feet' && direction === 'Front' && twists === .5) {
    name = 'Barani';
  } else if (quarters === 4 && start === 'feet' && end === 'feet' && direction === 'Back' && twists === 1) {
    name = 'Back full';
  } else if (quarters === 1 && start === 'feet' && (end === 'front' || end === 'back')) {
    name = `${end === 'front' ? 'Front' : 'Back'} drop${twistLabel ? ' · ' + twistLabel.toLowerCase() : ''}`;
  } else if (quarters === 1 && (start === 'front' || start === 'back') && end === 'feet') {
    name = `${twistLabel ? twistLabel + ' to feet' : 'Return to feet'} from ${start}`;
  } else {
    const amount = {1:'¼ ', 2:'½ ', 3:'¾ ', 4:''}[quarters];
    name = `${amount}${direction ? direction.toLowerCase() + ' ' : ''}somersault`;
    name = name.charAt(0).toUpperCase() + name.slice(1);
    if (twistLabel) name += ` · ${twistLabel.toLowerCase()}`;
    if (start && start !== 'feet') name += ` from ${start}`;
    if (end && end !== 'feet') name += ` to ${end}`;
  }
  if (quarters > 0 && shape) name += ` (${shape.toLowerCase()})`;
  const missing = [];
  if (!start) missing.push('starting position');
  if (!end) missing.push('landing position');
  if (twists == null) missing.push('twist');
  if (!shape) missing.push('shape');
  if (quarters > 0 && !direction) missing.push('rotation direction');
  return {name, note:missing.length ? `Unconfirmed: ${missing.join(', ')}. Name may change when these are known.` : 'Estimated from starting position, landing position, shape, rotation, and twist.'};
}
function renderSkills(skills, images, reference) {
  if (!showContext && reference.routine_start_seconds != null) skills = skills.filter(skill => skill.visible_start_seconds >= reference.routine_start_seconds && (reference.routine_end_seconds == null || skill.visible_start_seconds < reference.routine_end_seconds));
  $('skill-list').replaceChildren(...skills.map((skill, index) => {
    const card = document.createElement('article'); card.className = 'skill-card';
    const title = document.createElement('h3');
    title.textContent = `Flight ${skill.skill_index}${skill.start_clipped || skill.end_clipped ? ' · partial clip' : ''}`;
    const supplied = (reference.skills || []).find(item => item.skill_index === skill.skill_index);
    if (supplied?.routine_index) title.textContent = `Routine skill ${supplied.routine_index} · Flight ${skill.skill_index}`;
    else if (!showContext && reference.routine_start_seconds != null) title.textContent = `Routine skill ${index + 1} · Flight ${skill.skill_index}${skill.start_clipped || skill.end_clipped ? ' · partial clip' : ''}`;
    const dl = document.createElement('dl'), rotation = skill.rotation || {}, shape = skill.execution || {};
    const supported = rotation.somersault_count == null ? 'Unknown / needs review' : `${rotation.somersault_count} ${rotation.direction || 'unknown'}`;
    const shapeNames = {straight_like:'Straight', pike_like:'Pike', tuck_like:'Tuck', flexed_or_open_shape:'Flexed / open', unknown:'Unknown / needs review'};
    field(dl, 'Shape (estimate)', shapeNames[shape.execution_position] || 'Unknown / needs review');
    field(dl, 'Rotations', supported);
    field(dl, 'Start / end frames', `${frame(skill.start_frame)} → ${frame(skill.end_frame)}`);
    field(dl, 'Estimated contact times', `${time(skill.start_seconds)} → ${time(skill.end_seconds)}`);
    field(dl, 'Quarter somersaults', rotation.somersault_quarters == null ? 'Unknown / needs review' : `${rotation.somersault_quarters} × ¼ · ${rotation.direction}`);
    if (rotation.somersault_quarters == null && rotation.candidate_somersault_quarters != null) field(dl, 'Tentative rotation — needs review', `${rotation.candidate_somersault_quarters} × ¼ · ${rotation.candidate_direction}`);
    field(dl, 'Time of tightest supported held shape', time(shape.shape_timestamp_seconds));
    if (shape.tightest_shape_review_interval_seconds) field(dl, 'Possible tighter shape — needs review', shape.tightest_shape_review_interval_seconds.map(time).join(' → '), true);
    field(dl, 'Hip / knee angles', `${angle(shape.sustained_hip_angle_degrees)} / ${angle(shape.sustained_knee_angle_degrees)}`);
    field(dl, 'Intended skill', 'Unconfirmed — review the video');
    if (supplied) field(dl, 'User-supplied reference', `${supplied.skill_identity} · ${supplied.somersault_quarters} × ¼ ${supplied.direction}${supplied.twist_count != null ? ' · twist ' + supplied.twist_count + ' (supplied; detection deferred)' : ''}`, true);
    const reasons = [...(rotation.reasons || []), ...(shape.reasons || [])];
    if (rotation.contact_outage_excluded) reasons.push('Tracking outage near body contact was excluded; check the contact time');
    field(dl, 'Items to review', reasons.length ? reasons.join(', ') : 'Check estimated contact times and body shape', true);
    const flightLabel = title.textContent, guess = bestGuess(skill);
    title.textContent = supplied?.user_confirmed ? `Skill: ${supplied.skill_identity}` : `Best guess: ${guess.name}`;
    const heading = document.createElement('div'); heading.className = 'skill-card-heading';
    const jump = document.createElement('button'); jump.type = 'button'; jump.className = 'primary'; jump.textContent = 'Jump to skill';
    const start = skill.start_seconds ?? skill.visible_start_seconds;
    jump.disabled = !Number.isFinite(start);
    jump.setAttribute('aria-label', `Jump to ${flightLabel.toLowerCase()}${jump.disabled ? ' (time unavailable)' : ' at ' + time(start)}`);
    jump.onclick = () => jumpToSkill(start);
    const titleBlock = document.createElement('div'), flight = document.createElement('p'), uncertainty = document.createElement('p');
    flight.className = 'small'; flight.textContent = flightLabel;
    uncertainty.className = 'small'; uncertainty.textContent = supplied?.user_confirmed ? `Confirmed by you${supplied.takeoff_position ? ' · Start: ' + supplied.takeoff_position : ''}${supplied.landing_position ? ' · Landing: ' + supplied.landing_position : ''}.` : guess.note;
    titleBlock.append(title, flight, uncertainty);
    heading.append(titleBlock, jump); card.append(heading);
    const evidence = images[String(skill.skill_index)];
    if (evidence?.available) {
      card.classList.add('has-image');
      const figure = document.createElement('figure'), image = document.createElement('img'), caption = document.createElement('figcaption'), heading = document.createElement('h4');
      heading.className = 'shape-image-title'; heading.textContent = 'Tightest shape during skill';
      image.src = `/api/jobs/${selectedId}/skill-images/${skill.skill_index}.jpg`;
      image.alt = `Flight ${skill.skill_index} pose overlay at the selected shape`; image.loading = 'lazy';
      caption.textContent = `Preview frame ${evidence.frame_index + 1} · ${time(evidence.timestamp_seconds)}. ${evidence.selection_method?.startsWith('occluded_shape_review') ? 'Review candidate during missing-joint interval; exact tightest time is unverified. Angles below come from the supported held shape.' : 'Angles are measured across the held shape, not just this frame.'}`;
      figure.append(heading, image, caption); card.append(figure);
    } else {
      const note = document.createElement('p'); note.className = 'small'; note.textContent = evidence?.message || 'No preview available for this flight.'; card.append(note);
    }
    card.append(dl); return card;
  }));
}
function renderJobs() {
  if (embedded) return;
  const tracked = jobs.filter(job => job.state === 'done'); $('empty-list').hidden = tracked.length > 0;
  $('job-list').replaceChildren(...tracked.map(job => {
    const button = document.createElement('button'); button.className = 'job-card' + (job.id === selectedId ? ' selected' : '');
    const name = document.createElement('strong'), status = document.createElement('span'); name.textContent = job.name; status.textContent = stateNames[job.analysis?.state || 'not_started']; button.append(name, status);
    button.onclick = () => selectJob(job.id).catch(error => notice(error.message)); return button;
  }));
}
async function selectJob(id) { clearSelection(); if (!embedded) { $('result-video').removeAttribute('src'); $('result-video').load(); } showContext = false; if ($('show-context')) $('show-context').checked = false; selectedId = id; const version = ++requestVersion; renderJobs(); await refreshSelected(version); }
async function refreshSelected(version = requestVersion) {
  if (!selectedId) return;
  const id = selectedId, job = jobs.find(item => item.id === id);
  if (!job || job.state !== 'done') { clearSelection(); return; }
  const data = await api(`/jobs/${id}/analysis`);
  if (version !== requestVersion || id !== selectedId) return;
  const analysis = data.analysis;
  if ($('empty-selection')) $('empty-selection').hidden = true;
  if (!embedded) {
    $('video-name').textContent = job.name;
    const src = `/api/jobs/${id}/results/overlay.mp4`;
    if ($('result-video').getAttribute('src') !== src) $('result-video').src = src;
  }
  $('analysis-panel').hidden = false; $('analysis-state').textContent = stateNames[analysis.state] || analysis.state; $('analysis-message').textContent = analysis.message;
  $('start-analysis').hidden = !['not_started', 'error', 'done'].includes(analysis.state); $('start-analysis').textContent = analysis.state === 'done' ? 'Redo analysis' : 'Run skill analysis'; $('start-analysis').disabled = submitting;
  $('analysis-results').hidden = analysis.state !== 'done';
  if (analysis.state === 'done') {
    let context = $('routine-context');
    if (!context) {
      context = document.createElement('div'); context.id = 'routine-context'; context.className = 'routine-context';
      const label = document.createElement('label'), checkbox = document.createElement('input'), note = document.createElement('p'); checkbox.type = 'checkbox'; checkbox.id = 'show-context';
      checkbox.onchange = () => { showContext = checkbox.checked; refreshSelected().catch(error => notice(error.message)); }; label.append(checkbox, document.createTextNode('Show preparation and outbounces')); note.id = 'routine-note'; note.className = 'small'; context.append(label, note); $('skill-list').before(context);
    }
    context.hidden = data.reference?.routine_start_seconds == null;
    $('routine-note').textContent = data.reference?.routine_start_seconds != null ? `Routine starts at ${time(data.reference.routine_start_seconds)} (${data.reference.window_provenance === 'agent_proposed_from_video' ? 'proposed from video; needs review' : 'user reference'}). End and sequence alignment are proposed and need review.` : '';
    const signature = JSON.stringify([id, showContext, data.skills, data.shape_images, data.reference]);
    if (signature !== renderedResult) {
      renderSkills(data.skills || [], data.shape_images || {}, data.reference || {});
      renderedResult = signature;
    }
    $('download-skills-json').href = `/api/jobs/${id}/results/skills.json`; $('download-skills-csv').href = `/api/jobs/${id}/results/skills.csv`;
  } else { $('skill-list').replaceChildren(); renderedResult = null; }
}
async function refresh() { if (polling) return; polling = true; try { const data = await api('/jobs'); unlock(); jobs = data.jobs; renderJobs(); await refreshSelected(); } finally { polling = false; } }
if (!embedded) {
$('login-form').onsubmit = async event => { event.preventDefault(); try { await api('/login', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({key:$('access-key').value.trim()})}); $('access-key').value = ''; await refresh(); } catch (error) { notice(error.message); } };
$('logout').onclick = async () => { try { await api('/logout', {method:'POST'}); lock(); } catch (error) { notice(error.message); } };
}
$('start-analysis').onclick = async () => {
  if (!selectedId || submitting) return;
  const id = selectedId; submitting = true; $('start-analysis').disabled = true;
  try { await api(`/jobs/${id}/analysis`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({redo:true})}); await refreshSelected(); }
  catch (error) { notice(error.message); }
  finally { submitting = false; $('start-analysis').disabled = false; }
};
if (!embedded) {
refresh().catch(error => { if (error.message !== 'Unlock the inbox first') notice(error.message); });
setInterval(() => { if ($('workspace').hidden) return; refresh().catch(error => notice(error.message)); }, 2500);

} else {
  window.skillAnalysis = {
    clear: clearSelection,
    async select(job) {
      if (!job || job.state !== 'done') { clearSelection(); return; }
      jobs = [job];
      if (selectedId !== job.id) await selectJob(job.id);
      else await refreshSelected();
    }
  };
}
})();
