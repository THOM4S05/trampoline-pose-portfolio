const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.join(__dirname, '../src/trampoline_pose/web');

function setup(embedded = true) {
  const elements = new Map();
  class Element {
    constructor(tag) { this.tagName = tag; this.children = []; this.events = {}; this.attrs = {}; this.classList = {add(){}}; this.readyState = 1; this.duration = 30; this.currentTime = 0; }
    set id(value) { this._id = value; elements.set(value, this); }
    get id() { return this._id; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    before() {}
    setAttribute(key, value) { this.attrs[key] = value; }
    getAttribute(key) { return this.attrs[key] ?? null; }
    removeAttribute(key) { delete this.attrs[key]; }
    set src(value) { this.attrs.src = value; this.sourceChanges = (this.sourceChanges || 0) + 1; }
    addEventListener(name, callback) { this.events[name] = callback; }
    scrollIntoView() { this.scrolled = true; }
    focus() { this.focused = true; }
    load() { this.currentTime = 0; }
  }
  const html = fs.readFileSync(path.join(root, embedded ? 'index.html' : 'skills.html'), 'utf8');
  for (const match of html.matchAll(/id="([^"]+)"/g)) { assert(!elements.has(match[1]), 'Unique HTML IDs'); const el = new Element('div'); el.id = match[1]; }
  const job = {id:'a', state:'done', name:'Routine'};
  let data = {analysis:{state:'done', message:'Ready'}, skills:[
    {skill_index:1, start_seconds:4.25, visible_start_seconds:4.25},
    {skill_index:2, start_seconds:null, visible_start_seconds:0, start_clipped:true},
    {skill_index:3, start_seconds:null},
  ], shape_images:{}, reference:{}};
  const context = {document:{getElementById:id => elements.get(id) || null, createElement:tag => new Element(tag), createTextNode:text => text}, window:{}, setTimeout(){}, clearTimeout(){}, setInterval(){}, fetch:async url => ({ok:true, json:async () => url === '/api/jobs' ? {jobs:[job]} : data})};
  vm.runInNewContext(fs.readFileSync(path.join(root, 'skills.js'), 'utf8'), context);
  return {elements, job, api:context.window.skillAnalysis, html, setData(value){data=value;}, button(index){return elements.get('skill-list').children[index].children[0].children[1];}};
}

test('embedded cards seek by contact time or first visible time, with disabled unknown timing', async () => {
  const ui = setup(); await ui.api.select(ui.job);
  assert(ui.html.indexOf('id="result-video"') < ui.html.indexOf('id="embedded-skills"'));
  const video = ui.elements.get('result-video');
  assert.equal(ui.button(0).textContent, 'Jump to skill');
  ui.button(0).onclick(); assert.equal(video.currentTime, 4.25); assert(video.scrolled && video.focused);
  ui.button(1).onclick(); assert.equal(video.currentTime, 0);
  assert.equal(ui.button(2).disabled, true);
});

test('seeking waits for metadata and clamps to the video duration', async () => {
  const ui = setup(); await ui.api.select(ui.job);
  const video = ui.elements.get('result-video'); video.readyState = 0;
  ui.button(0).onclick(); assert.equal(video.currentTime, 0);
  video.readyState = 1; video.duration = 4; video.events.loadedmetadata(); assert.equal(video.currentTime, 4);
});

test('switching videos cancels a pending seek; polling preserves controls and playback', async () => {
  const ui = setup(); await ui.api.select(ui.job);
  const video = ui.elements.get('result-video'); video.readyState = 0; ui.button(0).onclick();
  await ui.api.select({...ui.job, id:'b'});
  video.readyState = 1; video.events.loadedmetadata(); assert.equal(video.currentTime, 0);
  const button = ui.button(0); video.currentTime = 7;
  await ui.api.select({...ui.job, id:'b'});
  assert.equal(ui.button(0), button); assert.equal(video.currentTime, 7);
  ui.api.clear(); assert.equal(ui.elements.get('analysis-panel').hidden, true);
});

test('separate analysis page loads the matching video above the skills', async () => {
  const ui = setup(false);
  await new Promise(resolve => setImmediate(resolve));
  await ui.elements.get('job-list').children[0].onclick();
  const video = ui.elements.get('result-video');
  assert.equal(video.getAttribute('src'), '/api/jobs/a/results/overlay.mp4');
  assert(ui.html.indexOf('id="result-video"') < ui.html.indexOf('id="skill-list"'));
  ui.button(0).onclick(); assert.equal(video.currentTime, 4.25);
});

test('best guesses distinguish shape, start position, rotations, and twists', async () => {
  const ui = setup();
  const base = {skill_index:1, takeoff_position:'feet', landing_position:'feet', twist:0, rotation:{somersault_quarters:4, direction:'backward'}, execution:{execution_position:'tuck_like'}};
  const cases = [
    [base, 'Back somersault (tuck)'],
    [{...base, twist:1}, 'Back full (tuck)'],
    [{...base, twist:.5, rotation:{somersault_quarters:4, direction:'forward'}}, 'Barani (tuck)'],
    [{...base, takeoff_position:'back', twist:1}, 'Back somersault · full twist from back (tuck)'],
    [{...base, takeoff_position:'front', rotation:{somersault_quarters:1, direction:'backward'}}, 'Return to feet from front (tuck)'],
    [{...base, landing_position:'front', rotation:{somersault_quarters:1, direction:'forward'}}, 'Front drop (tuck)'],
    [{...base, rotation:{somersault_quarters:0}}, 'Tuck jump'],
    [{...base, twist:.5, rotation:{somersault_quarters:0}}, 'Half twist · tuck jump'],
    [{...base, twist:1, rotation:{somersault_quarters:0}}, 'Full twist · tuck jump'],
    [{...base, rotation:{somersault_quarters:5}}, 'Outside naming range'],
    [{...base, twist:1.5}, 'Outside naming range'],
    [{...base, start_clipped:true}, 'Unknown — partial skill'],
    [{...base, rotation:{somersault_quarters:null, candidate_somersault_quarters:4}}, 'Unknown'],
  ];
  for (const [skill, expected] of cases) {
    ui.setData({analysis:{state:'done'}, skills:[skill], reference:{}, shape_images:{}});
    await ui.api.select(ui.job);
    const heading = ui.elements.get('skill-list').children[0].children[0].children[0].children[0];
    assert.equal(heading.textContent, `Best guess: ${expected}`);
  }
});

test('missing twist and start remain unconfirmed; supplied labels do not become guesses', async () => {
  const ui = setup();
  ui.setData({analysis:{state:'done'}, skills:[{skill_index:1, takeoff_position:'unknown', twist:'deferred', rotation:{somersault_quarters:4, direction:'forward'}, execution:{execution_position:'pike_like'}}], reference:{skills:[{skill_index:1, skill_identity:'Barani', twist_count:.5}]}, shape_images:{}});
  await ui.api.select(ui.job);
  const titleBlock = ui.elements.get('skill-list').children[0].children[0].children[0];
  assert.equal(titleBlock.children[0].textContent, 'Best guess: Front somersault (pike)');
  assert.match(titleBlock.children[2].textContent, /starting position, landing position, twist/);
});
