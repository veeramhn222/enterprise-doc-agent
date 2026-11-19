const $ = s => document.querySelector(s);
let providers = {}, docs = [], settings = {configured: false}, chatStarted = false, busy = false;

// ------------------------------------------------------------------ helpers
async function api(url, opts) {
  const r = await fetch(url, opts);
  let data = {};
  try { data = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error(data.detail || ('Request failed (' + r.status + ')'));
  return data;
}
function msg(el, text, kind) { el.textContent = text; el.className = 'msg ' + (kind || ''); }
function toast(text, kind) {
  const t = document.createElement('div');
  t.className = 'toast ' + (kind || ''); t.textContent = text; $('#toasts').appendChild(t);
  setTimeout(() => t.classList.add('out'), 3200); setTimeout(() => t.remove(), 3600);
}
const EXT = {pdf: ['PDF', 'pdf'], txt: ['TXT', 'txt'], md: ['MD', 'txt'], csv: ['CSV', 'csv'], xlsx: ['XLS', 'xls'], xlsm: ['XLS', 'xls']};
function fileBadge(name) {
  const [label, cls] = EXT[(name.split('.').pop() || '').toLowerCase()] || ['DOC', 'txt'];
  return `<span class="ftype ${cls}">${label}</span>`;
}
const chips = (list, cls) => (list || []).map(q => `<button class="chip ${cls || ''}" data-q="${esc(q)}">${esc(q)}</button>`).join('');
const tags = list => (list || []).map(t => `<span class="tag">${esc(t)}</span>`).join('');
const fmtDate = iso => { try { return new Date(iso).toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'}); } catch (e) { return ''; } };
const descOf = d => d.description || {summary: 'No description yet. Click “Regenerate” to create one.', topics: [], questions: [], described_by: 'none'};
function sourceBadge(ds) {
  if (ds.described_by === 'ai') return '<span class="badge ai" title="Written by your configured LLM">✨ AI summary</span>';
  if (ds.described_by === 'auto') return '<span class="badge auto" title="Built from the file content. Save an API key in Admin and click Regenerate for an AI-written summary.">Auto summary</span>';
  return '';
}

// --------------------------------------------------------------------- tabs
function showTab(name) {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'docs') loadDocs();
  if (name === 'admin') loadSettings();
  if (name === 'chat') $('#question').focus();
}
document.querySelectorAll('.tab').forEach(b => b.onclick = () => showTab(b.dataset.tab));
$('#llm-badge').onclick = () => showTab('admin');

// ------------------------------------------------------------------- admin
const OTHER = '__other__';
function syncPick() {                                   // dropdown = every known model + the saved/current one + "Other…"
  const p = providers[$('#provider').value] || {models: []}, cur = $('#model').value.trim();
  const list = [...p.models]; if (cur && !list.includes(cur)) list.unshift(cur);
  const sel = $('#model-pick'), other = !cur && !p.models.length;
  sel.innerHTML = list.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('') + `<option value="${OTHER}">Other (type a model name)…</option>`;
  sel.value = other ? OTHER : (cur || list[0] || OTHER);
  if (sel.value !== OTHER && !cur) $('#model').value = sel.value;
  $('#model').hidden = sel.value !== OTHER;
}
$('#model-pick').onchange = () => {
  const v = $('#model-pick').value;
  if (v === OTHER) { $('#model').value = ''; $('#model').hidden = false; $('#model').focus(); }
  else { $('#model').value = v; $('#model').hidden = true; }
};
function fillModels() {
  const p = providers[$('#provider').value] || {models: [], hint: ''};
  syncPick();
  $('#hint').textContent = p.hint || '';
  const a = $('#key-link');
  if (p.key_url) { a.href = p.key_url; a.hidden = false; } else { a.hidden = true; }
}
function paintStatus() {
  const b = $('#llm-badge');
  $('#llm-text').textContent = settings.configured ? `${settings.provider} · ${settings.model}` : 'LLM not configured';
  b.className = 'status' + (settings.configured ? ' ok' : '');
  $('#cfg-banner').hidden = !!settings.configured;
}
function renderKeys() {
  const rows = settings.profiles || [];
  $('#env-note').hidden = !settings.env_key_present;
  $('#key-list').innerHTML = rows.length ? rows.map(p => {
    const usable = p.has_key || !p.needs_key;
    const badge = p.active ? (usable ? '<span class="st-badge active">● Active</span>' : '<span class="st-badge warn">Active · no key</span>')
                           : (usable ? '<span class="st-badge">Saved</span>' : '<span class="st-badge">No key</span>');
    const val = p.has_key ? `<code class="keyval">${esc(p.key_hint)}</code>`
              : (p.needs_key ? '<span class="keyval blank">— empty —</span>' : '<span class="keyval blank">no key needed</span>');
    return `<div class="key-row" data-provider="${esc(p.provider)}">
      <div class="who"><b>${esc(p.label)}</b><span>${esc(p.model || '—')}${p.base_url ? ' · ' + esc(p.base_url) : ''}</span></div>
      ${val}${badge}
      ${usable && !p.active ? `<button class="use" data-act="usekey" data-prov="${esc(p.provider)}" title="Make this the active model">Use</button>` : '<span></span>'}
      <button class="icon-btn" data-act="delkey" data-prov="${esc(p.provider)}" title="Delete this API key" aria-label="Delete ${esc(p.label)} API key">🗑</button>
    </div>`;
  }).join('') : '<p class="muted">No keys saved yet. Add one below.</p>';
}
async function loadSettings(resetForm = true) {
  const d = await api('/api/admin/settings');
  providers = d.providers; settings = d.settings;
  if (resetForm || !$('#provider').options.length) {          // opening the tab / saving: fill the form from the active entry
    $('#provider').innerHTML = Object.entries(providers).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join('');
    if (settings.provider) $('#provider').value = settings.provider;
    $('#model').value = settings.model || '';
    $('#base-url').value = settings.base_url || '';
    $('#api-key').value = '';
  }                                                           // deleting / switching a key leaves what the user is typing alone
  fillModels(); showFormKeyState(); renderKeys(); paintStatus();
}
function showFormKeyState() {
  const prov = $('#provider').value, saved = (settings.profiles || []).find(p => p.provider === prov);
  const needs = (providers[prov] || {}).needs_key !== false;
  $('#key-status').textContent = saved && saved.has_key ? '✔ A key is saved for this provider. Leave blank to keep it.'
    : (needs ? 'No key saved for this provider.' : 'No key needed for this provider.');
}
$('#provider').onchange = () => {
  $('#model').value = '';
  fillModels();
  const prov = $('#provider').value, saved = (settings.profiles || []).find(p => p.provider === prov);
  $('#model').value = (saved && saved.model) || (providers[prov].models || [])[0] || ''; syncPick();
  $('#base-url').value = (saved && saved.base_url) || '';
  $('#api-key').value = ''; showFormKeyState();
};
$('#submit-settings').onclick = async () => {
  const btn = $('#submit-settings'); btn.disabled = true;
  try {
    await api('/api/admin/settings', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({provider: $('#provider').value, model: $('#model').value, api_key: $('#api-key').value, base_url: $('#base-url').value})});
    msg($('#admin-msg'), 'Saved to data/settings.json. Click “Test connection” to verify.', 'ok');
    toast('Settings saved');
    await loadSettings();
  } catch (e) { msg($('#admin-msg'), e.message, 'err'); }
  btn.disabled = false;
};
$('#test-btn').onclick = async () => {
  const btn = $('#test-btn'); btn.disabled = true; msg($('#admin-msg'), 'Testing the connection…', 'busy');
  try { const r = await api('/api/admin/test', {method: 'POST'}); msg($('#admin-msg'), '✔ Connected. Model replied: ' + r.reply, 'ok'); toast('Connection works'); }
  catch (e) { msg($('#admin-msg'), '✘ ' + e.message, 'err'); }
  btn.disabled = false;
};

// --------------------------------------------------------------- documents
async function loadDocs() {
  try { docs = (await api('/api/documents')).documents; } catch (e) { docs = []; toast(e.message, 'err'); }
  renderDocGrid(); renderSide(); if (!chatStarted) renderWelcome();
}
function renderDocGrid() {
  $('#doc-count').textContent = docs.length ? `${docs.length} indexed` : '';
  $('#doc-grid').innerHTML = docs.length ? docs.map(d => {
    const ds = descOf(d);
    return `<article class="doc-card">
      <div class="head">${fileBadge(d.filename)}<div style="min-width:0"><div class="title">${esc(d.filename)}</div>
        <div class="meta">${esc(ds.kind || '')} · ${d.chunks} chunks · added ${fmtDate(d.uploaded_at)}</div></div></div>
      <div>${sourceBadge(ds)}</div>
      <p class="summary">${esc(ds.summary)}</p>
      ${ds.topics && ds.topics.length ? `<div><div class="label">Topics</div>${tags(ds.topics)}</div>` : ''}
      ${ds.questions && ds.questions.length ? `<div><div class="label">You could ask</div>${chips(ds.questions, 'sm')}</div>` : ''}
      <div class="actions"><button data-act="describe" data-id="${esc(d.doc_id)}">✨ Regenerate description</button>
        <button class="danger" data-act="del" data-id="${esc(d.doc_id)}">Delete</button></div>
    </article>`;
  }).join('') : '<div class="empty">No documents yet. Drop files above or load the samples to get started.</div>';
}
function renderSide() {
  $('#side-docs').innerHTML = docs.length ? docs.map(d => {
    const ds = descOf(d);
    return `<div class="side-doc"><div class="top">${fileBadge(d.filename)}<div class="name">${esc(d.filename)}</div></div>
      <div class="sum">${esc(ds.summary)}</div>${chips((ds.questions || []).slice(0, 2), 'sm')}</div>`;
  }).join('') : '<p class="muted">No documents yet. Add some in the Documents tab, then ask questions about them here.</p>';
}
function renderWelcome() {
  const log = $('#chat-log');
  const samples = docs.flatMap(d => (descOf(d).questions || []).slice(0, 2)).slice(0, 6);
  log.innerHTML = `<div class="welcome"><div class="emoji">📚</div><h2>Ask anything about your documents</h2>` +
    (docs.length
      ? `<p>${docs.length} document${docs.length > 1 ? 's' : ''} indexed. Pick a suggestion or type your own question below. Answers are grounded in your files and show their sources.</p><div class="chips">${chips(samples)}</div>`
      : `<p>Upload PDF, TXT, CSV or Excel files, then ask questions in plain English. You'll get cited answers produced by planning, retrieval, reasoning and validation agents.</p>
         <div class="cta"><button class="primary" data-goto="docs">Add documents</button><button data-act="samples">Load sample documents</button></div>`) +
    `</div>`;
}
async function uploadFiles(files) {
  if (!files || !files.length) return;
  const fd = new FormData(); for (const f of files) fd.append('files', f);
  msg($('#upload-msg'), `Indexing and describing ${files.length} file(s)… the first run downloads the embedding model`, 'busy');
  try { showResults(await api('/api/documents', {method: 'POST', body: fd})); } catch (e) { msg($('#upload-msg'), e.message, 'err'); }
  $('#file-input').value = ''; await loadDocs();
}
function showResults(res) {
  const lines = res.results.map(r => r.ok ? `✔ ${r.filename} – ${r.chunks} chunks indexed` : `✘ ${r.filename} – ${r.error}`);
  const ok = res.results.every(r => r.ok);
  msg($('#upload-msg'), lines.join('\n') || 'Nothing new to add (samples already loaded?).', ok ? 'ok' : 'err');
  if (res.results.some(r => r.ok)) toast('Documents indexed');
}
async function loadSamples() {
  msg($('#upload-msg'), 'Loading samples… the first run downloads the embedding model', 'busy');
  document.querySelectorAll('#samples-btn,[data-act=samples]').forEach(b => b.disabled = true);
  try { showResults(await api('/api/documents/load-samples', {method: 'POST'})); } catch (e) { msg($('#upload-msg'), e.message, 'err'); }
  document.querySelectorAll('#samples-btn,[data-act=samples]').forEach(b => b.disabled = false);
  await loadDocs();
}
const dz = $('#dropzone');
dz.onclick = () => $('#file-input').click();
dz.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file-input').click(); } };
['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', e => uploadFiles(e.dataTransfer.files));
$('#file-input').onchange = e => uploadFiles(e.target.files);
$('#samples-btn').onclick = loadSamples;

// one delegated handler for chips, goto links and card buttons
document.addEventListener('click', async e => {
  const t = e.target.closest('[data-q],[data-goto],[data-act]'); if (!t) return;
  if (t.dataset.q) { showTab('chat'); ask(t.dataset.q); }
  else if (t.dataset.goto) showTab(t.dataset.goto);
  else if (t.dataset.act === 'samples') loadSamples();
  else if (t.dataset.act === 'delkey') {
    const row = (settings.profiles || []).find(p => p.provider === t.dataset.prov);
    if (!confirm(`Delete the ${row ? row.label : ''} entry and its API key from the server?` + (row && row.active ? ' It is the active model, so questions will stop working until you add a key.' : ''))) return;
    t.disabled = true;
    try {
      await api('/api/admin/keys/' + encodeURIComponent(t.dataset.prov), {method: 'DELETE'});
      msg($('#admin-msg'), `The ${row ? row.label : ''} API key was deleted.`, 'ok'); toast('API key deleted');
    } catch (err) { msg($('#admin-msg'), err.message, 'err'); }
    await loadSettings(false);
  } else if (t.dataset.act === 'usekey') {
    try { await api('/api/admin/active', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({provider: t.dataset.prov})}); toast('Active model changed'); }
    catch (err) { msg($('#admin-msg'), err.message, 'err'); }
    await loadSettings(false);
  }
  else if (t.dataset.act === 'del') {
    if (!confirm('Delete this document and its index?')) return;
    try { await api('/api/documents/' + t.dataset.id, {method: 'DELETE'}); toast('Document deleted'); } catch (err) { toast(err.message, 'err'); }
    loadDocs();
  } else if (t.dataset.act === 'describe') {
    t.disabled = true; t.textContent = 'Working…';
    try { await api('/api/documents/' + t.dataset.id + '/describe', {method: 'POST'}); toast('Description updated'); }
    catch (err) { toast(err.message, 'err'); }
    loadDocs();
  }
});

// -------------------------------------------------------------------- chat
function renderAnswer(r) {
  const v = r.validation;
  let html = md(r.answer);
  if (v) html += `<span class="pill ${v.grounded && v.confidence >= .5 ? '' : 'warn'}">${v.grounded ? '✔ grounded' : '⚠ unverified'} · ${(v.confidence * 100).toFixed(0)}% confidence</span>`;
  html += `<details><summary>Agent trace (${r.trace.length} steps)</summary>` + r.trace.map(t =>
    `<div class="step"><span class="ag ag-${esc(t.agent.toLowerCase())}">${esc(t.agent)}</span><span>${esc(t.detail)}${t.seconds != null ? ` <span class="muted">(${t.seconds}s)</span>` : ''}</span></div>`).join('') + '</details>';
  if (r.sources.length) html += `<details><summary>Sources (${r.sources.length})</summary>` + r.sources.map(s =>
    `<div class="src ${s.cited ? 'cited' : ''}"><div class="src-head"><sup class="cite">${esc(s.id)}</sup><b>${esc(s.filename)}</b><span class="muted">${esc(s.location)} · distance ${s.distance}</span>${s.cited ? '<span class="badge ai">cited</span>' : ''}</div><pre>${esc(s.text)}</pre></div>`).join('') + '</details>';
  return html;
}
function addMsg(who, html) {
  if (!chatStarted) { $('#chat-log').innerHTML = ''; chatStarted = true; }
  const row = document.createElement('div'); row.className = 'msg-row ' + who;
  row.innerHTML = who === 'user' ? `<div class="bubble user"></div>` : `<div class="avatar">AI</div><div class="bubble bot"></div>`;
  const b = row.querySelector('.bubble'); b.innerHTML = html;
  $('#chat-log').appendChild(row); row.scrollIntoView({behavior: 'smooth', block: 'end'});
  return b;
}
async function ask(text) {
  const q = (text || $('#question').value).trim(); if (!q || busy) return;
  busy = true; $('#question').value = ''; autosize(); $('#ask-btn').disabled = true;
  addMsg('user', esc(q));
  const wait = addMsg('bot', '<span class="typing"><i></i><i></i><i></i>&nbsp; Agents are working: plan, retrieve, reason, validate…</span>');
  try {
    const r = await api('/api/query', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: q})});
    wait.innerHTML = renderAnswer(r);
  } catch (e) { wait.className = 'bubble bot err'; wait.textContent = e.message; }
  busy = false; $('#ask-btn').disabled = false; wait.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}
function autosize() { const t = $('#question'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 160) + 'px'; }
$('#ask-btn').onclick = () => ask();
$('#question').addEventListener('input', autosize);
$('#question').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } });

// -------------------------------------------------------------------- init
renderWelcome();
loadSettings().catch(() => {});
loadDocs();
