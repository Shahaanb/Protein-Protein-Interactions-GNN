const $ = (id) => document.getElementById(id);

const form = $('predictForm');
const btnPredict = $('btnPredict');
const statusEl = $('status');

const interactionProbabilityEl = $('interactionProbability');
const inferenceModeEl = $('inferenceMode');
const evidenceScoreEl = $('evidenceScore');
const evidenceBreakdownEl = $('evidenceBreakdown');
const hotspotsEl = $('hotspots');

const viewer1El = $('viewer1');
const viewer2El = $('viewer2');

const scoresModal = $('scoresModal');
const scoresContent = $('scoresContent');
const btnScores = $('btnScores');
const btnRefreshScores = $('btnRefreshScores');

let viewer1 = null;
let viewer2 = null;

const pdb1El = $('pdb1');
const pdb2El = $('pdb2');

function normalizePdbInput(v) {
  return String(v ?? '')
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, '')
    .slice(0, 4);
}

function isValidPdbId(v) {
  return typeof v === 'string' && /^[A-Z0-9]{4}$/.test(v);
}

[pdb1El, pdb2El].forEach((el) => {
  el.addEventListener('input', () => {
    const next = normalizePdbInput(el.value);
    if (el.value !== next) el.value = next;
  });
});

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? 'rgba(255,120,120,0.9)' : 'rgba(232,238,252,0.7)';
}

function fmtPercent(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return '--';
  return `${x.toFixed(2)}%`;
}

function fmtNum(x, digits = 3) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return '--';
  return x.toFixed(digits);
}

function clearOutputs() {
  interactionProbabilityEl.textContent = '--';
  inferenceModeEl.textContent = '--';
  evidenceScoreEl.textContent = '--';
  evidenceBreakdownEl.textContent = '';
  hotspotsEl.textContent = '--';

  if (viewer1) viewer1.clear();
  if (viewer2) viewer2.clear();
  viewer1El.innerHTML = '';
  viewer2El.innerHTML = '';
  viewer1 = null;
  viewer2 = null;
}

function ensure3dmol() {
  return typeof window.$3Dmol !== 'undefined';
}

async function fetchPdb(pdbId) {
  const id = pdbId.toUpperCase();
  const res = await fetch(`https://files.rcsb.org/download/${id}.pdb`);
  if (!res.ok) throw new Error(`Failed to fetch PDB ${id}`);
  return await res.text();
}

function renderViewer(containerEl, pdbData, hotspots) {
  if (!ensure3dmol()) return null;

  const v = window.$3Dmol.createViewer(containerEl, { backgroundColor: 'rgba(0,0,0,0)' });
  v.addModel(pdbData, 'pdb');
  v.setStyle({}, { cartoon: { color: 'spectrum', opacity: 0.95 } });

  if (Array.isArray(hotspots)) {
    for (const hs of hotspots) {
      if (!hs || typeof hs.node_idx !== 'number') continue;
      v.addStyle({ resi: hs.node_idx }, { sphere: { color: '#FF5E00', scale: 1.5 } });
      v.addStyle({ resi: hs.node_idx }, { stick: { color: '#FF5E00', thickness: 0.3 } });
    }
  }

  v.zoomTo();
  v.render();
  return v;
}

function renderHotspots(hotspots) {
  if (!Array.isArray(hotspots) || hotspots.length === 0) {
    hotspotsEl.textContent = '--';
    return;
  }

  hotspotsEl.innerHTML = '';
  for (const hs of hotspots) {
    const item = document.createElement('div');
    item.className = 'hotspotItem';

    const k1 = document.createElement('div');
    k1.className = 'k';
    k1.textContent = 'Residue';

    const v1 = document.createElement('div');
    v1.className = 'v';
    v1.textContent = String(hs.node_idx);

    const k2 = document.createElement('div');
    k2.className = 'k';
    k2.style.marginTop = '8px';
    k2.textContent = 'Attention';

    const v2 = document.createElement('div');
    v2.className = 'v';
    v2.textContent = fmtNum(hs.attention, 4);

    item.append(k1, v1, k2, v2);
    hotspotsEl.appendChild(item);
  }
}

async function predict(pdb1, pdb2) {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pdb1, pdb2 }),
  });

  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { error: text || 'Invalid response' };
  }

  if (!res.ok) {
    throw new Error(body?.detail || body?.error || 'Inference failed');
  }

  return body;
}

function openModal() {
  scoresModal.classList.remove('hidden');
  scoresModal.setAttribute('aria-hidden', 'false');
}

function closeModal() {
  scoresModal.classList.add('hidden');
  scoresModal.setAttribute('aria-hidden', 'true');
}

function renderScores(summary) {
  if (!summary) {
    scoresContent.innerHTML = '<div class="muted">No summary available.</div>';
    return;
  }

  const cards = [];
  const add = (label, value) => {
    cards.push({ label, value });
  };

  add('Generated At', summary.generated_at ?? '--');
  add('Dataset', summary.dataset?.name ?? '--');
  add('Max Positives', summary.dataset?.max_positives ?? '--');
  add('Neg/Pos', summary.dataset?.negatives_per_positive ?? '--');
  add('Test N', summary.split?.test?.n ?? '--');
  add('Threshold', summary.threshold_selection?.threshold ?? '--');

  const m = summary.test_metrics ?? {};
  const ci = summary.test_metrics_ci95 ?? {};
  const metric = (key, label) => {
    const v = m[key];
    const c = ci[key];
    const val = `${fmtNum(v)}  [${fmtNum(c?.lo)}, ${fmtNum(c?.hi)}]`;
    add(label, val);
  };

  metric('accuracy', 'Accuracy');
  metric('precision', 'Precision');
  metric('recall', 'Recall');
  metric('f1', 'F1');
  metric('roc_auc', 'ROC-AUC');
  metric('pr_auc', 'PR-AUC');
  metric('brier', 'Brier');
  metric('ece_10', 'ECE (10 bins)');

  scoresContent.innerHTML = '';
  for (const c of cards) {
    const el = document.createElement('div');
    el.className = 'scoreCard';

    const l = document.createElement('div');
    l.className = 'label';
    l.textContent = c.label;

    const v = document.createElement('div');
    v.className = 'value';
    v.textContent = String(c.value);

    el.append(l, v);
    scoresContent.appendChild(el);
  }
}

async function loadScores() {
  scoresContent.innerHTML = '<div class="muted">Loading…</div>';
  try {
    const res = await fetch('/api/validation/summary');
    const data = await res.json();
    renderScores(data);
  } catch (e) {
    scoresContent.innerHTML = `<div class="muted">Failed to load scores: ${String(e?.message ?? e)}</div>`;
  }
}

btnScores.addEventListener('click', async () => {
  openModal();
  await loadScores();
});

btnRefreshScores.addEventListener('click', loadScores);

scoresModal.addEventListener('click', (e) => {
  const t = e.target;
  if (t && t.dataset && t.dataset.close === 'true') closeModal();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const pdb1 = normalizePdbInput(pdb1El.value);
  const pdb2 = normalizePdbInput(pdb2El.value);

  pdb1El.value = pdb1;
  pdb2El.value = pdb2;

  clearOutputs();

  if (!isValidPdbId(pdb1) || !isValidPdbId(pdb2)) {
    setStatus('Enter two 4-character PDB IDs (letters/numbers), e.g. 1BRS and 1CHO.', true);
    return;
  }

  setStatus('Running inference…');
  btnPredict.disabled = true;

  try {
    const resp = await predict(pdb1, pdb2);
    const data = resp?.data;

    interactionProbabilityEl.textContent = fmtPercent(data?.interaction_probability);
    inferenceModeEl.textContent = `Mode: ${data?.inference_mode ?? '--'}   (source: ${resp?.source ?? '--'})`;
    evidenceScoreEl.textContent = typeof data?.evidence_score === 'number' ? `${Math.round(data.evidence_score)}/100` : '--';
    evidenceBreakdownEl.textContent = data?.evidence_breakdown ? JSON.stringify(data.evidence_breakdown, null, 2) : '';

    renderHotspots(data?.hotspots);

    // Render structures (best-effort; independent of backend)
    if (ensure3dmol()) {
      try {
        setStatus('Rendering structures…');
        const [pdbData1, pdbData2] = await Promise.all([fetchPdb(pdb1), fetchPdb(pdb2)]);
        viewer1 = renderViewer(viewer1El, pdbData1, data?.hotspots);
        viewer2 = renderViewer(viewer2El, pdbData2, data?.hotspots);
        setStatus('Done.');
      } catch (e2) {
        setStatus(`Done (structure render failed: ${String(e2?.message ?? e2)})`);
      }
    } else {
      setStatus('Done (3Dmol not available).');
    }
  } catch (err) {
    setStatus(err?.message ?? 'Inference failed', true);
  } finally {
    btnPredict.disabled = false;
  }
});
