import React, { useState, useMemo, useEffect } from 'react';
import { Search, Loader2, Dna, Activity, BarChart2, Share2, Globe, ExternalLink } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import ProteinViewer from './components/ProteinViewer';

function App() {
  const [pdb1, setPdb1] = useState('');
  const [pdb2, setPdb2] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Validation summary (model-level scores)
  const [validationSummary, setValidationSummary] = useState(null);
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationError, setValidationError] = useState('');
  
  // Modal states
  const [activeModal, setActiveModal] = useState(null); // 'docs' | 'metrics' | 'attention' | 'scores' | null
  const [showToast, setShowToast] = useState(false);

  const loadValidationSummary = async () => {
    setValidationLoading(true);
    setValidationError('');
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/validation/summary`);
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || 'Failed to load validation summary');
      }
      const data = await res.json();
      setValidationSummary(data);
    } catch (e) {
      setValidationSummary(null);
      setValidationError(e?.message || 'Failed to load validation summary');
    } finally {
      setValidationLoading(false);
    }
  };

  useEffect(() => {
    if (activeModal === 'scores') {
      loadValidationSummary();
    }
  }, [activeModal]);

  const chartData = useMemo(() => {
    if (!result || !result.hotspots) return [];
    return result.hotspots.map((hs, i) => ({
      name: `Residue ${hs.node_idx}`,
      attention: hs.attention,
      index: i
    }));
  }, [result]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!pdb1 || !pdb2) {
      setError("Please provide two PDB IDs.");
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdb1, pdb2 })
      });
      
      if (!res.ok) throw new Error("Inference failed.");
      const data = await res.json();
      setResult(data.data);
    } catch (err) {
      setError(err.message || 'Error connecting to model.');
    } finally {
      setLoading(false);
    }
  };

  const copyShareLink = () => {
    const url = window.location.href;
    navigator.clipboard.writeText(url);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  return (
    <div className="min-h-screen w-full relative pt-8 pb-24 px-4 sm:px-6 lg:px-12">
      {/* Toast Notification */}
      {showToast && (
        <div className="fixed top-8 left-1/2 -translate-x-1/2 z-[100] glass px-6 py-3 rounded-2xl border-hotspotSecondary shadow-2xl flex items-center space-x-3 animate-bounce">
          <Activity className="text-hotspotSecondary" size={20} />
          <span className="text-sm font-bold">Link copied to clipboard!</span>
        </div>
      )}

      {/* Modals */}
      {activeModal && (
        <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-md" onClick={() => setActiveModal(null)} />
          <div className="relative glass-card max-w-2xl w-full p-8 rounded-[2rem] border-slate-700 max-h-[80vh] overflow-y-auto">
            <button onClick={() => setActiveModal(null)} className="absolute top-6 right-6 text-slate-500 hover:text-white">✕</button>
            
            {activeModal === 'docs' ? (
              <div className="space-y-6">
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-hotspotSecondary to-slate-400">Documentation</h2>
                <div className="space-y-4">
                  <section>
                    <h4 className="text-white font-bold mb-2 uppercase text-xs tracking-widest text-hotspotSecondary">Architecture</h4>
                    <p className="text-slate-400 text-sm leading-relaxed">Proteome-X uses a Geometric Graph Attention Network (GAT) to model protein interactions as graph message-passing tasks. Nodes represent C-alpha atoms of amino acids, and edges represent spatial proximity within 8Å.</p>
                  </section>
                  <section>
                    <h4 className="text-white font-bold mb-2 uppercase text-xs tracking-widest text-hotspotSecondary">Interpretable Attention</h4>
                    <p className="text-slate-400 text-sm leading-relaxed">Unlike "black-box" models, we extract attention coefficients from the final GAT layer. High attention weights indicate "Hotspots"—clusters of amino acids that the model identifies as critical for the binding interface.</p>
                  </section>
                </div>
              </div>
            ) : activeModal === 'scores' ? (
              <div className="space-y-6">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-hotspot to-slate-400">Model Scores</h2>
                    <p className="text-sm text-slate-400 mt-2">
                      Benchmark-style metrics computed from a lightweight DB5.5 pass.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={loadValidationSummary}
                    className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold rounded-xl transition-all border border-slate-700 active:scale-95 disabled:opacity-50"
                    disabled={validationLoading}
                  >
                    {validationLoading ? 'Refreshing…' : 'Refresh'}
                  </button>
                </div>

                {validationError && (
                  <div className="p-4 bg-red-400/10 rounded-xl border border-red-400/20">
                    <p className="text-sm text-red-300 font-medium">{validationError}</p>
                    <p className="text-xs text-slate-400 mt-2">
                      If you haven’t generated scores yet, run:
                      <span className="ml-2 font-mono text-slate-200">cd backend ; .\venv312\Scripts\python.exe run_validation.py --max-positives 40</span>
                    </p>
                  </div>
                )}

                {validationLoading && !validationSummary && (
                  <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
                    <p className="text-sm text-slate-300">Loading validation summary…</p>
                  </div>
                )}

                {validationSummary && (
                  <>
                    {validationSummary.inference?.used_mock && (
                      <div className="p-4 bg-red-400/10 rounded-xl border border-red-400/20">
                        <p className="text-sm text-red-200 font-bold">Benchmark run used MOCK inference for at least some examples.</p>
                        <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                          Metrics are not meaningful unless inference mode is <span className="font-mono">onnx</span>. Check your backend ONNX setup and re-run validation.
                        </p>
                      </div>
                    )}

                    {((validationSummary.split?.test?.n ?? 0) < 30 || (validationSummary.split?.test?.positives ?? 0) < 10 || (validationSummary.split?.test?.negatives ?? 0) < 10) && (
                      <div className="p-4 bg-amber-400/10 rounded-xl border border-amber-400/20">
                        <p className="text-sm text-amber-200 font-bold">Sample size is too small for reliable benchmark metrics.</p>
                        <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                          Current test split is <span className="font-mono">n={String(validationSummary.split?.test?.n ?? 0)}</span> (<span className="font-mono">+={String(validationSummary.split?.test?.positives ?? 0)}</span>, <span className="font-mono">-={String(validationSummary.split?.test?.negatives ?? 0)}</span>). Increase <span className="font-mono">--max-positives</span>
                          (and optionally <span className="font-mono">--n-boot</span>) and re-run validation to get stable scores.
                        </p>
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { label: 'Generated At', val: validationSummary.generated_at || '--' },
                        { label: 'Dataset', val: validationSummary.dataset?.name || '--' },
                        { label: 'Max Positives', val: validationSummary.dataset?.max_positives ?? '--' },
                        { label: 'Neg/Pos', val: validationSummary.dataset?.negatives_per_positive ?? '--' },
                        { label: 'Test N', val: validationSummary.split?.test?.n ?? '--' },
                        { label: 'Test +', val: validationSummary.split?.test?.positives ?? '--' },
                        { label: 'Test -', val: validationSummary.split?.test?.negatives ?? '--' },
                        { label: 'Threshold', val: validationSummary.threshold_selection?.threshold ?? '--' },
                        { label: 'Test ONNX', val: validationSummary.inference?.modes?.test?.onnx ?? 0 },
                        { label: 'Test MOCK', val: validationSummary.inference?.modes?.test?.mock ?? 0 },
                      ].map(m => (
                        <div key={m.label} className="bg-slate-950/40 p-3 rounded-lg border border-slate-800">
                          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{m.label}</p>
                          <p className="text-sm font-mono text-white mt-1 break-words">{String(m.val)}</p>
                        </div>
                      ))}
                    </div>

                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800 space-y-2">
                      <p className="text-xs text-slate-500 uppercase font-bold tracking-widest">Test Metrics (95% CI)</p>
                      <div className="grid grid-cols-2 gap-3">
                        {[
                          ['accuracy', 'Accuracy'],
                          ['precision', 'Precision'],
                          ['recall', 'Recall'],
                          ['f1', 'F1'],
                          ['roc_auc', 'ROC-AUC'],
                          ['pr_auc', 'PR-AUC'],
                          ['brier', 'Brier'],
                          ['ece_10', 'ECE (10 bins)'],
                        ].map(([key, label]) => {
                          const v = validationSummary.test_metrics?.[key];
                          const ci = validationSummary.test_metrics_ci95?.[key];
                          const fmt = (x) => (typeof x === 'number' ? x.toFixed(3) : '--');
                          return (
                            <div key={key} className="bg-slate-950/40 p-3 rounded-lg border border-slate-800">
                              <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{label}</p>
                              <p className="text-lg font-mono text-white mt-1">{fmt(v)}</p>
                              <p className="text-[11px] text-slate-400 font-mono">[{fmt(ci?.lo)}, {fmt(ci?.hi)}]</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-500 uppercase font-bold tracking-widest">Notes</p>
                      <p className="text-sm text-slate-300 mt-2 leading-relaxed">
                        This evaluation is demo-oriented: positives come from DB5.5, negatives are synthetic mismatches, and chain-level identity is ignored.
                        Treat metrics as a sanity-check signal, not a publishable benchmark.
                      </p>
                    </div>
                  </>
                )}
              </div>
            ) : activeModal === 'attention' ? (
              <div className="space-y-6">
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-hotspotSecondary to-slate-400">Attention Distribution</h2>

                <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800 space-y-3">
                  <p className="text-sm text-slate-300 leading-relaxed">
                    This chart visualizes the <span className="font-bold text-slate-200">top hotspots</span> returned by the backend.
                    Each bar is a residue from the hotspot list, and the bar length is its associated <span className="font-mono">attention</span> value.
                  </p>
                  <p className="text-sm text-slate-300 leading-relaxed">
                    It is useful for interpretability ("where the model focused"), but it is <span className="font-bold text-slate-200">not</span> a probability and not a calibrated measure of biological importance.
                  </p>
                </div>

                <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                  <p className="text-xs text-slate-500 uppercase font-bold tracking-widest">How it is computed</p>
                  <ul className="mt-2 text-[12px] text-slate-300 space-y-1 list-disc pl-4">
                    <li>
                      Backend computes a per-node attention signal from the model output (or generates deterministic mock values in mock mode).
                    </li>
                    <li>
                      Backend selects the top-5 residues by attention and returns them as <span className="font-mono">hotspots</span>.
                    </li>
                    <li>
                      Frontend maps each hotspot to <span className="font-mono">Residue &lt;node_idx&gt;</span> and plots its <span className="font-mono">attention</span>.
                    </li>
                  </ul>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Inference Mode', val: result?.inference_mode || '--' },
                    { label: 'Hotspot Count', val: result?.hotspots?.length ?? '--' },
                  ].map(m => (
                    <div key={m.label} className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                      <p className="text-xs text-slate-500 uppercase font-bold">{m.label}</p>
                      <p className="text-2xl font-mono text-white mt-1">{String(m.val)}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-hotspot to-slate-400">Metric Details</h2>

                <div className="space-y-4">
                  <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                    <p className="text-xs text-slate-500 uppercase font-bold">Interaction Probability</p>
                    <p className="text-sm text-slate-300 mt-2 leading-relaxed">
                      The displayed percentage is the backend's prediction for the requested PDB pair.
                      It is a model output, not a guarantee and not a calibrated measure of biological truth.
                    </p>

                    <div className="mt-4 p-3 bg-slate-950/40 rounded-lg border border-slate-800">
                      <p className="text-[11px] text-slate-400 font-bold uppercase tracking-widest">How it is computed</p>
                      <ul className="mt-2 text-[12px] text-slate-300 space-y-1 list-disc pl-4">
                        <li>
                          Backend runs inference and reads the model's first output (<span className="font-mono">outputs[0]</span>).
                        </li>
                        <li>
                          If that value is already in <span className="font-mono">[0, 1]</span>, it is treated as a probability; otherwise it is treated as a logit and converted via
                          <span className="font-mono"> sigmoid(x) = 1/(1+e^-x)</span>.
                        </li>
                        <li>
                          The API returns <span className="font-mono">interaction_probability = round(probability * 100, 2)</span>.
                        </li>
                        <li>
                          In <span className="font-mono">mock</span> mode (used if ONNX fails/unavailable), the probability is a deterministic pseudo-random value in the range 50–90%, seeded by the PDB pair.
                        </li>
                      </ul>
                    </div>

                    <p className="text-[12px] text-slate-400 mt-3 leading-relaxed">
                      Interpret alongside <span className="font-bold text-slate-200">Evidence Score</span> and the <span className="font-mono">ONNX/MOCK</span> mode badge.
                    </p>
                  </div>

                  <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-xs text-slate-500 uppercase font-bold">Evidence Score (0-100)</p>
                        <p className="text-sm text-slate-300 mt-2 leading-relaxed">
                          Evidence Score is an engineering trust metric.
                          It measures robustness (stability under small perturbations), hotspot consistency, and basic input sanity.
                          It is explicitly <span className="font-bold text-slate-200">not</span> model self-reported confidence.
                        </p>
                      </div>
                      <div className="px-2 py-1 bg-hotspotSecondary/10 rounded text-[10px] text-hotspotSecondary font-mono border border-hotspotSecondary/20">
                        {result?.evidence_breakdown?.metric_version || 'evidence_v1'}
                      </div>
                    </div>

                    <div className="mt-4 space-y-3">
                      <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800">
                        <p className="text-[11px] text-slate-400 font-bold uppercase tracking-widest">How it is computed</p>
                        <ul className="mt-2 text-[12px] text-slate-300 space-y-1 list-disc pl-4">
                          <li>
                            Run the model multiple times (<span className="font-mono">runs</span>) with tiny feature noise (<span className="font-mono">feature_noise_std</span>) to test stability.
                          </li>
                          <li>
                            Compute probability stability from the standard deviation of predicted probabilities (lower std → higher score).
                          </li>
                          <li>
                            Compute hotspot consistency as mean pairwise Jaccard similarity of the top-5 hotspot sets across runs.
                          </li>
                          <li>
                            Compute an input sanity score from graph size/connectivity (nodes + edges).
                          </li>
                          <li>
                            Apply a mode penalty: <span className="font-mono">onnx</span> keeps full score; <span className="font-mono">mock</span> is heavily down-weighted.
                          </li>
                        </ul>

                        <div className="mt-3 p-3 bg-slate-900/40 rounded-lg border border-slate-800">
                          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Formula</p>
                          <p className="text-[12px] text-slate-300 mt-1 font-mono break-words">
                            base = w_prob·S_prob + w_hot·S_hot + w_input·S_input
                          </p>
                          <p className="text-[12px] text-slate-300 mt-1 font-mono break-words">
                            EvidenceScore = 100 · clamp(base) · mode_factor
                          </p>
                          {result?.evidence_breakdown?.weights && (
                            <p className="text-[12px] text-slate-300 mt-2 font-mono break-words">
                              weights = {JSON.stringify(result.evidence_breakdown.weights)}
                            </p>
                          )}
                          {typeof result?.evidence_breakdown?.prob_std_max_percent_points === 'number' && (
                            <p className="text-[12px] text-slate-300 mt-1 font-mono break-words">
                              prob_std_max = {String(result.evidence_breakdown.prob_std_max_percent_points)} pp
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        {[
                          { label: 'Inference Mode', val: result?.evidence_breakdown?.inference_mode || result?.inference_mode || '--' },
                          { label: 'Runs', val: result?.evidence_breakdown?.runs ?? '--' },
                          { label: 'Prob Std (pp)', val: result?.evidence_breakdown?.probability_std_percent_points ?? '--' },
                          { label: 'Hotspot Consistency', val: result?.evidence_breakdown?.hotspot_consistency_score ?? '--' },
                          { label: 'Input Quality', val: result?.evidence_breakdown?.input_quality_score ?? '--' },
                          { label: 'Mode Factor', val: result?.evidence_breakdown?.mode_factor ?? '--' },
                        ].map(m => (
                          <div key={m.label} className="bg-slate-950/40 p-3 rounded-lg border border-slate-800">
                            <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{m.label}</p>
                            <p className="text-sm font-mono text-white mt-1">{String(m.val)}</p>
                          </div>
                        ))}
                      </div>

                      {result?.evidence_breakdown?.notes && (
                        <div className="p-3 bg-slate-950/40 rounded-lg border border-slate-800">
                          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Notes</p>
                          <p className="text-[12px] text-slate-300 mt-1 leading-relaxed">{result.evidence_breakdown.notes}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {!result && (
                  <div className="p-4 bg-slate-900 rounded-xl border border-slate-800">
                    <p className="text-sm text-slate-400">
                      Run a prediction to see the live Evidence Score breakdown for your specific PDB pair.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Scores side button */}
      <button
        type="button"
        onClick={() => setActiveModal('scores')}
        className="fixed right-4 top-1/2 -translate-y-1/2 z-[70] glass px-4 py-3 rounded-2xl border border-slate-700/60 hover:border-slate-500 transition-colors flex items-center space-x-3"
        aria-label="Open model scores"
        title="Model Scores"
      >
        <div className="w-9 h-9 rounded-xl bg-hotspot/10 flex items-center justify-center text-hotspot">
          <BarChart2 size={18} />
        </div>
        <div className="text-left">
          <p className="text-xs text-slate-500 uppercase tracking-widest font-bold">Model</p>
          <p className="text-sm font-bold text-white leading-tight">Scores</p>
        </div>
      </button>

      {/* Background decorations */}
      <div className="absolute top-1/4 -left-64 w-[500px] h-[500px] bg-hotspot/10 rounded-full blur-[140px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-1/4 -right-64 w-[500px] h-[500px] bg-hotspotSecondary/10 rounded-full blur-[140px] pointer-events-none" />

      <div className="max-w-[1600px] mx-auto relative z-10">
        <header className="mb-12 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-2.5 glass rounded-2xl text-hotspotSecondary ring-1 ring-hotspotSecondary/20 ">
              <Dna size={28} strokeWidth={2} />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-500">
                Proteome-X
              </h1>
              <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">Interpretable GNN Explorer</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6 text-sm font-medium text-slate-400">
            <button onClick={() => setActiveModal('docs')} className="hover:text-hotspotSecondary transition-colors">Documentation</button>
            <button onClick={() => setActiveModal('metrics')} className="hover:text-hotspotSecondary transition-colors">Metric Details</button>
            <div className="px-4 py-2 glass rounded-full flex items-center space-x-2 text-hotspotSecondary ring-1 ring-hotspotSecondary/30">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span>GNN Engine Online</span>
            </div>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-4 space-y-6">
            <div className="glass-card p-6 rounded-2xl border-slate-800/50">
              <h2 className="text-lg font-semibold mb-6 flex items-center text-slate-100 uppercase tracking-tight">
                <Search className="w-4 h-4 mr-3 text-hotspot" />
                Input Parameters
              </h2>
              <form onSubmit={handleSearch} className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1.5 px-1">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Receptor (PDB)</label>
                    <span className="text-[10px] text-slate-500">Model Input A</span>
                  </div>
                  <input
                    type="text"
                    placeholder="e.g. 1BRS"
                    className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-1 focus:ring-hotspot transition-all font-mono uppercase text-sm"
                    value={pdb1}
                    onChange={(e) => setPdb1(e.target.value.toUpperCase())}
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-1.5 px-1">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Ligand (PDB)</label>
                    <span className="text-[10px] text-slate-500">Model Input B</span>
                  </div>
                  <input
                    type="text"
                    placeholder="e.g. 1CHO"
                    className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-1 focus:ring-hotspotSecondary transition-all font-mono uppercase text-sm"
                    value={pdb2}
                    onChange={(e) => setPdb2(e.target.value.toUpperCase())}
                  />
                </div>
                {error && <p className="text-red-400 text-[11px] font-medium bg-red-400/10 p-2 rounded-lg border border-red-400/20">{error}</p>}
                
                <button
                  type="submit"
                  disabled={loading}
                  className="group w-full mt-4 bg-white hover:bg-slate-100 text-black font-bold py-4 px-4 rounded-xl transition-all duration-300 flex items-center justify-center hover:shadow-[0_0_30px_rgba(255,255,255,0.15)] disabled:opacity-50"
                >
                  {loading ? (
                    <><Loader2 className="w-5 h-5 mr-3 animate-spin text-hotspot" /> Extracting Graphs...</>
                  ) : (
                    <>Run Discovery Engine <ExternalLink size={16} className="ml-2 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" /></>
                  )}
                </button>
              </form>
            </div>

            {result && (
              <div className="space-y-6">
                <div className="glass-card p-6 rounded-2xl border-slate-800/50 animate-[fadeIn_0.5s_ease-out]">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Interaction Probability</h3>
                    <div className="flex items-center space-x-2">
                      <button
                        type="button"
                        onClick={() => setActiveModal('metrics')}
                        className="w-6 h-6 rounded-full border border-slate-700 bg-slate-900/60 text-slate-200 hover:text-white hover:border-slate-500 transition-colors text-xs font-black"
                        aria-label="How Interaction Probability is computed"
                        title="How Interaction Probability is computed"
                      >
                        ?
                      </button>
                      <div className="px-2 py-1 bg-hotspot/10 rounded text-[10px] text-hotspot font-mono border border-hotspot/20">
                        {(result.inference_mode || 'onnx').toUpperCase()}
                      </div>
                    </div>
                  </div>
                  
                  <div className="relative h-24 flex items-center justify-center">
                    <div className="absolute inset-0 bg-gradient-to-r from-hotspot/20 to-transparent blur-3xl rounded-full" />
                    <span className="relative text-6xl font-black tracking-tighter text-white">
                      {result.interaction_probability}<span className="text-2xl text-slate-600">%</span>
                    </span>
                  </div>
                  <p className="text-center text-[11px] text-slate-500 mt-2 font-medium">Model output for this PDB pair</p>
                </div>

                <div className="glass-card p-6 rounded-2xl border-slate-800/50 animate-[fadeIn_0.55s_ease-out]">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Evidence Score</h3>
                    <div className="flex items-center space-x-2">
                      <button
                        type="button"
                        onClick={() => setActiveModal('metrics')}
                        className="w-6 h-6 rounded-full border border-slate-700 bg-slate-900/60 text-slate-200 hover:text-white hover:border-slate-500 transition-colors text-xs font-black"
                        aria-label="How Evidence Score is calculated"
                        title="How Evidence Score is calculated"
                      >
                        ?
                      </button>
                      <div className="px-2 py-1 bg-hotspotSecondary/10 rounded text-[10px] text-hotspotSecondary font-mono border border-hotspotSecondary/20">
                        ROBUSTNESS
                      </div>
                    </div>
                  </div>

                  <div className="relative h-24 flex items-center justify-center">
                    <div className="absolute inset-0 bg-gradient-to-r from-hotspotSecondary/20 to-transparent blur-3xl rounded-full" />
                    <span className="relative text-6xl font-black tracking-tighter text-white">
                      {typeof result.evidence_score === 'number' ? Math.round(result.evidence_score) : '--'}
                      <span className="text-2xl text-slate-600">/100</span>
                    </span>
                  </div>
                  <p className="text-center text-[11px] text-slate-500 mt-2 font-medium">Stability + hotspot consistency + input sanity</p>
                </div>

                <div className="glass-card p-6 rounded-2xl border-slate-800/50 animate-[fadeIn_0.6s_ease-out]">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center">
                      <BarChart2 className="w-4 h-4 mr-2 text-hotspotSecondary" />
                      Attention Distribution
                    </h3>
                    <button
                      type="button"
                      onClick={() => setActiveModal('attention')}
                      className="w-6 h-6 rounded-full border border-slate-700 bg-slate-900/60 text-slate-200 hover:text-white hover:border-slate-500 transition-colors text-xs font-black"
                      aria-label="What is Attention Distribution?"
                      title="What is Attention Distribution?"
                    >
                      ?
                    </button>
                  </div>
                  
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart layout="vertical" data={chartData} margin={{ left: -10, right: 10 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 10 }} width={80} />
                        <Tooltip 
                          cursor={{ fill: 'rgba(255,255,255,0.05)' }} 
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', fontSize: '11px' }}
                        />
                        <Bar dataKey="attention" radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={index === 0 ? '#FF5E00' : '#00D4FF'} fillOpacity={1 - (index * 0.15)} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="lg:col-span-8 space-y-6">
            <div className="glass-card p-1 rounded-[2rem] border-slate-800/50 overflow-hidden bg-slate-950/40 min-h-[600px] flex flex-col">
              {result ? (
                <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-px bg-slate-800/20">
                   <div className="relative group bg-slate-950/40 p-4">
                     <div className="absolute top-6 left-6 z-10">
                        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] block mb-1">Topology A</span>
                        <h4 className="text-lg font-bold text-white uppercase">{result.pdb1}</h4>
                     </div>
                     <div className="absolute top-6 right-6 z-10 px-2 py-1 glass rounded-lg border-white/5 flex items-center space-x-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                        <span className="text-[10px] text-slate-400 font-mono">RECEPTOR</span>
                     </div>
                     <ProteinViewer pdbId={result.pdb1} hotspots={result.hotspots} />
                   </div>
                   <div className="relative group bg-slate-950/40 p-4 border-l border-slate-800/50">
                     <div className="absolute top-6 left-6 z-10">
                        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] block mb-1">Topology B</span>
                        <h4 className="text-lg font-bold text-white uppercase">{result.pdb2}</h4>
                     </div>
                     <div className="absolute top-6 right-6 z-10 px-2 py-1 glass rounded-lg border-white/5 flex items-center space-x-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-hotspot" />
                        <span className="text-[10px] text-slate-400 font-mono">LIGAND</span>
                     </div>
                     <ProteinViewer pdbId={result.pdb2} hotspots={result.hotspots} />
                   </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
                   <div className="w-24 h-24 bg-slate-900 rounded-full flex items-center justify-center mb-6 ring-1 ring-white/5">
                    <Activity size={40} className="text-slate-800 animate-pulse" />
                   </div>
                   <h3 className="text-xl font-bold text-slate-300 mb-2">Awaiting Computation</h3>
                   <p className="text-sm text-slate-500 max-w-sm">Enter receptor and ligand IDs to initialize the spatial GNN transformer and detect hotspot interactions.</p>
                </div>
              )}
            </div>

            <div className="glass-card p-6 rounded-2xl border-slate-800/50 bg-gradient-to-r from-slate-950 to-slate-900 border-dashed border-2">
               <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="p-3 bg-hotspot/10 rounded-xl text-hotspot">
                      <Share2 size={24} />
                    </div>
                    <div>
                      <h4 className="font-bold text-white tracking-tight">Deploy & Share</h4>
                      <p className="text-xs text-slate-400">Ready to present this tool to researchers or recruiters?</p>
                    </div>
                  </div>
                  <button onClick={copyShareLink} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white text-sm font-bold rounded-xl transition-all border border-slate-700 active:scale-95">
                    {showToast ? 'Copied!' : 'Get Shareable Link'}
                  </button>
               </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
