import express from 'express';
import helmet from 'helmet';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

const PORT = Number.parseInt(process.env.PORT ?? '5173', 10);
if (!Number.isFinite(PORT) || PORT <= 0 || PORT >= 65536) {
  throw new Error('Invalid PORT');
}

const BACKEND_URL_RAW = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000';
let BACKEND_URL;
try {
  const u = new URL(BACKEND_URL_RAW);
  if (!['http:', 'https:'].includes(u.protocol)) throw new Error('Unsupported protocol');
  // Remove trailing slash to avoid double-slash joins.
  u.pathname = u.pathname.replace(/\/+$/, '');
  BACKEND_URL = u.toString().replace(/\/+$/, '');
} catch {
  throw new Error('Invalid BACKEND_URL');
}

app.disable('x-powered-by');

app.use(
  helmet({
    // The page loads 3Dmol from a CDN; allow that script.
    contentSecurityPolicy: {
      useDefaults: true,
      directives: {
        "script-src": ["'self'", 'https://3Dmol.org'],
        "connect-src": ["'self'", 'https://files.rcsb.org'],
        "img-src": ["'self'", 'data:'],
        // 3Dmol uses inline styles in some cases; keep permissive for local demo.
        "style-src": ["'self'", "'unsafe-inline'"],
      },
    },
  })
);

app.use(express.json({ limit: '10kb' }));

app.use(express.static(path.join(__dirname, 'public'), { extensions: ['html'] }));

function isValidPdbId(s) {
  return typeof s === 'string' && /^[A-Za-z0-9]{4}$/.test(s.trim());
}

app.get('/api/health', async (_req, res) => {
  try {
    const r = await fetch(`${BACKEND_URL}/health`, { method: 'GET' });
    const text = await r.text();
    res.status(r.status).type('application/json').send(text);
  } catch (e) {
    res.status(502).json({ error: 'Backend unreachable' });
  }
});

app.get('/api/validation/summary', async (_req, res) => {
  try {
    const r = await fetch(`${BACKEND_URL}/validation/summary`, { method: 'GET' });
    const text = await r.text();
    res.status(r.status).type('application/json').send(text);
  } catch {
    res.status(502).json({ error: 'Backend unreachable' });
  }
});

app.post('/api/predict', async (req, res) => {
  const { pdb1, pdb2 } = req.body ?? {};
  if (!isValidPdbId(pdb1) || !isValidPdbId(pdb2)) {
    res.status(400).json({ error: 'pdb1 and pdb2 must be 4-character alphanumeric PDB IDs' });
    return;
  }

  try {
    const r = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdb1: pdb1.trim(), pdb2: pdb2.trim() }),
    });

    const text = await r.text();
    res.status(r.status).type('application/json').send(text);
  } catch {
    res.status(502).json({ error: 'Backend unreachable' });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  // eslint-disable-next-line no-console
  console.log(`Proteome-X Node frontend running on http://localhost:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`Proxying API to ${BACKEND_URL}`);
});
