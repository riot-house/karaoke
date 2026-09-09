/*
 * karaoke-asr.js — calage automatique des paroles, dans le navigateur.
 *
 * À déposer sur karaoke.riothouse.fr, à côté de la page du studio.
 * NE FONCTIONNE PAS dans un artefact Claude : la page y est empêchée de
 * télécharger un modèle. Sur votre domaine, aucune de ces limites.
 *
 * Deux étapes, séparées à dessein :
 *
 *   1. TRANSCRIRE  — Whisper écoute et écrit ce qu'il entend, avec un
 *      horodatage par mot. Il se trompe : il invente, oublie, coupe mal.
 *
 *   2. RÉALIGNER   — on ne garde PAS le texte de Whisper. On garde ses
 *      instants, et on les recolle sur VOS paroles, qui font foi. Un mot
 *      mal transcrit garde donc son orthographe d'origine et récupère
 *      quand même le bon instant.
 *
 * C'est exactement la chaîne du panneau After Effects (faster-whisper +
 * réalignement), transposée côté navigateur.
 */

'use strict';

/* ------------------------------------------------------------------ *
 * 1. Transcription
 * ------------------------------------------------------------------ */

const MODELS = {
  // Le plus petit qui tienne la route en français. Premier chargement
  // long, ensuite mis en cache par le navigateur.
  petit:  'Xenova/whisper-tiny',
  moyen:  'Xenova/whisper-base',
  grand:  'Xenova/whisper-small'
};

let _pipe = null, _pipeName = '';

/**
 * Charge Whisper. `onProgress` reçoit {status, name, progress} pendant
 * le téléchargement du modèle, pour afficher une vraie barre.
 */
export async function loadASR({ model = 'moyen', onProgress } = {}) {
  const name = MODELS[model] || model;
  if (_pipe && _pipeName === name) return _pipe;
  // import dynamique : la page reste utilisable si l'utilisateur ne
  // demande jamais de transcription
  const { pipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js'
  );
  env.allowLocalModels = false;
  _pipe = await pipeline('automatic-speech-recognition', name, {
    dtype: 'q8',                       // quantifié : ~4× plus léger
    device: (navigator.gpu ? 'webgpu' : 'wasm'),
    progress_callback: onProgress
  });
  _pipeName = name;
  return _pipe;
}

/**
 * Transcrit un AudioBuffer déjà décodé.
 * Renvoie [{w, s, e}] — un mot, son début, sa fin, en secondes.
 */
export async function transcribe(audioBuffer, { model = 'moyen', language = 'fr', onProgress } = {}) {
  const asr = await loadASR({ model, onProgress });
  const mono = toMono16k(audioBuffer);
  const out = await asr(mono, {
    language,
    task: 'transcribe',
    return_timestamps: 'word',
    chunk_length_s: 30,
    stride_length_s: 5
  });
  const chunks = out && out.chunks ? out.chunks : [];
  const words = [];
  for (const c of chunks) {
    const t = (c.timestamp || [])[0], u = (c.timestamp || [])[1];
    const w = String(c.text || '').trim();
    if (!w || typeof t !== 'number') continue;
    words.push({ w, s: t, e: (typeof u === 'number' && u > t) ? u : t + 0.25 });
  }
  return words;
}

/** Whisper attend du mono à 16 kHz. */
export function toMono16k(buf) {
  const sr = 16000;
  const n = Math.round(buf.duration * sr);
  const out = new Float32Array(n);
  const chans = [];
  for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
  const ratio = buf.sampleRate / sr;
  for (let i = 0; i < n; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src), i1 = Math.min(buf.length - 1, i0 + 1), f = src - i0;
    let v = 0;
    for (let c = 0; c < chans.length; c++) v += chans[c][i0] * (1 - f) + chans[c][i1] * f;
    out[i] = v / chans.length;
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * 2. Réalignement sur les paroles fournies
 * ------------------------------------------------------------------ */

/** Forme comparable d'un mot : sans accent, sans ponctuation, en bas de casse. */
export function norm(w) {
  return String(w)
    .toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]/g, '');
}

/** Ressemblance de deux mots, 0 à 1 — bigrammes communs (Sørensen–Dice). */
export function similar(a, b) {
  if (!a || !b) return 0;
  if (a === b) return 1;
  if (a.length < 2 || b.length < 2) return a === b ? 1 : 0;
  const grams = s => {
    const m = new Map();
    for (let i = 0; i < s.length - 1; i++) {
      const g = s.slice(i, i + 2);
      m.set(g, (m.get(g) || 0) + 1);
    }
    return m;
  };
  const A = grams(a), B = grams(b);
  let inter = 0;
  for (const [g, n] of A) inter += Math.min(n, B.get(g) || 0);
  return (2 * inter) / (a.length - 1 + b.length - 1);
}

/**
 * Aligne la transcription sur les paroles (Needleman–Wunsch), puis
 * transporte les instants.
 *
 * @param asrWords   [{w, s, e}] tels que sortis de Whisper
 * @param lyricLines [[mot, mot, ...], ...] vos paroles, ligne par ligne
 * @returns {words, lines, matched} — mots datés, débuts de ligne, et la
 *          part de paroles réellement retrouvées dans l'audio.
 */
export function alignToLyrics(asrWords, lyricLines) {
  // paroles à plat, en gardant l'appartenance à une ligne
  const lyr = [];
  lyricLines.forEach((line, li) =>
    line.forEach(w => lyr.push({ w, n: norm(w), line: li })));
  const asr = asrWords.map(x => ({ ...x, n: norm(x.w) })).filter(x => x.n);

  const N = lyr.length, M = asr.length;
  if (!N) return { words: [], lines: [], matched: 0 };
  if (!M) return { words: [], lines: [], matched: 0 };

  const GAP = -0.6, MISS = -0.9;
  const F = new Float32Array((N + 1) * (M + 1));
  const P = new Uint8Array((N + 1) * (M + 1));      // 1 diag, 2 haut, 3 gauche
  const at = (i, j) => i * (M + 1) + j;
  for (let i = 1; i <= N; i++) { F[at(i, 0)] = i * GAP; P[at(i, 0)] = 2; }
  for (let j = 1; j <= M; j++) { F[at(0, j)] = j * GAP; P[at(0, j)] = 3; }
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= M; j++) {
      const sim = similar(lyr[i - 1].n, asr[j - 1].n);
      const d = F[at(i - 1, j - 1)] + (sim >= 0.62 ? sim : MISS);
      const u = F[at(i - 1, j)] + GAP;
      const l = F[at(i, j - 1)] + GAP;
      let best = d, p = 1;
      if (u > best) { best = u; p = 2; }
      if (l > best) { best = l; p = 3; }
      F[at(i, j)] = best; P[at(i, j)] = p;
    }
  }
  // remontée : on date les paroles retrouvées
  let i = N, j = M, matched = 0;
  while (i > 0 || j > 0) {
    const p = P[at(i, j)];
    if (p === 1) {
      const sim = similar(lyr[i - 1].n, asr[j - 1].n);
      if (sim >= 0.62) { lyr[i - 1].s = asr[j - 1].s; lyr[i - 1].e = asr[j - 1].e; matched++; }
      i--; j--;
    } else if (p === 2) i--;
    else if (p === 3) j--;
    else break;
  }

  // Les trous : un mot que Whisper n'a pas entendu reçoit un instant
  // interpolé entre ses deux voisins datés, au prorata des syllabes.
  fillGaps(lyr, asr);

  // débuts de ligne
  const lines = [];
  lyr.forEach(x => { if (lines[x.line] === undefined && typeof x.s === 'number') lines[x.line] = x.s; });
  for (let k = 0; k < lyricLines.length; k++) if (lines[k] === undefined) lines[k] = null;

  return {
    words: lyr.map(x => ({ w: x.w, s: x.s, e: x.e, line: x.line })),
    lines,
    matched: matched / N
  };
}

function fillGaps(lyr, asr) {
  const n = lyr.length;
  const firstT = asr.length ? asr[0].s : 0;
  const lastT = asr.length ? asr[asr.length - 1].e : 0;
  let i = 0;
  while (i < n) {
    if (typeof lyr[i].s === 'number') { i++; continue; }
    let j = i;
    while (j < n && typeof lyr[j].s !== 'number') j++;
    const before = i > 0 ? lyr[i - 1].e : firstT;
    const after = j < n ? lyr[j].s : lastT;
    const span = Math.max(0.12 * (j - i), after - before);
    const step = span / (j - i);
    for (let k = i; k < j; k++) {
      lyr[k].s = before + step * (k - i);
      lyr[k].e = lyr[k].s + step * 0.92;
    }
    i = j;
  }
  // ordre strictement croissant, quoi qu'il arrive
  for (let k = 1; k < n; k++) {
    if (!(lyr[k].s > lyr[k - 1].s)) lyr[k].s = lyr[k - 1].s + 0.05;
    if (!(lyr[k].e > lyr[k].s)) lyr[k].e = lyr[k].s + 0.12;
  }
}

/* ------------------------------------------------------------------ *
 * 3. Enchaînement complet
 * ------------------------------------------------------------------ */

/**
 * De l'audio + les paroles collées, au format JSON que lit le studio
 * (et le panneau After Effects).
 */
export async function autoAlign(audioBuffer, lyricsText, opts = {}) {
  const lines = String(lyricsText).split(/\r?\n/)
    .map(l => l.trim()).filter(l => l.length)
    .map(l => l.split(/\s+/));
  const asr = await transcribe(audioBuffer, opts);
  const r = alignToLyrics(asr, lines);
  return {
    audio: opts.name || 'chanson',
    duration: audioBuffer.duration,
    language: opts.language || 'fr',
    model: 'whisper-' + (opts.model || 'moyen'),
    matched: r.matched,
    words: r.words.map(w => ({
      w: w.w, s: Math.round(w.s * 1000) / 1000, e: Math.round(w.e * 1000) / 1000,
      line: w.line, lvl: 1
    }))
  };
}
