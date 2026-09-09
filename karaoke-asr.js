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

// Les mêmes paliers que le panneau After Effects. Les tailles sont
// celles des modèles quantifiés réellement téléchargés, une seule fois,
// puis gardés en cache par le navigateur.
// Le suffixe « _timestamped » n'est pas décoratif : ce sont les seules
// variantes exportées avec les attentions croisées du décodeur. Sans
// elles, Whisper transcrit mais ne peut PAS dater les mots — et c'est
// exactement la datation qui nous intéresse.
export const MODELS = {
  petit: { id: 'onnx-community/whisper-tiny_timestamped',  mo: 41,  nom: 'Rapide (tiny)' },
  moyen: { id: 'onnx-community/whisper-base_timestamped',  mo: 77,  nom: 'Équilibré (base)' },
  grand: { id: 'onnx-community/whisper-small_timestamped', mo: 249, nom: 'Précis (small)' }
};

let _pipe = null, _pipeName = '';

/**
 * Charge Whisper. `onProgress` reçoit {status, name, progress} pendant
 * le téléchargement du modèle, pour afficher une vraie barre.
 */
export async function loadASR({ model = 'moyen', onProgress } = {}) {
  const name = (MODELS[model] && MODELS[model].id) || model;
  if (_pipe && _pipeName === name) return _pipe;
  // import dynamique : la page reste utilisable si l'utilisateur ne
  // demande jamais de transcription
  const { pipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js'
  );
  env.allowLocalModels = false;
  // CALCUL PROCESSEUR, TOUJOURS.
  //
  // Sur carte graphique (WebGPU), Whisper ne sort ici que du charabia
  // multilingue en boucle — et, signe qui ne trompe pas, EXACTEMENT la
  // même suite de jetons quels que soient le modèle (base, base
  // _timestamped, small) et la précision (q8, encodeur en fp32). Le
  // modèle n'entend donc pas l'audio du tout : l'encodeur rend du bruit,
  // et le décodeur, n'ayant rien à quoi se raccrocher, radote.
  //
  // Le même audio, le même modèle, en processeur : « Ah SM, tout s'en
  // route pour le stade ». Mesuré sur le chant ASM (67 s) : 12 secondes
  // de calcul, 66 mots datés. Cinq fois plus rapide que le temps réel,
  // donc la lenteur supposée du processeur n'est pas un argument.
  //
  // La panne est silencieuse — pas d'erreur, juste du faux — donc pas
  // question de « tenter la carte graphique et voir ». On ne l'utilise
  // pas.
  _pipe = await pipeline('automatic-speech-recognition', name, {
    dtype: 'q8',                       // quantifié : ~4× plus léger
    device: 'wasm',
    progress_callback: onProgress
  });
  _pipeName = name;
  return _pipe;
}

/**
 * Transcrit un AudioBuffer déjà décodé.
 * Renvoie [{w, s, e}] — un mot, son début, sa fin, en secondes.
 */
export async function transcribe(audioBuffer, { model = 'moyen', language = 'fr', centre = false, onProgress } = {}) {
  // language === 'auto' : on laisse Whisper décider (option simplement omise)
  const asr = await loadASR({ model, onProgress });
  const mono = normalise(toMono16k(audioBuffer, { centre }));

  const COMMON = { task: 'transcribe' };
  if (language && language !== 'auto') COMMON.language = language;

  // ON DÉCOUPE NOUS-MÊMES.
  //
  // Whisper n'écoute que 30 secondes à la fois. Confier le découpage à la
  // bibliothèque (chunk_length_s / stride_length_s) donne un texte juste
  // mais des INSTANTS FAUX : mesuré sur le chant ASM, les quatre
  // premières lignes tombent à 0,1–0,5 s près, et tout ce qui suit dérive
  // de 20 à 40 secondes, jusqu'à se placer après la fin du morceau.
  //
  // Les mêmes fenêtres transcrites une par une, en revanche, sont justes :
  // 0–28 s rend « Ah SM, tout s'en route pour le stade » daté 0,3 → 27,5,
  // et 28–56 s repart proprement à 0. Il suffit donc d'ajouter le décalage
  // de la fenêtre. C'est ce que fait le code ci-dessous.
  const SR = 16000, FEN = 28, RECOUV = 3;      // fenêtre, et son recouvrement
  const PAS = FEN - RECOUV;
  const dur = mono.length / SR;
  const debuts = [];
  for (let t = 0; t < dur; t += PAS) debuts.push(t);
  if (debuts.length > 1 && dur - debuts[debuts.length - 1] < 1.5) debuts.pop();

  let mode = 'word';
  const words = [];

  for (let k = 0; k < debuts.length; k++) {
    const d0 = debuts[k];
    const tranche = mono.slice(Math.round(d0 * SR), Math.round(Math.min(dur, d0 + FEN) * SR));
    if (tranche.length < SR * 0.5) continue;

    let out;
    try {
      out = await asr(tranche, Object.assign({ return_timestamps: 'word' }, COMMON));
    } catch (e) {
      // Filet de sécurité : si la datation par mot n'est pas disponible
      // (modèle sans attentions croisées), on retombe sur la datation par
      // phrase. C'est moins fin, mais le recollage sur vos paroles s'en
      // accommode — il vaut toujours mieux que rien.
      mode = 'phrase';
      out = await asr(tranche, Object.assign({ return_timestamps: true }, COMMON));
    }

    // On coupe le recouvrement en deux : la première moitié appartient à
    // la fenêtre précédente, la seconde à celle-ci. Un mot n'est donc
    // jamais compté deux fois.
    const planche = (k === 0) ? -1 : d0 + RECOUV / 2;
    const plafond = (k === debuts.length - 1) ? dur + 1 : d0 + FEN - RECOUV / 2;

    for (const c of ((out && out.chunks) ? out.chunks : [])) {
      const t = (c.timestamp || [])[0], u = (c.timestamp || [])[1];
      const txt = String(c.text || '').trim();
      if (!txt || typeof t !== 'number') continue;
      const fin = (typeof u === 'number' && u > t) ? u : t + 0.25;

      if (mode === 'word') {
        const abs = d0 + t;
        if (abs < planche || abs >= plafond) continue;
        words.push({ w: txt, s: abs, e: d0 + fin });
        continue;
      }

      // une phrase entière : on répartit sa durée sur ses mots, au prorata
      // du nombre de lettres
      const parts = txt.split(/\s+/).filter(Boolean);
      if (!parts.length) continue;
      const total = parts.reduce((a, w) => a + w.length, 0) || parts.length;
      let cur = t;
      for (const w of parts) {
        const dd = (fin - t) * (w.length / total);
        const abs = d0 + cur;
        if (abs >= planche && abs < plafond) words.push({ w, s: abs, e: abs + dd * 0.92 });
        cur += dd;
      }
    }

    // une vraie progression : on sait combien de fenêtres il reste
    if (onProgress) onProgress({ status: 'listening', done: k + 1, total: debuts.length });
  }

  words.sort((a, b) => a.s - b.s);
  words.mode = mode;
  words.loop = detectLoop(words);
  words.duration = dur;
  return words;
}

/**
 * Whisper, mis en difficulté, se met à répéter indéfiniment la même
 * séquence de mots. C'est reconnaissable : on compte la part des mots
 * qui appartiennent à un motif déjà vu. Au-delà de la moitié, la
 * transcription ne vaut rien — et il vaut mieux le dire que laisser
 * l'utilisateur croire à un problème de calage.
 */
export function detectLoop(words) {
  if (words.length < 40) return 0;
  const seq = words.map(w => w.w);
  const seen = new Map();
  let repeated = 0;
  const K = 6;                                   // motifs de six mots
  for (let i = 0; i + K <= seq.length; i++) {
    const key = seq.slice(i, i + K).join(' ');
    const n = (seen.get(key) || 0) + 1;
    seen.set(key, n);
    if (n > 1) repeated++;
  }
  return repeated / Math.max(1, seq.length - K + 1);
}

/**
 * Mise en avant du CENTRE stéréo.
 *
 * Dans un enregistrement, la voix est presque toujours au centre : elle
 * apparaît identique à gauche et à droite. L'ambiance, la foule, les
 * instruments larges, eux, diffèrent d'un canal à l'autre. En comparant
 * les deux canaux fréquence par fréquence, on peut donc atténuer ce qui
 * est large et garder ce qui est centré — ce qui donne à Whisper une
 * voix moins encombrée.
 *
 * L'effet dépend entièrement du fichier : sur un enregistrement quasi
 * mono, il n'y a rien à retirer et le résultat est inchangé.
 */
export function centreEmphasis(L, R, { strength = 2 } = {}) {
  const n = Math.min(L.length, R.length);
  const win = 1024, hop = 256, bins = win / 2 + 1;
  const fft = makeFFT(win), ifft = makeIFFT(win);
  const w = new Float32Array(win);
  for (let i = 0; i < win; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win - 1));

  const out = new Float32Array(n), norm = new Float32Array(n);
  const lr = new Float32Array(win), li = new Float32Array(win);
  const rr = new Float32Array(win), ri = new Float32Array(win);

  for (let off = 0; off + win <= n; off += hop) {
    for (let i = 0; i < win; i++) {
      lr[i] = L[off + i] * w[i]; li[i] = 0;
      rr[i] = R[off + i] * w[i]; ri[i] = 0;
    }
    fft(lr, li); fft(rr, ri);
    for (let k = 0; k < bins; k++) {
      const mr = (lr[k] + rr[k]) / 2, mi = (li[k] + ri[k]) / 2;
      const sr = (lr[k] - rr[k]) / 2, si = (li[k] - ri[k]) / 2;
      const mMag = Math.hypot(mr, mi), sMag = Math.hypot(sr, si);
      // 1 quand tout est centré, vers 0 quand les côtés dominent
      const g = mMag / (mMag + strength * sMag + 1e-9);
      const vr = mr * g, vi = mi * g;
      lr[k] = vr; li[k] = vi;
      if (k > 0 && k < win / 2) {                 // symétrie hermitienne
        lr[win - k] = vr; li[win - k] = -vi;
      }
    }
    ifft(lr, li);
    for (let i = 0; i < win; i++) {
      out[off + i] += lr[i] * w[i];
      norm[off + i] += w[i] * w[i];
    }
  }
  for (let i = 0; i < n; i++) if (norm[i] > 1e-6) out[i] /= norm[i];
  return out;
}

/**
 * Remise à niveau. Un signal trop faible est une cause connue de
 * divagation : Whisper prend le quasi-silence pour de la parole et part
 * en boucle. On ramène donc la crête à 0,9 avant de lui donner l'audio.
 */
export function normalise(x) {
  let peak = 0;
  for (let i = 0; i < x.length; i++) { const v = Math.abs(x[i]); if (v > peak) peak = v; }
  if (!(peak > 1e-6) || peak > 0.85) return x;      // déjà correct, ou muet
  const g = 0.9 / peak;
  for (let i = 0; i < x.length; i++) x[i] *= g;
  return x;
}

/** Whisper attend du mono à 16 kHz. */
export function toMono16k(buf, { centre = false } = {}) {
  const sr = 16000;
  const n = Math.round(buf.duration * sr);
  const out = new Float32Array(n);
  const chans = [];
  for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
  const ratio = buf.sampleRate / sr;

  if (centre && chans.length >= 2) {
    // on rééchantillonne les deux canaux, puis on met le centre en avant
    const Lr = new Float32Array(n), Rr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const src = i * ratio;
      const i0 = Math.floor(src), i1 = Math.min(buf.length - 1, i0 + 1), f = src - i0;
      Lr[i] = chans[0][i0] * (1 - f) + chans[0][i1] * f;
      Rr[i] = chans[1][i0] * (1 - f) + chans[1][i1] * f;
    }
    return centreEmphasis(Lr, Rr);
  }

  for (let i = 0; i < n; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src), i1 = Math.min(buf.length - 1, i0 + 1), f = src - i0;
    let v = 0;
    for (let c = 0; c < chans.length; c++) v += chans[c][i0] * (1 - f) + chans[c][i1] * f;
    out[i] = v / chans.length;
  }
  return out;
}

/* --- FFT, pour la mise en avant du centre --- */
function makeFFT(n, inverse) {
  const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2);
  const sgn = inverse ? 1 : -1;
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos(sgn * 2 * Math.PI * i / n);
    sin[i] = Math.sin(sgn * 2 * Math.PI * i / n);
  }
  const rev = new Uint32Array(n);
  let bits = 0; while ((1 << bits) < n) bits++;
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    rev[i] = r;
  }
  return function (re, im) {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tr = re[l] * cos[k] - im[l] * sin[k];
          const ti = re[l] * sin[k] + im[l] * cos[k];
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr; im[j] += ti;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  };
}
function makeIFFT(n) { return makeFFT(n, true); }

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
export function alignToLyrics(asrWords, lyricLines, duree) {
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
  fillGaps(lyr, asr, duree);

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

function fillGaps(lyr, asr, duree) {
  const n = lyr.length;
  const firstT = asr.length ? asr[0].s : 0;
  const lastT = asr.length ? asr[asr.length - 1].e : 0;
  // Rien ne doit être placé après la fin du morceau. Sans cette borne, une
  // queue de paroles que Whisper n'a pas entendues s'étalait de 0,05 s en
  // 0,05 s et finissait 20 secondes APRÈS le dernier son — mesuré sur le
  // chant ASM : dernière ligne à 81,98 s pour un morceau de 62,23 s.
  const FIN = (typeof duree === 'number' && duree > 0) ? duree : Math.max(lastT, 1);
  let i = 0;
  while (i < n) {
    if (typeof lyr[i].s === 'number') { i++; continue; }
    let j = i;
    while (j < n && typeof lyr[j].s !== 'number') j++;
    const before = i > 0 ? lyr[i - 1].e : firstT;
    // Une queue sans point d'appui va jusqu'à la fin du morceau, et s'y
    // répartit — plutôt que de défiler à pas fixe et de déborder.
    const after = j < n ? lyr[j].s : Math.max(before + 0.2, FIN);
    const span = Math.max(0.12 * (j - i), after - before);
    const step = span / (j - i);
    for (let k = i; k < j; k++) {
      lyr[k].s = before + step * (k - i);
      lyr[k].e = lyr[k].s + step * 0.92;
    }
    i = j;
  }
  // ordre strictement croissant, quoi qu'il arrive — mais jamais au-delà
  // de la fin : au pire on tasse les derniers mots contre elle.
  for (let k = 1; k < n; k++) {
    if (!(lyr[k].s > lyr[k - 1].s)) lyr[k].s = lyr[k - 1].s + 0.05;
    if (!(lyr[k].e > lyr[k].s)) lyr[k].e = lyr[k].s + 0.12;
  }
  for (let k = n - 1; k >= 0; k--) {
    if (lyr[k].s > FIN) lyr[k].s = FIN;
    if (k < n - 1 && lyr[k].s >= lyr[k + 1].s) lyr[k].s = lyr[k + 1].s - 0.02;
    if (lyr[k].s < 0) lyr[k].s = 0;
    if (!(lyr[k].e > lyr[k].s)) lyr[k].e = lyr[k].s + 0.12;
    if (lyr[k].e > FIN) lyr[k].e = FIN;
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
  const r = alignToLyrics(asr, lines, audioBuffer.duration);
  return {
    audio: opts.name || 'chanson',
    duration: audioBuffer.duration,
    language: opts.language || 'fr',
    model: 'whisper-' + (opts.model || 'moyen'),
    matched: r.matched,
    heard: asr.map(x => x.w).join(' '),        // la transcription brute, pour diagnostic
    heardCount: asr.length,
    timing: asr.mode || 'word',                // 'word' ou 'phrase' si repli
    loop: asr.loop || 0,                       // part de la transcription en boucle
    words: r.words.map(w => ({
      w: w.w, s: Math.round(w.s * 1000) / 1000, e: Math.round(w.e * 1000) / 1000,
      line: w.line, lvl: 1
    }))
  };
}
