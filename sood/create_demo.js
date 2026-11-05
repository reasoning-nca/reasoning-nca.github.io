// Reasoning NCA: Sudoku-OOD (SOOD) Demo
// Implements parallel multi-sample rollouts with confidence scoring,
// damage/noise brush injection, and SwissGL rendering.

export function createDemoSood(glslOrDivId, maybeDivId) {
  let divId = 'demo';
  let passedGlsl = null;
  if (typeof glslOrDivId === 'string') {
    divId = glslOrDivId;
  } else if (glslOrDivId) {
    passedGlsl = glslOrDivId;
    if (typeof maybeDivId === 'string') divId = maybeDivId;
  }

  const root = document.getElementById(divId);
  if (!root) {
    console.error(`Root element #${divId} not found.`);
    return;
  }

  const $ = (selector) => root.querySelector(selector);

  // Setup offscreen WebGL canvas for SwissGL digit rendering
  const renderCanvas = document.createElement('canvas');
  renderCanvas.width = 256;
  renderCanvas.height = 256;
  const gl = renderCanvas.getContext('webgl2', {
    alpha: false,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  const glsl = passedGlsl || SwissGL(gl || renderCanvas);
  const sourceCanvas = (glsl && glsl.gl && glsl.gl.canvas) || renderCanvas;

  const H = 9, W = 9;
  const CHN = 128;
  const INPUT_EMB_D = 32;
  const NUM_UNITS = 4;
  const HEADS = 16;
  const HEAD_DIM = 8; // 128 / 16

  const dset_path = '../assets/json/sooddata.json';
  const model_path = '../assets/json/soodnca.json';

  let dsets = {};
  let model = {};
  let digits = {};

  // Shared puzzle data
  const input_masks = new Uint8Array(81);
  const input_digits = new Uint8Array(81);
  const target_digits = new Uint8Array(81);

  // Precomputed token embedding norms
  const ie_norms = new Float32Array(10);

  // Neighbor offsets for 3x3 wrapping torus
  const NEIGHBORS = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 0], [0, 1],
    [1, -1], [1, 0], [1, 1],
  ];

  // Reusable buffers for step computation
  const val_buf = new Float32Array(81 * HEADS * HEAD_DIM);
  const z_buf = new Float32Array(81 * CHN);
  const h_buf = new Float32Array(512);
  const dx_buf = new Float32Array(CHN);

  // Parallel trial instances (up to 20)
  const MAX_SAMPLES = 20;
  const trials = [];
  for (let t = 0; t < MAX_SAMPLES; t++) {
    trials.push({
      x_state: new Float32Array(81 * CHN),
      pred_digits: new Float32Array(81),
      confidence: 0.0,
      is_correct: false,
      correct_streak: 0,
      canvas: null,
      ctx: null,
      cardEl: null,
      titleEl: null,
      confEl: null,
      badgeEl: null,
    });
  }

  // SwissGL textures
  let input_grid_tex = null;
  let target_grid_tex = null;
  const pred_textures = new Array(MAX_SAMPLES).fill(null);

  let step_count = 0;
  let lastStepTime = 0;
  let currentBestIdx = 0;

  const params = {
    difficulty: '60',
    num_samples: 8,
    runModel: false,
    stepFps: 15,
    fireRate: 0.8,
    noiseRadius: 1, // 0: 1x1, 1: 3x3, 2: 5x5, 3: 7x7, 4: 9x9
    noiseStd: 0.3,  // 0 to 1
  };

  function decodeBase64(dataStr, dtype) {
    const binary = atob(dataStr);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    if (dtype === 'float32') {
      return new Float32Array(bytes.buffer);
    } else if (dtype === 'uint8' || dtype === 'bool') {
      return bytes;
    } else if (dtype === 'int32') {
      return new Int32Array(bytes.buffer);
    }
    return bytes;
  }

  function randGaussian(mean, std) {
    let u1 = Math.random();
    let u2 = Math.random();
    while (u1 <= 1e-7) u1 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  }

  function createDigitTextures() {
    const d_tex = {};
    for (let d = 0; d <= 9; d++) {
      const off = document.createElement('canvas');
      off.width = 128;
      off.height = 128;
      const ctx = off.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, 128, 128);

      if (d > 0) {
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 64px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(d.toString(), 64, 68);
      }

      d_tex['d' + d] = glsl({}, { data: off, tag: 'd' + d });
    }
    return d_tex;
  }

  function updateStepCounter() {
    const el = $('#step-count');
    if (el) el.textContent = `Step: ${step_count}`;
  }

  function updatePlayPauseUI() {
    const playIcon = $('#play');
    const pauseIcon = $('#pause');
    if (playIcon && pauseIcon) {
      if (params.runModel) {
        playIcon.style.display = 'none';
        pauseIcon.style.display = 'block';
      } else {
        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
      }
    }
  }

  function predictTrial(t) {
    const trial = trials[t];
    const x_state = trial.x_state;
    const pred_digits = trial.pred_digits;
    const ie = model.input_embed;
    let correct_count = 0;
    let total_conf = 0.0;

    for (let i = 0; i < 81; i++) {
      let dot_sq = 0.0;
      const base_x = i * CHN;
      for (let c = 0; c < INPUT_EMB_D; c++) {
        const val = x_state[base_x + c];
        dot_sq += val * val;
      }
      const x_norm = Math.sqrt(dot_sq) + 1e-6;

      let best_sim = -1e9;
      let best_digit = 0;

      for (let d = 0; d < 10; d++) {
        let dot = 0.0;
        const base_ie = d * INPUT_EMB_D;
        for (let c = 0; c < INPUT_EMB_D; c++) {
          dot += x_state[base_x + c] * ie[base_ie + c];
        }
        const sim = dot / (x_norm * ie_norms[d]);
        if (sim > best_sim) {
          best_sim = sim;
          best_digit = d;
        }
      }

      pred_digits[i] = best_digit;
      if (best_digit === target_digits[i]) {
        correct_count++;
      }
      total_conf += best_sim;
    }

    const matches_target = (correct_count === 81);
    if (matches_target) {
      trial.correct_streak = (trial.correct_streak || 0) + 1;
    } else {
      trial.correct_streak = 0;
    }

    trial.is_correct = (trial.correct_streak >= 10);
    // Round confidence to 2 decimals
    trial.confidence = Math.round((total_conf / 81.0) * 100) / 100;
  }

  function updateHighlights() {
    if (currentBestIdx >= params.num_samples) {
      currentBestIdx = 0;
    }

    let currentBestConf = trials[currentBestIdx]
      ? trials[currentBestIdx].confidence
      : -Infinity;

    // Switch best only when confidence is strictly higher
    for (let t = 0; t < params.num_samples; t++) {
      if (trials[t].confidence > currentBestConf) {
        currentBestConf = trials[t].confidence;
        currentBestIdx = t;
      }
    }

    for (let t = 0; t < params.num_samples; t++) {
      const trial = trials[t];
      if (!trial.cardEl) continue;

      if (trial.confEl) {
        trial.confEl.textContent = trial.confidence.toFixed(2);
      }

      // Checkmark & green border if correct
      if (trial.is_correct) {
        trial.cardEl.classList.add('correct');
      } else {
        trial.cardEl.classList.remove('correct');
      }

      // Blue border if current best confidence
      if (t === currentBestIdx) {
        trial.cardEl.classList.add('highest-conf');
      } else {
        trial.cardEl.classList.remove('highest-conf');
      }
    }
  }

  function renderTrial(t) {
    const trial = trials[t];
    if (!trial.ctx) return;

    pred_textures[t] = glsl(
      {},
      { size: [W, H], format: 'r32f', data: trial.pred_digits, tag: 'pred_' + t }
    );

    glsl({
      ...digits,
      pred_grid: pred_textures[t],
      input_grid: input_grid_tex,
      target_grid: target_grid_tex,
      H: 9,
      W: 9,
      Grid: [9, 9],
      Aspect: 'fit',
      Clear: 1.0,
      VP: `
        ivec2 id = ivec2(ID.x, int(H) - ID.y - 1);
        VPos.xy = (vec2(id.xy) - vec2(W, H) / 2.0 + 0.5) / (W / 2.0) + XY / W;
      `,
      FP: `
        vec2 uv = vec2(UV.x, 1.0 - UV.y);
        int rx = ID.x;
        int ry = ID.y;

        vec4 bg = vec4(1.0, 1.0, 1.0, 1.0);

        // Thick borders separating 3x3 blocks and board edges
        bool thickX = (rx % 3 == 0 && uv.x <= 0.05) || (rx == 8 && uv.x >= 0.95);
        bool thickY = (ry % 3 == 0 && uv.y <= 0.05) || (ry == 8 && uv.y >= 0.95);
        if (thickX || thickY) {
          FOut = vec4(0.12, 0.12, 0.16, 1.0);
          return;
        }

        // Subtle cell grid lines
        bool thinX = (uv.x <= 0.02 || uv.x >= 0.98);
        bool thinY = (uv.y <= 0.02 || uv.y >= 0.98);
        if (thinX || thinY) {
          FOut = vec4(0.78, 0.80, 0.84, 1.0);
          return;
        }

        uint digit = uint(pred_grid(ID.xy).r);
        uint indigit = uint(input_grid(ID.xy).r);
        uint gtdigit = uint(target_grid(ID.xy).r);

        vec4 digitSample = vec4(1.0);
        switch (digit) {
          case 0u: digitSample = d0(uv); break;
          case 1u: digitSample = d1(uv); break;
          case 2u: digitSample = d2(uv); break;
          case 3u: digitSample = d3(uv); break;
          case 4u: digitSample = d4(uv); break;
          case 5u: digitSample = d5(uv); break;
          case 6u: digitSample = d6(uv); break;
          case 7u: digitSample = d7(uv); break;
          case 8u: digitSample = d8(uv); break;
          case 9u: digitSample = d9(uv); break;
          default: digitSample = d0(uv);
        }

        float textMask = 1.0 - digitSample.r;

        if (indigit > 0u) {
          // Given clue digit: dark slate
          vec4 clueColor = vec4(0.08, 0.08, 0.12, 1.0);
          FOut = mix(bg, clueColor, textMask);
        } else {
          if (digit == 0u) {
            FOut = bg;
          } else if (digit == gtdigit) {
            // Correct prediction: vibrant emerald green
            vec4 correctColor = vec4(0.0, 0.55, 0.2, 1.0);
            FOut = mix(bg, correctColor, textMask);
          } else {
            // Incorrect prediction: bold crimson red
            vec4 wrongColor = vec4(0.85, 0.12, 0.12, 1.0);
            FOut = mix(bg, wrongColor, textMask);
          }
        }
      `,
    });

    trial.ctx.drawImage(
      sourceCanvas,
      0,
      0,
      trial.canvas.width,
      trial.canvas.height
    );
  }

  function renderAllTrials() {
    for (let t = 0; t < params.num_samples; t++) {
      renderTrial(t);
    }
  }

  function reset() {
    step_count = 0;
    currentBestIdx = 0;
    updateStepCounter();
    const ie = model.input_embed;

    for (let t = 0; t < MAX_SAMPLES; t++) {
      trials[t].correct_streak = 0;
      trials[t].is_correct = false;
      const x_state = trials[t].x_state;

      for (let i = 0; i < 81; i++) {
        const digit = input_digits[i];
        const base_x = i * CHN;

        if (digit > 0) {
          // Clue cell: first 32 channels initialized to token embedding
          for (let c = 0; c < INPUT_EMB_D; c++) {
            x_state[base_x + c] = ie[digit * INPUT_EMB_D + c];
          }
          for (let c = INPUT_EMB_D; c < CHN; c++) {
            x_state[base_x + c] = randGaussian(0, 0.15);
          }
        } else {
          // Empty cell: all 128 channels initialized with fresh Gaussian noise
          for (let c = 0; c < CHN; c++) {
            x_state[base_x + c] = randGaussian(0, 0.15);
          }
        }

        // Group norm across 4 units (each unit is 32 channels)
        for (let u = 0; u < NUM_UNITS; u++) {
          let sum_sq = 0.0;
          const base_u = base_x + u * 32;
          for (let c = 0; c < 32; c++) {
            sum_sq += x_state[base_u + c] * x_state[base_u + c];
          }
          const norm = Math.sqrt(sum_sq) + 1e-6;
          for (let c = 0; c < 32; c++) {
            x_state[base_u + c] /= norm;
          }
        }
      }

      predictTrial(t);
    }

    updateHighlights();
    renderAllTrials();
  }

  function newSudoku() {
    const dset = dsets[params.difficulty];
    if (!dset) return;

    const num_samples = dset.input_masks.shape[0];
    const sample_idx = Math.floor(Math.random() * num_samples);

    const mask_raw = dset.input_masks.data.slice(
      sample_idx * 81,
      (sample_idx + 1) * 81
    );
    const target_raw = dset.target_grids.data.slice(
      sample_idx * 81,
      (sample_idx + 1) * 81
    );

    for (let i = 0; i < 81; i++) {
      input_masks[i] = mask_raw[i] ? 1 : 0;
      target_digits[i] = target_raw[i];
      input_digits[i] = mask_raw[i] ? target_raw[i] : 0;
    }

    input_grid_tex = glsl(
      {},
      {
        size: [W, H],
        format: 'r32f',
        data: new Float32Array(input_digits),
        tag: 'input_grid',
      }
    );
    target_grid_tex = glsl(
      {},
      {
        size: [W, H],
        format: 'r32f',
        data: new Float32Array(target_digits),
        tag: 'target_grid',
      }
    );

    currentBestIdx = 0;
    reset();
  }

  function stepTrial(t) {
    const x_state = trials[t].x_state;
    const v = model.v;
    const fa = model.fixed_attention;
    const w1 = model.w1;
    const b1 = model.b1;
    const w2 = model.w2;
    const b2 = model.b2;

    // 1. Project all 81 cells with V matrix: [81, 16, 8]
    for (let i = 0; i < 81; i++) {
      const base_x = i * CHN;
      const base_val_cell = i * 128;
      for (let m = 0; m < HEADS; m++) {
        const base_v_head = m * 1024;
        const base_val = base_val_cell + m * HEAD_DIM;
        for (let d = 0; d < HEAD_DIM; d++) {
          let dot = 0.0;
          const base_v = base_v_head + d * CHN;
          for (let c = 0; c < CHN; c++) {
            dot += v[base_v + c] * x_state[base_x + c];
          }
          val_buf[base_val + d] = dot;
        }
      }
    }

    // 2. Perceive z: linear combination of neighbors weighted by fixed_attention scores
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const i = r * 9 + c;
        const base_z_cell = i * CHN;
        const base_fa_cell = i * 144;

        for (let m = 0; m < HEADS; m++) {
          const base_fa = base_fa_cell + m * 9;
          const base_z = base_z_cell + m * HEAD_DIM;

          for (let d = 0; d < HEAD_DIM; d++) {
            let z_val = 0.0;
            for (let n = 0; n < 9; n++) {
              const dr = NEIGHBORS[n][0];
              const dc = NEIGHBORS[n][1];
              const nr = (r + dr + 9) % 9;
              const nc = (c + dc + 9) % 9;
              const ni = nr * 9 + nc;

              const score = fa[base_fa + n];
              z_val += score * val_buf[(ni * HEADS + m) * HEAD_DIM + d];
            }
            z_buf[base_z + d] = z_val;
          }
        }
      }
    }

    // 3. Update MLP + Async fire mask + Clue mask + Tangent projection + Group norm
    for (let i = 0; i < 81; i++) {
      if (Math.random() > params.fireRate) {
        continue;
      }

      const base_z = i * CHN;
      const base_x = i * CHN;

      // Layer 1: ReLU(W1 @ z + b1)
      for (let j = 0; j < 512; j++) {
        let dot = b1[j];
        const base_w1 = j * CHN;
        for (let c = 0; c < CHN; c++) {
          dot += w1[base_w1 + c] * z_buf[base_z + c];
        }
        h_buf[j] = dot > 0.0 ? dot : 0.0;
      }

      // Layer 2: W2 @ h + b2
      for (let c = 0; c < CHN; c++) {
        let dot = b2[c];
        const base_w2 = c * 512;
        for (let j = 0; j < 512; j++) {
          dot += w2[base_w2 + j] * h_buf[j];
        }
        dx_buf[c] = dot;
      }

      // Clue cells keep token embedding channels fixed
      if (input_digits[i] > 0) {
        for (let c = 0; c < INPUT_EMB_D; c++) {
          dx_buf[c] = 0.0;
        }
      }

      // Tangent projection for each unit
      for (let u = 0; u < NUM_UNITS; u++) {
        const base_u = base_x + u * 32;
        const base_dx_u = u * 32;

        let dot = 0.0;
        for (let c = 0; c < 32; c++) {
          dot += dx_buf[base_dx_u + c] * x_state[base_u + c];
        }
        for (let c = 0; c < 32; c++) {
          dx_buf[base_dx_u + c] -= dot * x_state[base_u + c];
        }
      }

      // Residual update + Group norm
      for (let u = 0; u < NUM_UNITS; u++) {
        const base_u = base_x + u * 32;
        const base_dx_u = u * 32;

        let sum_sq = 0.0;
        for (let c = 0; c < 32; c++) {
          const updated = x_state[base_u + c] + dx_buf[base_dx_u + c];
          x_state[base_u + c] = updated;
          sum_sq += updated * updated;
        }
        const norm = Math.sqrt(sum_sq) + 1e-6;
        for (let c = 0; c < 32; c++) {
          x_state[base_u + c] /= norm;
        }
      }
    }

    predictTrial(t);
  }

  function step() {
    for (let t = 0; t < params.num_samples; t++) {
      stepTrial(t);
    }
    step_count++;
    updateStepCounter();
    updateHighlights();
    renderAllTrials();
  }

  function getGridPos(canvasEl, e) {
    const rect = canvasEl.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const c = Math.min(canvasEl.clientWidth, canvasEl.clientHeight);
    const offsetX = (canvasEl.clientWidth - c) / 2.0;
    const offsetY = (canvasEl.clientHeight - c) / 2.0;
    const normX = (mouseX - offsetX) / c;
    const normY = (mouseY - offsetY) / c;

    if (normX < 0.0 || normX >= 1.0 || normY < 0.0 || normY >= 1.0) {
      return [-1, -1];
    }
    const gx = Math.floor(normX * 9);
    const gy = Math.floor(normY * 9);
    return [gx, gy];
  }

  function applyBrushToTrial(t, gx, gy) {
    if (gx < 0 || gx >= 9 || gy < 0 || gy >= 9) return;
    const trial = trials[t];
    const x_state = trial.x_state;
    const r = params.noiseRadius;
    const std = params.noiseStd;

    if (std <= 0) return;

    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const rx = gx + dx;
        const ry = gy + dy;
        if (rx < 0 || rx >= 9 || ry < 0 || ry >= 9) continue;

        const cell_idx = ry * 9 + rx;
        const base_x = cell_idx * CHN;
        const is_clue = input_digits[cell_idx] > 0;

        // Noise mode: inject Gaussian noise scaled by noiseStd
        for (let c = 0; c < CHN; c++) {
          if (is_clue && c < INPUT_EMB_D) continue;
          x_state[base_x + c] += randGaussian(0, std);
        }

        // Group norm across 4 units
        for (let u = 0; u < NUM_UNITS; u++) {
          let sum_sq = 0.0;
          const base_u = base_x + u * 32;
          for (let c = 0; c < 32; c++) {
            sum_sq += x_state[base_u + c] * x_state[base_u + c];
          }
          const norm = Math.sqrt(sum_sq) + 1e-6;
          for (let c = 0; c < 32; c++) {
            x_state[base_u + c] /= norm;
          }
        }
      }
    }

    trials[t].correct_streak = 0;
    trials[t].is_correct = false;
    predictTrial(t);
    updateHighlights();
    renderTrial(t);
  }

  function updateNumSamplesVisibility() {
    for (let t = 0; t < MAX_SAMPLES; t++) {
      const card = trials[t].cardEl;
      if (!card) continue;
      if (t < params.num_samples) {
        card.style.display = 'flex';
      } else {
        card.style.display = 'none';
      }
    }
  }

  function setupDOM() {
    const trialsGrid = $('#trials-grid');
    if (!trialsGrid) return;
    trialsGrid.innerHTML = '';

    for (let t = 0; t < MAX_SAMPLES; t++) {
      const card = document.createElement('div');
      card.className = 'trial-card';
      card.id = `trial-card-${t}`;
      card.setAttribute('data-trial', t);

      const header = document.createElement('div');
      header.className = 'trial-header';

      const title = document.createElement('span');
      title.className = 'trial-title';
      title.innerHTML = `Trial ${t + 1}: confidence <span class="conf-scalar" id="conf-${t}">0.00</span>`;

      const badge = document.createElement('span');
      badge.className = 'trial-badge';
      badge.textContent = '✓';

      header.appendChild(title);
      header.appendChild(badge);

      const canvasEl = document.createElement('canvas');
      canvasEl.className = 'trial-canvas';
      canvasEl.width = 256;
      canvasEl.height = 256;

      card.appendChild(header);
      card.appendChild(canvasEl);
      trialsGrid.appendChild(card);

      const ctx = canvasEl.getContext('2d');
      trials[t].canvas = canvasEl;
      trials[t].ctx = ctx;
      trials[t].cardEl = card;
      trials[t].titleEl = title;
      trials[t].confEl = title.querySelector('.conf-scalar');
      trials[t].badgeEl = badge;

      // Pointer event for interactive brush
      canvasEl.addEventListener('pointerdown', (e) => {
        canvasEl.setPointerCapture(e.pointerId);
        const [gx, gy] = getGridPos(canvasEl, e);
        if (gx >= 0) applyBrushToTrial(t, gx, gy);
      });

      canvasEl.addEventListener('pointermove', (e) => {
        if (e.buttons === 1) {
          const [gx, gy] = getGridPos(canvasEl, e);
          if (gx >= 0) applyBrushToTrial(t, gx, gy);
        }
      });
    }

    updateNumSamplesVisibility();
  }

  function initUI() {
    // 1) First row controls
    const diffSelect = $('#difficulty-select');
    if (diffSelect) {
      diffSelect.innerHTML = '';
      const diffKeys = Object.keys(dsets).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
      for (const k of diffKeys) {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k;
        if (k === params.difficulty) opt.selected = true;
        diffSelect.appendChild(opt);
      }
      diffSelect.addEventListener('change', (e) => {
        params.difficulty = e.target.value;
        newSudoku();
      });
    }

    const samplesSelect = $('#samples-select');
    if (samplesSelect) {
      samplesSelect.value = params.num_samples.toString();
      samplesSelect.addEventListener('change', (e) => {
        params.num_samples = parseInt(e.target.value, 10);
        updateNumSamplesVisibility();
        reset();
      });
    }

    // 3) Demo controls row (i): play-pause, reset, step, speed, new-sudoku
    const playPauseBtn = $('#play-pause');
    if (playPauseBtn) {
      playPauseBtn.addEventListener('click', () => {
        params.runModel = !params.runModel;
        updatePlayPauseUI();
      });
    }

    const resetBtn = $('#reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        reset();
      });
    }

    const stepBtn = $('#step');
    if (stepBtn) {
      stepBtn.addEventListener('click', () => {
        if (params.runModel) {
          params.runModel = false;
          updatePlayPauseUI();
        }
        step();
      });
    }

    const speedSlider = $('#speed');
    const speedLabel = $('#speedLabel');
    if (speedSlider && speedLabel) {
      speedSlider.addEventListener('input', (e) => {
        params.stepFps = parseInt(e.target.value, 10);
        speedLabel.textContent = `${params.stepFps} steps/s`;
      });
    }
    const resetSpeedBtn = $('#reset-speed');
    if (resetSpeedBtn && speedSlider && speedLabel) {
      resetSpeedBtn.addEventListener('click', () => {
        params.stepFps = 15;
        speedSlider.value = 15;
        speedLabel.textContent = '15 steps/s';
      });
    }

    const newSudokuBtn = $('#new-sudoku');
    if (newSudokuBtn) {
      newSudokuBtn.addEventListener('click', () => {
        newSudoku();
      });
    }

    // 4) Noise controls row (ii): radius slider, noise std slider
    const noiseRadiusSlider = $('#noise-radius');
    const noiseRadiusLabel = $('#noiseRadiusLabel');
    const radiusNames = ['1x1', '3x3', '5x5', '7x7', '9x9'];
    function updateNoiseRadiusLabel(val) {
      if (!noiseRadiusLabel) return;
      noiseRadiusLabel.textContent = radiusNames[val] || `${2 * val + 1}x${2 * val + 1}`;
    }

    if (noiseRadiusSlider && noiseRadiusLabel) {
      noiseRadiusSlider.addEventListener('input', (e) => {
        params.noiseRadius = parseInt(e.target.value, 10);
        updateNoiseRadiusLabel(params.noiseRadius);
      });
    }
    const resetNoiseRadiusBtn = $('#reset-noise-radius');
    if (resetNoiseRadiusBtn && noiseRadiusSlider && noiseRadiusLabel) {
      resetNoiseRadiusBtn.addEventListener('click', () => {
        params.noiseRadius = 1;
        noiseRadiusSlider.value = 1;
        updateNoiseRadiusLabel(1);
      });
    }

    const noiseStdSlider = $('#noise-std');
    const noiseStdLabel = $('#noiseStdLabel');
    if (noiseStdSlider && noiseStdLabel) {
      noiseStdSlider.addEventListener('input', (e) => {
        params.noiseStd = parseFloat(e.target.value);
        noiseStdLabel.textContent = params.noiseStd.toFixed(2);
      });
    }
    const resetNoiseStdBtn = $('#reset-noise-std');
    if (resetNoiseStdBtn && noiseStdSlider && noiseStdLabel) {
      resetNoiseStdBtn.addEventListener('click', () => {
        params.noiseStd = 0.3;
        noiseStdSlider.value = 0.3;
        noiseStdLabel.textContent = '0.30';
      });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName))
        return;
      if (e.key === ' ') {
        e.preventDefault();
        params.runModel = !params.runModel;
        updatePlayPauseUI();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        currentBestIdx = 0;
        reset();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        if (params.runModel) {
          params.runModel = false;
          updatePlayPauseUI();
        }
        step();
      } else if (e.key === 'n' || e.key === 'N') {
        e.preventDefault();
        newSudoku();
      }
    });
  }

  function frame(time_ms) {
    const time = (time_ms || performance.now()) / 1000.0;
    if (params.runModel) {
      const stepInterval = 1.0 / params.stepFps;
      if (time - lastStepTime >= stepInterval) {
        step();
        lastStepTime = time;
      }
    }
    requestAnimationFrame(frame);
  }

  async function init() {
    try {
      // 1. Create digit textures
      digits = createDigitTextures();

      // 2. Load Reasoning NCA model
      const model_res = await fetch(model_path);
      const model_json = await model_res.json();

      model.input_embed = decodeBase64(model_json.input_embed.data, 'float32');
      model.fixed_attention = decodeBase64(model_json.sensing.fixed_attention.data, 'float32');
      model.v = decodeBase64(model_json.sensing.v.data, 'float32');
      model.w1 = decodeBase64(model_json.update.w1.data, 'float32');
      model.b1 = decodeBase64(model_json.update.b1.data, 'float32');
      model.w2 = decodeBase64(model_json.update.w2.data, 'float32');
      model.b2 = decodeBase64(model_json.update.b2.data, 'float32');

      // Precalculate input_embed norms
      for (let t = 0; t < 10; t++) {
        let sum_sq = 0.0;
        for (let c = 0; c < INPUT_EMB_D; c++) {
          const val = model.input_embed[t * INPUT_EMB_D + c];
          sum_sq += val * val;
        }
        ie_norms[t] = Math.sqrt(sum_sq) + 1e-6;
      }

      // 3. Load Sudoku-OOD dataset
      const dset_res = await fetch(dset_path);
      const dset_json = await dset_res.json();

      for (const diff_key in dset_json) {
        dsets[diff_key] = {
          input_masks: {
            data: decodeBase64(dset_json[diff_key].input_masks.data, 'bool'),
            shape: dset_json[diff_key].input_masks.shape,
          },
          target_grids: {
            data: decodeBase64(dset_json[diff_key].target_grids.data, 'uint8'),
            shape: dset_json[diff_key].target_grids.shape,
          },
        };
      }

      // 4. Setup grid DOM and UI controls
      setupDOM();
      initUI();

      // 5. Initialize first puzzle and launch animation
      newSudoku();
      requestAnimationFrame(frame);
    } catch (err) {
      console.error('Failed to initialize Sudoku-OOD demo:', err);
    }
  }

  init();
}

export const createDemo = createDemoSood;
