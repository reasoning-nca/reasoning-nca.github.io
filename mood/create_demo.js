// Reasoning NCA: Maze Out-Of-Distribution (MOOD) & Adaptive Compute (MOOD-AC) Demo
// Unified implementation supporting:
// 1. Dual rollouts: 9x9 (train) vs test size (13x13 to 201x201) (mood.html)
// 2. Adaptive Compute on 201x201 Maze-XL with active cell overlays and Damage/Noise/Move brushes (mood_ac.html)

export function createDemoMoodAC(glslOrDivId, maybeDivId) {
  return createDemoMood(glslOrDivId, maybeDivId, { isAdaptiveCompute: true });
}

export function createDemoMood(glslOrDivId, maybeDivId, options = {}) {
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
  const $$ = (selector) => root.querySelectorAll(selector);

  // Detect mode: single-grid Adaptive Compute (mood_ac.html) or dual-grid comparison (mood.html)
  const isACMode = options.isAdaptiveCompute || !!$('#demo-canvas') || !!$('#ac-toggle-btn') || !!$('#ac-group');

  // Setup offscreen WebGL canvas for SwissGL operations
  const renderCanvas = document.createElement('canvas');
  renderCanvas.width = 609;
  renderCanvas.height = 609;
  const gl = renderCanvas.getContext('webgl2', {
    alpha: false,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  const glsl = passedGlsl || SwissGL(gl || renderCanvas);
  const sourceCanvas = (glsl && glsl.gl && glsl.gl.canvas) || renderCanvas;

  const CHN = 16;
  const C4 = 4; // 16 / 4 = 4 RGBA layers

  const dset_path = '../assets/json/mooddata.json';
  const model_path = '../assets/json/moodnca.json';

  let dsets = {};
  let model = null;

  const defaultRadius = isACMode ? 6 : 1; // 6: 13x13, 1: 3x3

  const params = {
    testSize: isACMode ? '201' : '59',
    runModel: false,
    stepFps: 60,
    fireRate: 0.8,
    adaptiveCompute: isACMode, // Active by default in AC demo
    brush_mode: 'damage',       // 'damage', 'noise', 'move'
    noiseRadius: defaultRadius, // default 6 (13x13) in AC mode, 1 (3x3) in dual mode
    noiseStd: 0.3,             // 0 to 1
  };

  let lastTime = 0;
  let prevRunMode = false;
  let dragging = null; // { gridObj, token, x, y } in move mode
  let activePointer = null;

  function makeGridObj(key, label) {
    return {
      key,
      label,
      W: 0,
      H: 0,
      step_count: 0,
      input_grid: null,
      nca_grid: null,
      tracker_tex: null, // r32f texture for EWMA unconfident frequency (AC mode)
      input_state: null,
      target_grid: null,
      is_correct: false,
      correct_streak: 0,
      canvas: null,
      ctx: null,
      cardEl: null,
      stepEl: null,
    };
  }

  const grids = {};
  if (isACMode) {
    grids.main = makeGridObj('201', '201x201');
  } else {
    grids.train = makeGridObj('9', '9x9');
    grids.test = makeGridObj('59', '59x59');
  }

  function decodeBase64Model(obj) {
    for (const k in obj) {
      if (obj[k] && obj[k].data64) {
        const byteChars = atob(obj[k].data64);
        const byteNumbers = new Uint8Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
          byteNumbers[i] = byteChars.charCodeAt(i);
        }
        obj[k].data = new Float32Array(byteNumbers.buffer);
        delete obj[k].data64;
      } else if (typeof obj[k] === 'object' && obj[k] !== null) {
        decodeBase64Model(obj[k]);
      }
    }
  }

  function load_model(raw_model) {
    const w2 = raw_model.update.w2.data;
    const w2t_data = new Float32Array(128 * 16);
    for (let c = 0; c < 16; c++) {
      for (let h = 0; h < 128; h++) {
        w2t_data[h * 16 + c] = w2[c * 128 + h];
      }
    }

    return {
      K: raw_model.sensing.kernels.data, // 36 weights (4 heads x 9 neighbors)
      input_embed: glsl(
        {},
        {
          size: [4, 4],
          format: 'rgba32f',
          data: raw_model.input_embed.data,
          tag: 'input_embed',
        }
      ),
      w1: glsl(
        {},
        {
          size: [16, 128],
          format: 'rgba32f',
          data: raw_model.update.w1.data,
          tag: 'w1',
        }
      ),
      b1: glsl(
        {},
        {
          size: [1, 128],
          format: 'r32f',
          data: raw_model.update.b1.data,
          tag: 'b1',
        }
      ),
      w2t: glsl(
        {},
        {
          size: [4, 128],
          format: 'rgba32f',
          data: w2t_data,
          tag: 'w2t',
        }
      ),
      b2: glsl(
        {},
        {
          size: [4, 1],
          format: 'rgba32f',
          data: raw_model.update.b2.data,
          tag: 'b2',
        }
      ),
    };
  }

  function initGridMaze(gridObj, dsetKey) {
    if (!dsets[dsetKey]) return;
    const dset = dsets[dsetKey].target_grids;
    gridObj.H = dset.shape[1];
    gridObj.W = dset.shape[2];
    gridObj.key = dsetKey;
    gridObj.label = `${dsetKey}x${dsetKey}`;

    const num_samples = dset.shape[0];
    const maze_idx = Math.floor(Math.random() * num_samples);
    const maze = dset.data.slice(maze_idx * gridObj.H * gridObj.W, (maze_idx + 1) * gridObj.H * gridObj.W);

    gridObj.input_state = new Float32Array(gridObj.H * gridObj.W);
    gridObj.target_grid = new Uint8Array(gridObj.H * gridObj.W);

    for (let i = 0; i < gridObj.H * gridObj.W; i++) {
      gridObj.target_grid[i] = maze[i];
      if (maze[i] === 2) {
        gridObj.input_state[i] = 2.0; // wall
      } else if (maze[i] === 3) {
        gridObj.input_state[i] = 3.0; // source / target
      } else {
        gridObj.input_state[i] = 0.0; // empty corridor
      }
    }

    gridObj.input_grid = glsl(
      {},
      {
        size: [gridObj.W, gridObj.H],
        format: 'r32f',
        data: gridObj.input_state,
        tag: `input_grid_${gridObj.key}`,
        story: 2,
      }
    );

    resetGridState(gridObj);
  }

  function resetGridState(gridObj) {
    if (!model || !gridObj.input_grid) return;
    gridObj.step_count = 0;
    gridObj.correct_streak = 0;
    gridObj.is_correct = false;
    if (gridObj.cardEl) gridObj.cardEl.classList.remove('correct');

    gridObj.nca_grid = glsl(
      {
        ...model,
        seed: Math.random() * 26321,
        input_grid: gridObj.input_grid[0],
        FP: `
        vec2 box_muller(float u1, float u2) {
            const float PI = 3.1415926535;
            float R = sqrt(-2.0 * log(u1));
            float theta = 2.0 * PI * u2;
            return vec2(R * cos(theta), R * sin(theta));
        }

        vec4 rand_normal(vec4 mean, float std, vec2 U1, vec2 U2) {
            float ua = U1.x + 1e-6, ub = U1.y;
            float uc = U2.x + 1e-6, ud = U2.y;
            vec2 zab = box_muller(ua, ub);
            vec2 zcd = box_muller(uc, ud);
            return mean + std * vec4(zab.r, zab.g, zcd.r, zcd.g);
        }

        void fragment() {
            int iseed = int(seed);
            int token = int(input_grid(I).r);

            if (token == 0) {
                // Non-input cell: initialize all 16 channels with Gaussian noise N(0, 0.15^2)
                FOut  = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed)).xy,      hash(ivec3(I, iseed + 1)).xy);
                FOut1 = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed + 2)).xy,  hash(ivec3(I, iseed + 3)).xy);
                FOut2 = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed + 4)).xy,  hash(ivec3(I, iseed + 5)).xy);
                FOut3 = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed + 6)).xy,  hash(ivec3(I, iseed + 7)).xy);
            } else {
                // Input cells (wall = token 2, source/target = token 3):
                // Set all 16 channels to input_embed[token]
                FOut  = input_embed(ivec2(0, token));
                FOut1 = input_embed(ivec2(1, token));
                FOut2 = input_embed(ivec2(2, token));
                FOut3 = input_embed(ivec2(3, token));
            }
        }
        `,
      },
      {
        size: [gridObj.W, gridObj.H],
        layern: C4,
        format: 'rgba32f',
        story: 2,
        tag: `nca_grid_${gridObj.key}`,
      }
    );

    // Initialize unconfident tracker history texture in AC mode to 0.0
    if (isACMode) {
      gridObj.tracker_tex = glsl(
        {
          FP: `
          void fragment() {
            FOut.r = 0.0;
          }
          `,
        },
        {
          size: [gridObj.W, gridObj.H],
          format: 'r32f',
          story: 2,
          tag: `tracker_tex_${gridObj.key}`,
        }
      );
      glsl(
        {
          FP: `
          void fragment() {
            FOut.r = 0.0;
          }
          `,
        },
        gridObj.tracker_tex
      );
    }

    checkGridCorrectness(gridObj);
  }

  function reset() {
    for (const g of Object.values(grids)) {
      resetGridState(g);
    }
    updateStepCountUI();
    renderAllGrids();
  }

  function newMaze() {
    if (isACMode) {
      initGridMaze(grids.main, '201');
    } else {
      initGridMaze(grids.train, '9');
      initGridMaze(grids.test, params.testSize);
    }
    updateStepCountUI();
    renderAllGrids();
  }

  function changeTestSize(newSize) {
    if (!dsets[newSize] || isACMode) return;
    params.testSize = newSize;
    initGridMaze(grids.test, params.testSize);
    updateStepCountUI();
    renderAllGrids();
  }

  function stepGrid(gridObj) {
    if (!model || !gridObj.input_grid || !gridObj.nca_grid) return;
    const nca = model;
    const W_grid = gridObj.W;
    const H_grid = gridObj.H;
    const useAC = params.adaptiveCompute && !!gridObj.tracker_tex;

    // 1. Advance NCA simulation
    glsl(
      {
        ...nca,
        input_grid: gridObj.input_grid[0],
        seed: Math.random() * 26321,
        fire_rate: params.fireRate,
        fire_rate_low: 0.4,
        halt_conf_thresh: 0.95,
        adaptive_compute: useAC ? 1.0 : 0.0,
        dt: 1.0,
        W_grid: W_grid,
        H_grid: H_grid,
        FP: `
        const int C4 = 4;
        uniform float K[36];

        vec4 perc[16];

        void fragment() {
            // If input cell (wall or source/target), do not update
            if (input_grid(I).r > 0.5) {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
                return;
            }

            // Stochastic update & adaptive compute check
            float effective_fire_rate = fire_rate;
            if (adaptive_compute == 1.0) {
                vec2 out_emb = Src(I, 0).rg;
                float d0 = length(out_emb - input_embed(ivec2(0, 0)).rg);
                float d1 = length(out_emb - input_embed(ivec2(0, 1)).rg);
                float d2 = length(out_emb - input_embed(ivec2(0, 2)).rg);
                float d3 = length(out_emb - input_embed(ivec2(0, 3)).rg);
                float conf = max(max(1.0 / (1.0 + d0), 1.0 / (1.0 + d1)),
                                 max(1.0 / (1.0 + d2), 1.0 / (1.0 + d3)));
                if (conf > halt_conf_thresh) {
                    effective_fire_rate = fire_rate_low;
                }
            }

            if (hash(ivec3(I, int(seed))).x > effective_fire_rate) {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
                return;
            }

            // Perception: 4 heads x 9 neighbors convolution
            for (int i = 0; i < 16; ++i) {
                perc[i] = vec4(0.0);
            }

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int n = (dy + 1) * 3 + (dx + 1);
                    ivec2 p = (I + ivec2(dx, dy) + ivec2(W_grid, H_grid)) % ivec2(W_grid, H_grid);
                    for (int l = 0; l < 4; ++l) {
                        vec4 v = Src(p, l);
                        for (int m = 0; m < 4; ++m) {
                            perc[m * 4 + l] += K[m * 9 + n] * v;
                        }
                    }
                }
            }

            // MLP Update:
            vec4 dx_out[4];
            for (int l = 0; l < 4; ++l) {
                dx_out[l] = b2(ivec2(l, 0));
            }

            const int ci = 16;  // 64 / 4
            const int ch = 128; // hidden units

            for (int h = 0; h < ch; ++h) {
                float y = b1(ivec2(0, h)).x;
                for (int i = 0; i < ci; ++i) {
                    y += dot(perc[i], w1(ivec2(i, h)));
                }
                if (y <= 0.0) continue; // ReLU
                for (int l = 0; l < 4; ++l) {
                    dx_out[l] += y * w2t(ivec2(l, h));
                }
            }

            // Residual update: x = x + dt * dx
            FOut  = Src(I, 0) + dx_out[0] * dt;
            FOut1 = Src(I, 1) + dx_out[1] * dt;
            FOut2 = Src(I, 2) + dx_out[2] * dt;
            FOut3 = Src(I, 3) + dx_out[3] * dt;
        }
        `,
      },
      gridObj.nca_grid
    );

    // 2. If AC mode and tracker_tex exists, update EWMA unconfident tracker history texture
    if (useAC) {
      glsl(
        {
          ...model,
          state: gridObj.nca_grid[0],
          input_grid: gridObj.input_grid[0],
          halt_conf_thresh: 0.95,
          alpha: 0.001,
          FP: `
          void fragment() {
            if (input_grid(I).r > 0.5) {
              FOut.r = 0.0;
              return;
            }
            vec2 out_emb = state(I, 0).rg;
            float d0 = length(out_emb - input_embed(ivec2(0, 0)).rg);
            float d1 = length(out_emb - input_embed(ivec2(0, 1)).rg);
            float d2 = length(out_emb - input_embed(ivec2(0, 2)).rg);
            float d3 = length(out_emb - input_embed(ivec2(0, 3)).rg);
            float conf = max(max(1.0 / (1.0 + d0), 1.0 / (1.0 + d1)),
                             max(1.0 / (1.0 + d2), 1.0 / (1.0 + d3)));

            float is_unconf = (conf <= halt_conf_thresh) ? 1.0 : 0.0;
            float prev_tracker = Src(I).r;
            FOut.r = mix(prev_tracker, is_unconf, alpha);
          }
          `,
        },
        gridObj.tracker_tex
      );
    }
  }

  function step() {
    const list = Object.values(grids);
    let all_correct = true;
    for (const g of list) {
      if (!g.is_correct) {
        stepGrid(g);
        g.step_count++;
        checkGridCorrectness(g);
      }
      if (!g.is_correct) {
        all_correct = false;
      }
    }

    updateStepCountUI();
    renderAllGrids();

    if (all_correct) {
      params.runModel = false;
      updatePlayPauseUI();
    }
  }

  function checkGridCorrectness(gridObj) {
    if (!gridObj.target_grid || !gridObj.input_grid || !gridObj.nca_grid) return;
    const W = gridObj.W;
    const H = gridObj.H;

    glsl(
      {
        ...model,
        state: gridObj.nca_grid[0],
        input_grid: gridObj.input_grid[0],
        size: [W, H],
        FP: `
        void fragment() {
            int token = int(input_grid(I).r);
            if (token != 0) {
                FOut = vec4(0.0);
                return;
            }
            vec2 out_emb = state(I, 0).rg;
            float d0 = length(out_emb - input_embed(ivec2(0, 0)).rg);
            float d1 = length(out_emb - input_embed(ivec2(0, 1)).rg);
            float d2 = length(out_emb - input_embed(ivec2(0, 2)).rg);
            float d3 = length(out_emb - input_embed(ivec2(0, 3)).rg);

            // argmax(logits) is argmin(dist). Token 1 is path.
            bool pred_is_path = (d1 < d0 && d1 < d2 && d1 < d3);
            FOut = vec4(pred_is_path ? 1.0 : 0.0, 0.0, 0.0, 1.0);
        }
        `,
      },
      { size: [W, H], tag: `correct_mask_${gridObj.key}`, format: 'rgba8' }
    );

    const mask_buf = glsl.buffers[`correct_mask_${gridObj.key}`].readSync();
    let is_match = true;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        const target_val = gridObj.target_grid[idx];
        const is_path_target = (target_val === 1);
        const pred_path = (mask_buf[idx * 4] > 128);

        if (gridObj.input_state[idx] === 0.0) {
          if (is_path_target !== pred_path) {
            is_match = false;
            break;
          }
        }
      }
      if (!is_match) break;
    }

    if (is_match) {
      gridObj.correct_streak = (gridObj.correct_streak || 0) + 1;
    } else {
      gridObj.correct_streak = 0;
    }

    gridObj.is_correct = (gridObj.correct_streak >= 10);
    if (gridObj.cardEl) {
      if (gridObj.is_correct) {
        gridObj.cardEl.classList.add('correct');
      } else {
        gridObj.cardEl.classList.remove('correct');
      }
    }
  }

  function moveCell(gridObj, fromX, fromY, toX, toY, token) {
    const W = gridObj.W;
    const H = gridObj.H;
    if (fromX < 0 || fromX >= W || fromY < 0 || fromY >= H) return false;
    if (toX < 0 || toX >= W || toY < 0 || toY >= H) return false;
    if (fromX === toX && fromY === toY) return false;

    // Disallow moving onto border boundary or existing non-empty cell
    if (toX === 0 || toX === W - 1 || toY === 0 || toY === H - 1) return false;
    if (gridObj.input_state && gridObj.input_state[toY * W + toX] !== 0.0) return false;

    if (gridObj.input_state) {
      gridObj.input_state[fromY * W + fromX] = 0.0;
      gridObj.input_state[toY * W + toX] = token;
    }

    // 1. Update input_grid
    glsl(
      {
        fromX,
        fromY,
        toX,
        toY,
        token,
        FP: `
        void fragment() {
          if (I == ivec2(fromX, fromY)) {
            FOut.r = 0.0;
          } else if (I == ivec2(toX, toY)) {
            FOut.r = token;
          } else {
            FOut.r = Src(I).r;
          }
        }
        `,
      },
      gridObj.input_grid
    );

    // 2. Update nca_grid: previous cell replaced with Gaussian noise, new cell with token embedding
    glsl(
      {
        ...model,
        input_grid: gridObj.input_grid[0],
        fromX,
        fromY,
        toX,
        toY,
        token,
        seed: Math.random() * 26321,
        FP: `
        vec2 box_muller(float u1, float u2) {
            const float PI = 3.1415926535;
            float R = sqrt(-2.0 * log(u1));
            float theta = 2.0 * PI * u2;
            return vec2(R * cos(theta), R * sin(theta));
        }

        vec4 rand_normal(vec4 mean, float std, vec2 U1, vec2 U2) {
            float ua = U1.x + 1e-6, ub = U1.y;
            float uc = U2.x + 1e-6, ud = U2.y;
            vec2 zab = box_muller(ua, ub);
            vec2 zcd = box_muller(uc, ud);
            return mean + std * vec4(zab.r, zab.g, zcd.r, zcd.g);
        }

        void fragment() {
            if (I == ivec2(fromX, fromY)) {
                int iseed = int(seed);
                FOut  = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed)).xy,      hash(ivec3(I, iseed + 1)).xy);
                FOut1 = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed + 2)).xy,  hash(ivec3(I, iseed + 3)).xy);
                FOut2 = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed + 4)).xy,  hash(ivec3(I, iseed + 5)).xy);
                FOut3 = rand_normal(vec4(0.0), 0.15, hash(ivec3(I, iseed + 6)).xy,  hash(ivec3(I, iseed + 7)).xy);
            } else if (I == ivec2(toX, toY)) {
                int t = int(token);
                FOut  = input_embed(ivec2(0, t));
                FOut1 = input_embed(ivec2(1, t));
                FOut2 = input_embed(ivec2(2, t));
                FOut3 = input_embed(ivec2(3, t));
            } else {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
            }
        }
        `,
      },
      gridObj.nca_grid
    );

    if (gridObj.tracker_tex) {
      glsl(
        {
          fromX,
          fromY,
          toX,
          toY,
          FP: `
          void fragment() {
            if (I == ivec2(fromX, fromY) || I == ivec2(toX, toY)) {
              FOut.r = 0.0;
            } else {
              FOut.r = Src(I).r;
            }
          }
          `,
        },
        gridObj.tracker_tex
      );
    }

    gridObj.correct_streak = 0;
    checkGridCorrectness(gridObj);
    return true;
  }

  function injectNoise(gridObj, gx, gy) {
    const W = gridObj.W;
    const H = gridObj.H;
    if (gx < 0 || gx >= W || gy < 0 || gy >= H) return;
    const r = params.noiseRadius;
    const std = params.noiseStd;
    if (std <= 0) return;

    glsl(
      {
        clickX: gx,
        clickY: gy,
        noise_radius: r,
        noise_std: std,
        seed: Math.random() * 26321,
        input_grid: gridObj.input_grid[0],
        FP: `
        vec2 box_muller(float u1, float u2) {
            const float PI = 3.1415926535;
            float R = sqrt(-2.0 * log(u1));
            float theta = 2.0 * PI * u2;
            return vec2(R * cos(theta), R * sin(theta));
        }

        vec4 rand_normal(vec4 mean, float std_val, vec2 U1, vec2 U2) {
            float ua = U1.x + 1e-6, ub = U1.y;
            float uc = U2.x + 1e-6, ud = U2.y;
            vec2 zab = box_muller(ua, ub);
            vec2 zcd = box_muller(uc, ud);
            return mean + std_val * vec4(zab.r, zab.g, zcd.r, zcd.g);
        }

        void fragment() {
            ivec2 diff = abs(I - ivec2(clickX, clickY));
            // Keep input cells fixed and only update within noise_radius
            if (input_grid(I).r > 0.5 || diff.x > int(noise_radius) || diff.y > int(noise_radius)) {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
                return;
            }

            int iseed = int(seed);
            FOut  = Src(I, 0) + rand_normal(vec4(0.0), noise_std, hash(ivec3(I, iseed)).xy,      hash(ivec3(I, iseed + 1)).xy);
            FOut1 = Src(I, 1) + rand_normal(vec4(0.0), noise_std, hash(ivec3(I, iseed + 2)).xy,  hash(ivec3(I, iseed + 3)).xy);
            FOut2 = Src(I, 2) + rand_normal(vec4(0.0), noise_std, hash(ivec3(I, iseed + 4)).xy,  hash(ivec3(I, iseed + 5)).xy);
            FOut3 = Src(I, 3) + rand_normal(vec4(0.0), noise_std, hash(ivec3(I, iseed + 6)).xy,  hash(ivec3(I, iseed + 7)).xy);
        }
        `,
      },
      gridObj.nca_grid
    );

    if (gridObj.tracker_tex) {
      glsl(
        {
          clickX: gx,
          clickY: gy,
          damage_radius: r,
          input_grid: gridObj.input_grid[0],
          FP: `
          void fragment() {
              ivec2 diff = abs(I - ivec2(clickX, clickY));
              if (input_grid(I).r > 0.5 || diff.x > int(damage_radius) || diff.y > int(damage_radius)) {
                  FOut.r = Src(I).r;
                  return;
              }
              FOut.r = 0.0;
          }
          `,
        },
        gridObj.tracker_tex
      );
    }

    gridObj.correct_streak = 0;
    checkGridCorrectness(gridObj);
  }

  function injectDamage(gridObj, gx, gy) {
    const W = gridObj.W;
    const H = gridObj.H;
    if (gx < 0 || gx >= W || gy < 0 || gy >= H) return;
    const r = params.noiseRadius;

    glsl(
      {
        clickX: gx,
        clickY: gy,
        damage_radius: r,
        input_grid: gridObj.input_grid[0],
        FP: `
        void fragment() {
            ivec2 diff = abs(I - ivec2(clickX, clickY));
            if (input_grid(I).r > 0.5 || diff.x > int(damage_radius) || diff.y > int(damage_radius)) {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
                return;
            }

            // Damage resets corridor cells to 0
            FOut  = vec4(0.0);
            FOut1 = vec4(0.0);
            FOut2 = vec4(0.0);
            FOut3 = vec4(0.0);
        }
        `,
      },
      gridObj.nca_grid
    );

    if (gridObj.tracker_tex) {
      glsl(
        {
          clickX: gx,
          clickY: gy,
          damage_radius: r,
          input_grid: gridObj.input_grid[0],
          FP: `
          void fragment() {
              ivec2 diff = abs(I - ivec2(clickX, clickY));
              if (input_grid(I).r > 0.5 || diff.x > int(damage_radius) || diff.y > int(damage_radius)) {
                  FOut.r = Src(I).r;
                  return;
              }
              FOut.r = 0.0;
          }
          `,
        },
        gridObj.tracker_tex
      );
    }

    gridObj.correct_streak = 0;
    checkGridCorrectness(gridObj);
  }

  function getCanvasGridPos(canvasEl, gridObj, e) {
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
    const gx = Math.floor(normX * gridObj.W);
    const gy = Math.floor(normY * gridObj.H);
    return [gx, gy];
  }

  function renderGridToCanvas(gridObj) {
    if (!gridObj.ctx || !gridObj.input_grid || !gridObj.nca_grid) return;
    const W = gridObj.W;
    const H = gridObj.H;
    const useAC = params.adaptiveCompute && !!gridObj.tracker_tex;

    glsl({
      ...model,
      state: gridObj.nca_grid[0],
      input_grid: gridObj.input_grid[0],
      tracker_tex: useAC ? gridObj.tracker_tex[0] : gridObj.input_grid[0],
      adaptive_compute: useAC ? 1.0 : 0.0,
      halt_conf_thresh: 0.95,
      Aspect: 'fit',
      size: [W, H],
      FP: `
        vec2 uv = vec2(UV.x, 1.0 - UV.y);
        int token = int(input_grid(uv).r);

        if (token == 2) {
            // Wall: black
            FOut.rgb = vec3(0.0);
            FOut.a = 1.0;
            return;
        }

        if (token == 3) {
            // Source / Target: magenta
            FOut.rgb = vec3(0.9, 0.0, 0.9);
            FOut.a = 1.0;
            return;
        }

        // Empty corridor: predict path logits via l2_similarity
        vec2 out_emb = state(uv, 0).rg;
        vec2 path_emb = vec2(0.0, 1.0);
        float dist = length(out_emb - path_emb);
        float logit = 1.0 / (1.0 + dist);
        float yellow = clamp(2.0 * logit - 1.0, 0.0, 1.0);

        vec3 rgb = vec3(0.5);
        rgb += vec3(0.5, 0.5, -0.5) * yellow;

        // Adaptive Compute Active cell overlays:
        if (adaptive_compute == 1.0) {
            float d0 = length(out_emb - input_embed(ivec2(0, 0)).rg);
            float d1 = length(out_emb - input_embed(ivec2(0, 1)).rg);
            float d2 = length(out_emb - input_embed(ivec2(0, 2)).rg);
            float d3 = length(out_emb - input_embed(ivec2(0, 3)).rg);
            float conf = max(max(1.0 / (1.0 + d0), 1.0 / (1.0 + d1)),
                             max(1.0 / (1.0 + d2), 1.0 / (1.0 + d3)));

            float unconf_freq = tracker_tex(uv).r;
            bool currently_unconfident = (conf <= halt_conf_thresh);
            bool often_active = currently_unconfident && (unconf_freq >= 0.2);
            bool currently_only_active = currently_unconfident && (unconf_freq < 0.2);

            // Only show currently active cells (unconfident with low historical frequency)
            if (currently_only_active) {
                rgb = vec3(0.867, 0.627, 0.867); // plum (#dda0dd)
            }
        }

        FOut.rgb = clamp(rgb, 0.0, 1.0);
        FOut.a = 1.0;
      `,
      Clear: 1.0,
    });

    const ctx = gridObj.ctx;
    ctx.clearRect(0, 0, gridObj.canvas.width, gridObj.canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(sourceCanvas, 0, 0, gridObj.canvas.width, gridObj.canvas.height);
  }

  function renderAllGrids() {
    for (const g of Object.values(grids)) {
      renderGridToCanvas(g);
    }
  }

  function setupCanvasEvents(gridObj) {
    const canvasEl = gridObj.canvas;
    if (!canvasEl) return;

    canvasEl.addEventListener('pointerdown', (e) => {
      canvasEl.setPointerCapture(e.pointerId);
      activePointer = e.pointerId;
      const [gx, gy] = getCanvasGridPos(canvasEl, gridObj, e);
      if (gx < 0) return;

      if (params.brush_mode === 'move') {
        const token = gridObj.input_state ? gridObj.input_state[gy * gridObj.W + gx] : 0.0;
        if (token === 3.0) {
          dragging = { gridObj, token, x: gx, y: gy };
          canvasEl.style.cursor = 'grabbing';
          prevRunMode = params.runModel;
          params.runModel = false;
          updatePlayPauseUI();
        }
      } else if (params.brush_mode === 'damage') {
        canvasEl.style.cursor = 'crosshair';
        prevRunMode = params.runModel;
        params.runModel = false;
        updatePlayPauseUI();
        injectDamage(gridObj, gx, gy);
        renderGridToCanvas(gridObj);
      } else {
        // Noise mode
        canvasEl.style.cursor = 'crosshair';
        prevRunMode = params.runModel;
        params.runModel = false;
        updatePlayPauseUI();
        injectNoise(gridObj, gx, gy);
        renderGridToCanvas(gridObj);
      }
    });

    canvasEl.addEventListener('pointermove', (e) => {
      const [gx, gy] = getCanvasGridPos(canvasEl, gridObj, e);
      if (e.buttons === 1) {
        if (gx < 0) return;
        if (params.brush_mode === 'move') {
          if (dragging && dragging.gridObj === gridObj) {
            if (gx !== dragging.x || gy !== dragging.y) {
              const moved = moveCell(gridObj, dragging.x, dragging.y, gx, gy, dragging.token);
              if (moved) {
                dragging.x = gx;
                dragging.y = gy;
                renderGridToCanvas(gridObj);
              }
            }
          }
        } else if (params.brush_mode === 'damage') {
          injectDamage(gridObj, gx, gy);
          renderGridToCanvas(gridObj);
        } else {
          injectNoise(gridObj, gx, gy);
          renderGridToCanvas(gridObj);
        }
      } else {
        if (params.brush_mode === 'move') {
          if (gx >= 0 && gridObj.input_state) {
            const token = gridObj.input_state[gy * gridObj.W + gx];
            canvasEl.style.cursor = (token === 3.0) ? 'grab' : 'default';
          } else {
            canvasEl.style.cursor = 'default';
          }
        } else {
          canvasEl.style.cursor = 'crosshair';
        }
      }
    });

    const finishPointer = (e) => {
      canvasEl.style.cursor = (params.brush_mode === 'move') ? 'grab' : 'crosshair';
      dragging = null;
      const any_not_correct = Object.values(grids).some((g) => !g.is_correct);
      params.runModel = prevRunMode && any_not_correct;
      updatePlayPauseUI();
      try {
        if (activePointer !== null) {
          canvasEl.releasePointerCapture(activePointer);
          activePointer = null;
        }
      } catch (err) { }
    };

    canvasEl.addEventListener('pointerup', finishPointer);
    canvasEl.addEventListener('pointercancel', finishPointer);
  }

  function updatePlayPauseUI() {
    const playIcon = $('#play');
    const pauseIcon = $('#pause');
    if (playIcon && pauseIcon) {
      if (params.runModel) {
        playIcon.style.display = 'none';
        pauseIcon.style.display = 'inline';
      } else {
        playIcon.style.display = 'inline';
        pauseIcon.style.display = 'none';
      }
    }
  }

  function updateStepCountUI() {
    if (isACMode) {
      if (grids.main.stepEl) {
        grids.main.stepEl.textContent = `Step: ${grids.main.step_count}`;
      }
    } else {
      if (grids.train && grids.train.stepEl) {
        grids.train.stepEl.textContent = `Step: ${grids.train.step_count}`;
      }
      if (grids.test && grids.test.stepEl) {
        grids.test.stepEl.textContent = `Step: ${grids.test.step_count}`;
      }
    }
  }

  function initUI() {
    if (isACMode) {
      grids.main.cardEl = $('#demo-card');
      grids.main.canvas = $('#demo-canvas');
      grids.main.stepEl = $('#step-count');
      if (grids.main.canvas) grids.main.ctx = grids.main.canvas.getContext('2d');
      setupCanvasEvents(grids.main);

      // Adaptive Compute toggle buttons (ON / OFF)
      const acRadios = $$('input[name="ac_toggle"]');
      if (acRadios.length > 0) {
        acRadios.forEach((radio) => {
          radio.addEventListener('change', (e) => {
            params.adaptiveCompute = (e.target.value === 'on');
            $$('.ac-mode-btn').forEach((btn) => btn.classList.remove('active'));
            const label = radio.closest('.ac-mode-btn');
            if (label) label.classList.add('active');
            renderAllGrids();
          });
        });
      }
      const acToggleBtn = $('#ac-toggle-btn');
      if (acToggleBtn) {
        acToggleBtn.addEventListener('click', () => {
          params.adaptiveCompute = !params.adaptiveCompute;
          acToggleBtn.classList.toggle('active', params.adaptiveCompute);
          renderAllGrids();
        });
      }
    } else {
      grids.train.cardEl = $('#train-card');
      grids.train.canvas = $('#train-canvas');
      grids.train.stepEl = $('#step-count-train');
      if (grids.train.canvas) grids.train.ctx = grids.train.canvas.getContext('2d');

      grids.test.cardEl = $('#test-card');
      grids.test.canvas = $('#test-canvas');
      grids.test.stepEl = $('#step-count-test');
      if (grids.test.canvas) grids.test.ctx = grids.test.canvas.getContext('2d');

      setupCanvasEvents(grids.train);
      setupCanvasEvents(grids.test);

      // Test size dropdown selector
      const testSizeSelect = $('#test-size-select');
      if (testSizeSelect) {
        testSizeSelect.innerHTML = '';
        const sizes = Object.keys(dsets)
          .map((k) => parseInt(k, 10))
          .filter((k) => k > 9)
          .sort((a, b) => a - b);

        for (const s of sizes) {
          const opt = document.createElement('option');
          opt.value = s.toString();
          opt.textContent = `${s}x${s}`;
          if (s.toString() === params.testSize) opt.selected = true;
          testSizeSelect.appendChild(opt);
        }

        testSizeSelect.addEventListener('change', (e) => {
          changeTestSize(e.target.value);
        });
      }
    }

    updateStepCountUI();

    // Play / Pause button
    const playPauseBtn = $('#play-pause');
    if (playPauseBtn) {
      playPauseBtn.addEventListener('click', () => {
        params.runModel = !params.runModel;
        updatePlayPauseUI();
      });
    }

    // Reset button
    const resetBtn = $('#reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        reset();
      });
    }

    // Step button
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

    // Speed slider
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
        params.stepFps = 60;
        speedSlider.value = 60;
        speedLabel.textContent = '60 steps/s';
      });
    }

    // Cell fire rate slider
    const fireRateSlider = $('#fire-rate');
    const fireRateLabel = $('#fireRateLabel');
    if (fireRateSlider && fireRateLabel) {
      fireRateSlider.addEventListener('input', (e) => {
        params.fireRate = parseFloat(e.target.value);
        fireRateLabel.textContent = params.fireRate.toFixed(2);
      });
    }
    const resetFireRateBtn = $('#reset-fire-rate');
    if (resetFireRateBtn && fireRateSlider && fireRateLabel) {
      resetFireRateBtn.addEventListener('click', () => {
        params.fireRate = 0.8;
        fireRateSlider.value = 0.8;
        fireRateLabel.textContent = '0.80';
      });
    }

    // New Maze button
    const newMazeBtn = $('#new-maze');
    if (newMazeBtn) {
      newMazeBtn.addEventListener('click', () => {
        newMaze();
      });
    }

    // Brush type radio buttons (Damage / Noise / Move)
    const brushRadios = $$('input[name="brush_type"]');
    const tipText = $('#tip-text');
    if (brushRadios.length > 0) {
      brushRadios.forEach((radio) => {
        radio.addEventListener('change', (e) => {
          params.brush_mode = e.target.value;
          $$('.brush-mode-btn').forEach((btn) => btn.classList.remove('active'));
          const label = radio.closest('.brush-mode-btn');
          if (label) label.classList.add('active');

          if (tipText) {
            if (params.brush_mode === 'move') {
              tipText.textContent = 'Drag and drop start or target points to relocate them.';
            } else if (params.brush_mode === 'damage') {
              tipText.textContent = 'Click or drag on the maze to damage (zero out) corridor cells.';
            } else {
              tipText.textContent = 'Click or drag on the maze to inject Gaussian noise live.';
            }
          }
        });
      });
    }

    // Noise/Damage radius slider
    const noiseRadiusSlider = $('#noise-radius');
    const noiseRadiusLabel = $('#noiseRadiusLabel');
    function updateNoiseRadiusLabel(val) {
      if (!noiseRadiusLabel) return;
      noiseRadiusLabel.textContent = `${2 * val + 1}x${2 * val + 1}`;
    }
    if (noiseRadiusSlider && noiseRadiusLabel) {
      params.noiseRadius = parseInt(noiseRadiusSlider.value, 10) || defaultRadius;
      updateNoiseRadiusLabel(params.noiseRadius);
      noiseRadiusSlider.addEventListener('input', (e) => {
        params.noiseRadius = parseInt(e.target.value, 10);
        updateNoiseRadiusLabel(params.noiseRadius);
      });
    }
    const resetNoiseRadiusBtn = $('#reset-noise-radius');
    if (resetNoiseRadiusBtn && noiseRadiusSlider && noiseRadiusLabel) {
      resetNoiseRadiusBtn.addEventListener('click', () => {
        params.noiseRadius = defaultRadius;
        noiseRadiusSlider.value = defaultRadius;
        updateNoiseRadiusLabel(defaultRadius);
      });
    }

    // Noise std slider
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
      if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === ' ') {
        e.preventDefault();
        params.runModel = !params.runModel;
        updatePlayPauseUI();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        reset();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        if (params.runModel) {
          params.runModel = false;
          updatePlayPauseUI();
        }
        step();
      }
    });
  }

  function loop(currentTime) {
    if (params.runModel) {
      const interval = 1000 / params.stepFps;
      if (currentTime - lastTime >= interval) {
        lastTime = currentTime;
        step();
      }
    }
    requestAnimationFrame(loop);
  }

  async function init() {
    try {
      const dsetRes = await fetch(dset_path);
      dsets = await dsetRes.json();
      for (const name in dsets) {
        const src = dsets[name].target_grids;
        const byteChars = atob(src.data64);
        const byteNumbers = new Uint8Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
          byteNumbers[i] = byteChars.charCodeAt(i);
        }
        src.data = byteNumbers;
        delete src.data64;
      }

      const modelRes = await fetch(model_path);
      const raw_model = await modelRes.json();
      decodeBase64Model(raw_model);
      model = load_model(raw_model);

      initUI();
      newMaze();
      requestAnimationFrame(loop);
    } catch (err) {
      console.error('Failed to initialize MOOD demo:', err);
    }
  }

  init();
}
