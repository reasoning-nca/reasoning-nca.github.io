// Visual Sudoku Demo implementation for Reasoning NCA
// Uses SwissGL for WebGL2-accelerated NCA simulation and live digit accuracy evaluation.

export function createDemoVsudoku(glsl, divId) {
  const root = document.getElementById(divId);
  if (!root) {
    console.error(`Root element #${divId} not found.`);
    return;
  }

  const $ = (selector) => root.querySelector(selector);
  const canvas = $('#demo-canvas');
  if (!canvas) {
    console.error('#demo-canvas not found inside root element.');
    return;
  }

  const dset_path = '../assets/json/vsudokudata.json';
  const model_path = '../assets/json/vsudokunca.json';

  let dsets = null;
  let w1_tex = null;
  let w2_tex = null;
  let templates_tex = null;
  let nca_grid = null;
  let z_grid = null;
  let h_grid0 = null;
  let h_grid1 = null;
  let pred_tex = null;

  let current_puzzle_img = null;
  let current_mask = null;
  let current_target_grid = null;

  let step_count = 0;
  let plot_history = [];
  let prevRunMode = false;
  let lastStepTime = 0;

  const params = {
    difficulty: 'easy',
    runModel: false,
    stepFps: 30,
    fireRate: 0.5,
    brushSize: 2.0,
    brush: 'white',
  };

  function decodeBase64(dataStr, dtype) {
    const binary = atob(dataStr);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    if (dtype === 'float32' || dtype === '<f4') {
      return new Float32Array(bytes.buffer);
    } else if (dtype === 'uint8' || dtype === '|u1' || dtype === 'bool') {
      return bytes;
    } else if (dtype === 'int32' || dtype === '<i4') {
      return new Int32Array(bytes.buffer);
    }
    return new Float32Array(bytes.buffer);
  }

  const common_inc = `
  ivec2 p_cell, p_left, p_right, p_up, p_down;
  ivec2 p00, p02, p20, p22;

  void init_neighbors(ivec2 I) {
    p_cell = I % 256;
    p_left  = ivec2((p_cell.x - 1 + 256) % 256, p_cell.y);
    p_right = ivec2((p_cell.x + 1) % 256, p_cell.y);
    p_up    = ivec2(p_cell.x, (p_cell.y - 1 + 256) % 256);
    p_down  = ivec2(p_cell.x, (p_cell.y + 1) % 256);

    p00 = ivec2((p_cell.x - 1 + 256) % 256, (p_cell.y - 1 + 256) % 256);
    p02 = ivec2((p_cell.x + 1) % 256, (p_cell.y - 1 + 256) % 256);
    p20 = ivec2((p_cell.x - 1 + 256) % 256, (p_cell.y + 1) % 256);
    p22 = ivec2((p_cell.x + 1) % 256, (p_cell.y + 1) % 256);
  }

  vec4 get_nca(ivec2 p, int l) {
    return state(ivec2((l % 8) * 256 + p.x, (l / 8) * 256 + p.y));
  }

  vec4 get_z(int i) {
    if (i < 64) {
      return get_nca(p_cell, i);
    } else if (i < 128) {
      int l = i - 64;
      vec4 v00 = get_nca(p00, l);
      vec4 v02 = get_nca(p02, l);
      vec4 v10 = get_nca(p_left, l);
      vec4 v12 = get_nca(p_right, l);
      vec4 v20 = get_nca(p20, l);
      vec4 v22 = get_nca(p22, l);
      return ((v02 - v00) + 2.0 * (v12 - v10) + (v22 - v20)) * 0.125;
    } else {
      int l = i - 128;
      vec4 v00 = get_nca(p00, l);
      vec4 v01 = get_nca(p_up, l);
      vec4 v02 = get_nca(p02, l);
      vec4 v20 = get_nca(p20, l);
      vec4 v21 = get_nca(p_down, l);
      vec4 v22 = get_nca(p22, l);
      return ((v20 - v00) + 2.0 * (v21 - v01) + (v22 - v02)) * 0.125;
    }
  }
  `;

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

  function renderPuzzleImage(mask, target) {
    if (!dsets || !mask || !target) return null;
    const puzzle = new Float32Array(252 * 252);
    for (let i = 0; i < puzzle.length; i++) {
      puzzle[i] = (Math.random() - 0.5) * 0.1;
    }

    const digit_imgs = dsets.digit_images;
    const num_digit_samples =
      digit_imgs && digit_imgs.shape ? digit_imgs.shape[1] : 20;

    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const idx = r * 9 + c;
        const is_clue = mask[idx] > 0;
        const d = is_clue ? target[idx] : 0;
        if (d > 0) {
          const img_idx = Math.floor(Math.random() * num_digit_samples);
          const base_img = (d * num_digit_samples + img_idx) * 28 * 28;
          for (let tr = 0; tr < 28; tr++) {
            const row_dst = (r * 28 + tr) * 252;
            const row_src = base_img + tr * 28;
            for (let tc = 0; tc < 28; tc++) {
              puzzle[row_dst + c * 28 + tc] = digit_imgs.array[row_src + tc];
            }
          }
        }
      }
    }
    return puzzle;
  }

  function newSudoku() {
    if (!dsets) return;
    const split_data = dsets[params.difficulty];
    if (!split_data) return;

    const in_data = split_data.input_masks;
    const num_samples = in_data.shape[0];
    const sample_idx = Math.floor(Math.random() * num_samples);
    current_mask = in_data.array.slice(sample_idx * 81, (sample_idx + 1) * 81);
    current_target_grid = split_data.target_grids.array.slice(
      sample_idx * 81,
      (sample_idx + 1) * 81
    );

    reset();
  }

  function reset() {
    if (!current_mask || !current_target_grid) return;

    // Resample fresh random MNIST digit variations for the same Sudoku puzzle grid
    current_puzzle_img = renderPuzzleImage(current_mask, current_target_grid);
    if (!current_puzzle_img) return;

    // Construct initial 2048x2048 state (64 tiles of 256x256)
    const stateData = new Float32Array(2048 * 2048 * 4);
    for (let i = 0; i < stateData.length; i++) {
      stateData[i] = (Math.random() - 0.5) * 0.2;
    }

    // Set channel 0 (red component of tile 0) to padded puzzle
    for (let r = 0; r < 252; r++) {
      for (let c = 0; c < 252; c++) {
        stateData[(r * 2048 + c) * 4] = current_puzzle_img[r * 252 + c];
      }
    }
    for (let r = 252; r < 256; r++) {
      for (let c = 0; c < 256; c++) {
        stateData[(r * 2048 + c) * 4] = 0.0;
      }
    }
    for (let r = 0; r < 252; r++) {
      for (let c = 252; c < 256; c++) {
        stateData[(r * 2048 + c) * 4] = 0.0;
      }
    }

    nca_grid = glsl(
      {},
      {
        size: [2048, 2048],
        format: 'rgba32f',
        data: stateData,
        story: 2,
        tag: 'nca_grid',
      }
    );

    step_count = 0;
    updateStepCounter();
    plot_history = [];
    updateAccuracyPlot();
  }

  function step() {
    if (!nca_grid || !w1_tex || !w2_tex) return;
    step_count++;
    updateStepCounter();

    // Pass 0: Precompute perception vector z [192 channels = 16x12 tiles of 256x256 -> 4096 x 3072]
    z_grid = glsl(
      {
        state: nca_grid[0],
        Inc: [common_inc],
        FP: `
        void fragment() {
          init_neighbors(I);
          int i = (I.y / 256) * 16 + (I.x / 256);
          FOut = get_z(i);
        }
        `,
      },
      { size: [4096, 3072], format: 'rgba16f', tag: 'z_grid' }
    );

    // Pass 1a: Compute hidden units 0..1023 (texture 4096x4096)
    h_grid0 = glsl(
      {
        z_grid: z_grid,
        w1: w1_tex,
        FP: `
        void fragment() {
          ivec2 p = I % 256;
          int u = (I.y / 256) * 16 + (I.x / 256);
          vec4 h = vec4(0.0);
          int base_u = 4 * u;
          for (int i = 0; i < 192; i++) {
            ivec2 z_pos = ivec2((i % 16) * 256 + p.x, (i / 16) * 256 + p.y);
            vec4 z = z_grid(z_pos);
            h.x += dot(z, w1(ivec2(i, base_u + 0)));
            h.y += dot(z, w1(ivec2(i, base_u + 1)));
            h.z += dot(z, w1(ivec2(i, base_u + 2)));
            h.w += dot(z, w1(ivec2(i, base_u + 3)));
          }
          FOut = max(h, vec4(0.0));
        }
        `,
      },
      { size: [4096, 4096], format: 'rgba16f', tag: 'h0' }
    );

    // Pass 1b: Compute hidden units 1024..2047 (texture 4096x4096)
    h_grid1 = glsl(
      {
        z_grid: z_grid,
        w1: w1_tex,
        FP: `
        void fragment() {
          ivec2 p = I % 256;
          int u = (I.y / 256) * 16 + (I.x / 256);
          vec4 h = vec4(0.0);
          int base_u = 1024 + 4 * u;
          for (int i = 0; i < 192; i++) {
            ivec2 z_pos = ivec2((i % 16) * 256 + p.x, (i / 16) * 256 + p.y);
            vec4 z = z_grid(z_pos);
            h.x += dot(z, w1(ivec2(i, base_u + 0)));
            h.y += dot(z, w1(ivec2(i, base_u + 1)));
            h.z += dot(z, w1(ivec2(i, base_u + 2)));
            h.w += dot(z, w1(ivec2(i, base_u + 3)));
          }
          FOut = max(h, vec4(0.0));
        }
        `,
      },
      { size: [4096, 4096], format: 'rgba16f', tag: 'h1' }
    );

    // Pass 2: Compute NCA state update dx = h @ w2 and stochastic update
    glsl(
      {
        h0: h_grid0,
        h1: h_grid1,
        w2t: w2_tex,
        seed: Math.random() * 26321.0,
        fire_rate: params.fireRate,
        FP: `
        void fragment() {
          ivec2 p = I % 256;
          if (hash(ivec3(p, int(seed))).x > fire_rate) {
            FOut = Src(I);
            return;
          }

          int l = (I.y / 256) * 8 + (I.x / 256);
          vec4 dx = vec4(0.0);

          for (int u = 0; u < 256; u++) {
            ivec2 h_pos = ivec2((u % 16) * 256 + p.x, (u / 16) * 256 + p.y);
            vec4 val = h0(h_pos);
            dx += val.x * w2t(ivec2(l, 4 * u + 0));
            dx += val.y * w2t(ivec2(l, 4 * u + 1));
            dx += val.z * w2t(ivec2(l, 4 * u + 2));
            dx += val.w * w2t(ivec2(l, 4 * u + 3));
          }

          for (int u = 0; u < 256; u++) {
            ivec2 h_pos = ivec2((u % 16) * 256 + p.x, (u / 16) * 256 + p.y);
            vec4 val = h1(h_pos);
            int base_u = 1024 + 4 * u;
            dx += val.x * w2t(ivec2(l, base_u + 0));
            dx += val.y * w2t(ivec2(l, base_u + 1));
            dx += val.z * w2t(ivec2(l, base_u + 2));
            dx += val.w * w2t(ivec2(l, base_u + 3));
          }

          FOut = Src(I) + dx;
        }
        `,
      },
      nca_grid
    );

    updateAccuracyPlot();
  }

  function applyBrush(p0, p1) {
    if (!nca_grid) return;
    const brushVal = params.brush === 'white' ? 1.0 : 0.0;
    glsl(
      {
        p0: p0,
        p1: p1,
        radius: params.brushSize * 0.5,
        val: brushVal,
        FP: `
        float distToSeg(vec2 p, vec2 a, vec2 b) {
          vec2 pa = p - a, ba = b - a;
          float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-4), 0.0, 1.0);
          return length(pa - ba * h);
        }
        void fragment() {
          FOut = Src(I);
          if (I.x < 252 && I.y < 252) {
            if (distToSeg(vec2(I), p0, p1) <= radius) {
              FOut.r = val;
            }
          }
        }
        `,
      },
      nca_grid
    );
    updateAccuracyPlot();
  }

  function classifyGrid() {
    if (!nca_grid || !templates_tex) return 0;

    pred_tex = glsl(
      {
        grid: nca_grid[0],
        templates: templates_tex,
        FP: `
        void fragment() {
          int cell_c = I.x;
          int cell_r = I.y;

          int best_d = 1;
          float min_dist = 1e9;

          for (int d = 1; d <= 9; d++) {
            float dist = 0.0;
            int t_offset_y = d * 28;
            for (int tr = 0; tr < 28; tr++) {
              int grid_y = cell_r * 28 + tr;
              int temp_y = t_offset_y + tr;
              for (int tc = 0; tc < 28; tc++) {
                int grid_x = cell_c * 28 + tc;
                float board_v = grid(ivec2(grid_x, grid_y)).r;
                float temp_v = templates(ivec2(tc, temp_y)).r;
                dist += abs(board_v - temp_v);
              }
            }
            if (dist < min_dist) {
              min_dist = dist;
              best_d = d;
            }
          }
          FOut = vec4(float(best_d), 0.0, 0.0, 1.0);
        }
        `,
      },
      { size: [9, 9], format: 'rgba32f', tag: 'pred_tex' }
    );

    let num_correct = 0;
    if (
      pred_tex &&
      typeof pred_tex.readSync === 'function' &&
      current_target_grid
    ) {
      try {
        const buf = pred_tex.readSync();
        for (let i = 0; i < 81; i++) {
          const pred = Math.round(buf[i * 4]);
          if (pred === current_target_grid[i]) {
            num_correct++;
          }
        }
      } catch (e) {
        console.warn('readSync error:', e);
      }
    }
    return (num_correct / 81.0) * 100.0;
  }

  function updateAccuracyPlot() {
    const acc = classifyGrid();
    if (
      plot_history.length > 0 &&
      plot_history[plot_history.length - 1][0] === step_count
    ) {
      plot_history[plot_history.length - 1][1] = acc;
    } else {
      plot_history.push([step_count, acc]);
    }
    drawPlot();
  }

  function drawPlot() {
    const plotCanvas = $('#metric-plot');
    if (!plotCanvas) return;
    const ctx = plotCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = plotCanvas.getBoundingClientRect();
    const w =
      rect.width > 0
        ? rect.width
        : plotCanvas.parentElement
          ? plotCanvas.parentElement.clientWidth
          : 280;
    const h = rect.height > 0 ? rect.height : 180;

    if (
      plotCanvas.width !== Math.round(w * dpr) ||
      plotCanvas.height !== Math.round(h * dpr)
    ) {
      plotCanvas.width = Math.round(w * dpr);
      plotCanvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const padL = 46;
    const padR = 14;
    const padT = 16;
    const padB = 28;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    let maxT = 1000;
    let maxY = 100;
    for (const [t] of plot_history) {
      if (t > maxT) maxT = t;
    }

    const mapX = (t) => padL + (t / maxT) * plotW;
    const mapY = (y) => padT + plotH - (y / maxY) * plotH;

    // Grid lines & Y ticks (0%, 50%)
    ctx.strokeStyle = '#e8eaed';
    ctx.lineWidth = 1;
    ctx.font = "11px 'Noto Sans', sans-serif";
    ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = [0, 50];
    for (const yVal of yTicks) {
      const py = mapY(yVal);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.fillText(`${yVal}%`, padL - 6, py);
    }

    // X ticks (0, maxT/2, maxT)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = [0, Math.round(maxT / 2), maxT];
    for (const tVal of xTicks) {
      const px = mapX(tVal);
      ctx.fillText(tVal, px, padT + plotH + 5);
    }

    // Axes
    ctx.strokeStyle = '#bdc1c6';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // Axis labels
    ctx.font = "12px 'Noto Sans', sans-serif";
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.textAlign = 'center';
    ctx.fillText('Step (t)', padL + plotW / 2, padT + plotH + 16);

    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Accuracy (%)', 0, 0);
    ctx.restore();

    // Target horizontal line: y = 100 (target accuracy 100%, dotted green line only, no 100 label text)
    const targetY = mapY(100);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#1b5e20'; // dark green
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, targetY);
    ctx.lineTo(padL + plotW, targetY);
    ctx.stroke();
    ctx.restore();

    // Data curve
    if (plot_history.length > 0) {
      ctx.beginPath();
      ctx.moveTo(mapX(plot_history[0][0]), padT + plotH);
      for (const [t, y] of plot_history) {
        ctx.lineTo(mapX(t), mapY(y));
      }
      ctx.lineTo(mapX(plot_history[plot_history.length - 1][0]), padT + plotH);
      ctx.closePath();
      ctx.fillStyle = 'rgba(70, 130, 180, 0.12)';
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(mapX(plot_history[0][0]), mapY(plot_history[0][1]));
      for (let i = 1; i < plot_history.length; i++) {
        ctx.lineTo(mapX(plot_history[i][0]), mapY(plot_history[i][1]));
      }
      ctx.strokeStyle = 'steelblue';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Latest point marker
      const [latestT, latestY] = plot_history[plot_history.length - 1];
      const curX = mapX(latestT);
      const curY = mapY(latestY);
      ctx.beginPath();
      ctx.arc(curX, curY, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = 'steelblue';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.restore();
  }

  function initUI() {
    // Play/Pause
    const playPauseBtn = $('#play-pause');
    if (playPauseBtn) {
      playPauseBtn.addEventListener('click', () => {
        params.runModel = !params.runModel;
        updatePlayPauseUI();
      });
    }

    // Reset
    const resetBtn = $('#reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        reset();
      });
    }

    // Step
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

    // Speed slider (Row 1)
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
        params.stepFps = 30;
        speedSlider.value = 30;
        speedLabel.textContent = '30 steps/s';
      });
    }

    // New Sudoku (Row 1)
    const newSudokuBtn = $('#new-sudoku');
    if (newSudokuBtn) {
      newSudokuBtn.addEventListener('click', () => {
        newSudoku();
      });
    }

    // Brush size slider (Row 2)
    const brushSizeSlider = $('#brush-size');
    const brushSizeLabel = $('#brushSizeLabel');
    if (brushSizeSlider && brushSizeLabel) {
      brushSizeSlider.addEventListener('input', (e) => {
        params.brushSize = parseFloat(e.target.value);
        brushSizeLabel.textContent = `${params.brushSize.toFixed(1)} px`;
      });
    }
    const resetBrushSizeBtn = $('#reset-brush-size');
    if (resetBrushSizeBtn && brushSizeSlider && brushSizeLabel) {
      resetBrushSizeBtn.addEventListener('click', () => {
        params.brushSize = 2.0;
        brushSizeSlider.value = 2.0;
        brushSizeLabel.textContent = '2.0 px';
      });
    }

    // Brush color radio selector (Row 2)
    const brushRadios = root.querySelectorAll('input[name="brush"]');
    brushRadios.forEach((radio) => {
      radio.addEventListener('change', () => {
        params.brush = radio.value;
        const options = root.querySelectorAll('.brush-option');
        options.forEach((opt) => {
          const r = opt.querySelector('input[type="radio"]');
          if (r && r.checked) {
            opt.classList.add('active');
          } else {
            opt.classList.remove('active');
          }
        });
      });
    });

    // Difficulty dropdown
    const diffSelect = $('#difficulty-select');
    if (diffSelect) {
      diffSelect.addEventListener('change', (e) => {
        params.difficulty = e.target.value;
        newSudoku();
      });
    }

    // Plot toggle
    const plotToggle = $('#plot-toggle');
    const plotContainer = $('#plot-container');
    const plotTitle = $('#plot-toggle-title');
    if (plotToggle && plotContainer) {
      plotToggle.onclick = () => {
        const isVisible = plotContainer.classList.toggle('visible');
        const icon = plotToggle.querySelector('.plot-toggle-icon');
        if (icon) {
          icon.classList.toggle('expanded', isVisible);
        }
        if (plotTitle) {
          plotTitle.textContent = isVisible
            ? 'Hide Accuracy'
            : 'Show Accuracy';
        }
        if (isVisible) {
          drawPlot();
        }
      };
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

    // Mouse / Touch drawing on canvas
    let isDrawing = false;
    let lastPos = [-1, -1];

    function getCanvasPos(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX =
        e.clientX !== undefined
          ? e.clientX
          : e.touches && e.touches[0]
            ? e.touches[0].clientX
            : 0;
      const clientY =
        e.clientY !== undefined
          ? e.clientY
          : e.touches && e.touches[0]
            ? e.touches[0].clientY
            : 0;
      const nx = (clientX - rect.left) / rect.width;
      const ny = (clientY - rect.top) / rect.height;
      return [nx * 252.0, ny * 252.0];
    }

    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      isDrawing = true;
      const pos = getCanvasPos(e);
      lastPos = pos;
      applyBrush(pos, pos);
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!isDrawing) return;
      const pos = getCanvasPos(e);
      applyBrush(lastPos, pos);
      lastPos = pos;
    });

    const stopDrawing = (e) => {
      if (isDrawing) {
        isDrawing = false;
        lastPos = [-1, -1];
      }
    };
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointercancel', stopDrawing);

    window.addEventListener('resize', () => {
      drawPlot();
    });
  }

  function frame(t) {
    glsl.adjustCanvas();
    const now = t || performance.now();
    if (params.runModel) {
      const stepInterval = 1000.0 / params.stepFps;
      if (now - lastStepTime >= stepInterval) {
        step();
        lastStepTime = now;
      }
    }

    if (nca_grid) {
      glsl({
        grid: nca_grid[0],
        Aspect: 'fit',
        size: [252, 252],
        FP: `
        void fragment() {
          ivec2 p = clamp(ivec2(UV.x * 252.0, (1.0 - UV.y) * 252.0), ivec2(0), ivec2(251));
          float v = clamp(grid(p).r, 0.0, 1.0);
          vec3 col = vec3(v);

          // 9x9 Sudoku grid overlay
          if (p.x % 84 == 0 || p.y % 84 == 0) {
            col = mix(col, vec3(0.95, 0.95, 0.95), 0.35); // 3x3 block borders
          } else if (p.x % 28 == 0 || p.y % 28 == 0) {
            col = mix(col, vec3(0.5), 0.18); // cell borders
          }

          FOut = vec4(col, 1.0);
        }
        `,
      });
    }

    requestAnimationFrame(frame);
  }

  async function init() {
    try {
      // 1. Load Visual Sudoku dataset
      const response = await fetch(dset_path);
      dsets = await response.json();
      for (const split of ['easy', 'hard']) {
        dsets[split].input_masks.array = decodeBase64(
          dsets[split].input_masks.data,
          dsets[split].input_masks.dtype
        );
        dsets[split].target_grids.array = decodeBase64(
          dsets[split].target_grids.data,
          dsets[split].target_grids.dtype
        );
      }
      dsets.digit_images.array = decodeBase64(
        dsets.digit_images.data,
        dsets.digit_images.dtype
      );
      dsets.digit_templates.array = decodeBase64(
        dsets.digit_templates.data,
        dsets.digit_templates.dtype
      );

      // Create digit templates texture: [28, 280] r32f
      templates_tex = glsl(
        {},
        {
          size: [28, 280],
          format: 'r32f',
          data: dsets.digit_templates.array,
          tag: 'templates',
        }
      );

      // 2. Load Visual Sudoku NCA model
      const response2 = await fetch(model_path);
      const raw_model = await response2.json();

      const w1_raw = decodeBase64(raw_model.update.w1.data, 'float32');
      const w2_raw = decodeBase64(raw_model.update.w2.data, 'float32');

      // Format w1 texture [192, 2048]
      const w1_data = new Float32Array(192 * 2048 * 4);
      for (let h = 0; h < 2048; h++) {
        const row_offset = h * 192 * 4;
        for (let i = 0; i < 192; i++) {
          const c0 = (4 * i + 0) * 2048 + h;
          const c1 = (4 * i + 1) * 2048 + h;
          const c2 = (4 * i + 2) * 2048 + h;
          const c3 = (4 * i + 3) * 2048 + h;
          const dst = row_offset + i * 4;
          w1_data[dst + 0] = w1_raw[c0];
          w1_data[dst + 1] = w1_raw[c1];
          w1_data[dst + 2] = w1_raw[c2];
          w1_data[dst + 3] = w1_raw[c3];
        }
      }
      w1_tex = glsl(
        {},
        { size: [192, 2048], format: 'rgba32f', data: w1_data, tag: 'w1' }
      );

      // Format w2 texture [64, 2048]
      w2_tex = glsl(
        {},
        { size: [64, 2048], format: 'rgba32f', data: w2_raw, tag: 'w2t' }
      );

      // 3. Initialize UI & Event listeners
      initUI();

      // 4. Start first puzzle
      newSudoku();

      // 5. Start render loop
      requestAnimationFrame(frame);
    } catch (err) {
      console.error('Failed to initialize Visual Sudoku demo:', err);
    }
  }

  init();
}

export const createDemo = createDemoVsudoku;
