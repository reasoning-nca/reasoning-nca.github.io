export function createDemoSext(glsl, divId, onCanvasRendered = null) {
  const root = document.getElementById(divId);
  const $ = (q) => root.querySelector(q);
  const $$ = (q) => root.querySelectorAll(q);

  const canvas = $("#demo-canvas");

  const H = 9, W = 9;
  const CHN = 256;
  const INPUT_EMB_D = 32;
  const NUM_UNITS = 8;
  const HEADS = 16;
  const HEAD_DIM = 16;     // 256 / 16
  const HIDDEN_DIM = 1024; // 256 * 4

  const dset_path = "../assets/json/sextdata.json";
  const model_path = "../assets/json/sextnca.json";

  const dsets = {};
  const model = {};
  let digits = {};

  // NCA State & Simulation variables
  const x_state = new Float32Array(81 * CHN);
  const input_masks = new Uint8Array(81);
  const input_digits = new Uint8Array(81);
  const target_digits = new Uint8Array(81);
  const pred_digits = new Float32Array(81);

  let step_count = 0;
  let lastTime = null;
  let stepAccumulator = 0;
  let plot_history = [];

  // Precomputed token embedding norms
  const ie_norms = new Float32Array(10);

  // Neighbor offsets for 3x3 wrapping torus
  const NEIGHBORS = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 0], [0, 1],
    [1, -1], [1, 0], [1, 1],
  ];

  // SwissGL target textures for rendering
  let input_grid_tex, target_grid_tex, pred_grid_tex;

  const params = {
    difficulty: "0-1",
    runModel: false,
    fireRate: 0.8,
    set_mode: "correct",
    stepsPerSecond: 15,
  };

  // Reusable buffers for step computation
  const val_buf = new Float32Array(81 * HEADS * HEAD_DIM);
  const z_buf = new Float32Array(81 * CHN);
  const h_buf = new Float32Array(HIDDEN_DIM);
  const dx_buf = new Float32Array(CHN);

  function updatePlayPauseUI() {
    const playBtn = $("#play");
    const pauseBtn = $("#pause");
    if (playBtn && pauseBtn) {
      playBtn.style.display = params.runModel ? "none" : "inline";
      pauseBtn.style.display = params.runModel ? "inline" : "none";
    }
  }

  function updateStepCountUI() {
    const stepBadge = $("#step-count");
    if (stepBadge) {
      stepBadge.textContent = `Step: ${step_count}`;
    }
  }

  function decodeBase64(dataStr, dtype) {
    const binary = atob(dataStr);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    if (dtype === "float32") {
      return new Float32Array(bytes.buffer);
    } else if (dtype === "uint8" || dtype === "bool") {
      return bytes;
    } else if (dtype === "int32") {
      return new Int32Array(bytes.buffer);
    }
    return bytes;
  }

  function createDigitTextures() {
    const d_tex = {};
    for (let d = 0; d <= 9; d++) {
      const off = document.createElement("canvas");
      off.width = 128;
      off.height = 128;
      const ctx = off.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, 128, 128);

      if (d > 0) {
        ctx.fillStyle = "#000000";
        ctx.font = "bold 64px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(d.toString(), 64, 68);
      }

      d_tex["d" + d] = glsl({}, { data: off, tag: "d" + d });
    }
    return d_tex;
  }

  function updateGLTextures() {
    input_grid_tex = glsl(
      {},
      { size: [W, H], format: "r32f", data: new Float32Array(input_digits), tag: "input_grid" },
    );
    target_grid_tex = glsl(
      {},
      { size: [W, H], format: "r32f", data: new Float32Array(target_digits), tag: "target_grid" },
    );
    pred_grid_tex = glsl(
      {},
      { size: [W, H], format: "r32f", data: pred_digits, tag: "pred_grid" },
    );
  }

  function countViolatingCells(digits) {
    let count = 0;
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const d = Math.round(digits[r * 9 + c]);
        if (d === 0) {
          count++;
          continue;
        }
        let violates = false;
        // Check row r
        for (let col = 0; col < 9; col++) {
          if (col !== c && Math.round(digits[r * 9 + col]) === d) {
            violates = true;
            break;
          }
        }
        // Check col c
        if (!violates) {
          for (let row = 0; row < 9; row++) {
            if (row !== r && Math.round(digits[row * 9 + c]) === d) {
              violates = true;
              break;
            }
          }
        }
        // Check 3x3 block
        if (!violates) {
          const br = Math.floor(r / 3) * 3;
          const bc = Math.floor(c / 3) * 3;
          for (let row = br; row < br + 3; row++) {
            for (let col = bc; col < bc + 3; col++) {
              if ((row !== r || col !== c) && Math.round(digits[row * 9 + col]) === d) {
                violates = true;
                break;
              }
            }
            if (violates) break;
          }
        }
        if (violates) count++;
      }
    }
    return count;
  }

  function updateViolationsPlot() {
    const v = countViolatingCells(pred_digits);
    plot_history.push([step_count, v]);
    drawPlot();
  }

  function drawPlot() {
    const plotCanvas = $("#metric-plot");
    if (!plotCanvas) return;
    const ctx = plotCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = plotCanvas.getBoundingClientRect();
    const w = rect.width > 0 ? rect.width : (plotCanvas.parentElement ? plotCanvas.parentElement.clientWidth : 280);
    const h = rect.height > 0 ? rect.height : 180;

    if (plotCanvas.width !== Math.round(w * dpr) || plotCanvas.height !== Math.round(h * dpr)) {
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

    let maxT = 20;
    let maxY = 81;
    for (const [t, y] of plot_history) {
      if (t > maxT) maxT = t;
      if (y > maxY) maxY = y;
    }
    maxT = Math.ceil(maxT * 1.05);
    maxY = Math.max(81, Math.ceil(maxY * 1.05));

    const mapX = (t) => padL + (t / maxT) * plotW;
    const mapY = (y) => padT + plotH - (y / maxY) * plotH;

    // Grid lines & Y ticks
    ctx.strokeStyle = "#e8eaed";
    ctx.lineWidth = 1;
    ctx.font = "11px 'Noto Sans', sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const yTicks = [0, Math.round(maxY / 2), maxY];
    for (const yVal of yTicks) {
      const py = mapY(yVal);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.fillText(yVal, padL - 6, py);
    }

    // X ticks
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = [0, Math.round(maxT / 2), maxT];
    for (const tVal of xTicks) {
      const px = mapX(tVal);
      ctx.fillText(tVal, px, padT + plotH + 5);
    }

    // Axes
    ctx.strokeStyle = "#bdc1c6";
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // Axis labels
    ctx.font = "12px 'Noto Sans', sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
    ctx.textAlign = "center";
    ctx.fillText("Step (t)", padL + plotW / 2, padT + plotH + 16);

    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("#Violating cells", 0, 0);
    ctx.restore();

    // Target horizontal line at y = 0
    const targetY = mapY(0);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#1b5e20"; // dark green
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
      ctx.fillStyle = "rgba(70, 130, 180, 0.12)";
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(mapX(plot_history[0][0]), mapY(plot_history[0][1]));
      for (let i = 1; i < plot_history.length; i++) {
        ctx.lineTo(mapX(plot_history[i][0]), mapY(plot_history[i][1]));
      }
      ctx.strokeStyle = "steelblue";
      ctx.lineWidth = 2;
      ctx.stroke();

      const lastPt = plot_history[plot_history.length - 1];
      const lx = mapX(lastPt[0]);
      const ly = mapY(lastPt[1]);
      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "steelblue";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.font = "11px 'Noto Sans', sans-serif";
      ctx.fillStyle = "steelblue";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`${lastPt[1]}`, Math.min(lx + 5, padL + plotW - 18), ly - 6);
    }

    ctx.restore();
  }

  function predict() {
    const ie = model.input_embed;

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

      for (let t = 0; t < 10; t++) {
        let dot = 0.0;
        const base_ie = t * INPUT_EMB_D;
        for (let c = 0; c < INPUT_EMB_D; c++) {
          dot += x_state[base_x + c] * ie[base_ie + c];
        }
        const sim = dot / (x_norm * ie_norms[t]);
        if (sim > best_sim) {
          best_sim = sim;
          best_digit = t;
        }
      }

      pred_digits[i] = best_digit;
    }

    updateGLTextures();
  }

  function randGaussian(mean, std) {
    let u1 = Math.random();
    let u2 = Math.random();
    while (u1 <= 1e-7) u1 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  }

  function reset() {
    step_count = 0;
    stepAccumulator = 0;
    updateStepCountUI();
    const ie = model.input_embed;

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
        // Empty cell: all 256 channels initialized with Gaussian noise
        for (let c = 0; c < CHN; c++) {
          x_state[base_x + c] = randGaussian(0, 0.15);
        }
      }

      // Group norm across 8 units (each unit is 32 channels: 8 * 32 = 256)
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

    predict();
    plot_history = [[0, countViolatingCells(pred_digits)]];
    drawPlot();
  }

  function newSudoku() {
    const dset = dsets[params.difficulty];
    if (!dset) return;

    const num_samples = dset.input_masks.shape[0];
    const sample_idx = Math.floor(Math.random() * num_samples);

    const mask_raw = dset.input_masks.data.slice(sample_idx * 81, (sample_idx + 1) * 81);
    const target_raw = dset.target_grids.data.slice(sample_idx * 81, (sample_idx + 1) * 81);

    for (let i = 0; i < 81; i++) {
      input_masks[i] = mask_raw[i] ? 1 : 0;
      target_digits[i] = target_raw[i];
      input_digits[i] = mask_raw[i] ? target_raw[i] : 0;
    }

    reset();
  }

  function step() {
    const v = model.v;
    const fa = model.fixed_attention;
    const w1 = model.w1;
    const b1 = model.b1;
    const w2 = model.w2;
    const b2 = model.b2;

    // 1. Project all 81 cells with V matrix: [81, 16, 16]
    for (let i = 0; i < 81; i++) {
      const base_x = i * CHN;
      const base_val_cell = i * 256; // 16 * 16
      for (let m = 0; m < HEADS; m++) {
        const base_v_head = m * 4096; // 16 * 256
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
        const base_fa_cell = i * 144; // 16 * 9

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
      for (let j = 0; j < HIDDEN_DIM; j++) {
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
        const base_w2 = c * HIDDEN_DIM;
        for (let j = 0; j < HIDDEN_DIM; j++) {
          dot += w2[base_w2 + j] * h_buf[j];
        }
        dx_buf[c] = dot;
      }

      // Clue cells (clamped) keep token embedding channels fixed
      if (input_masks[i] > 0) {
        for (let c = 0; c < INPUT_EMB_D; c++) {
          dx_buf[c] = 0.0;
        }
      }

      // proj_emb: project dx onto tangent space of x for each unit
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

    step_count++;
    updateStepCountUI();
    predict();
    updateViolationsPlot();
  }

  function getGridPos(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const c = Math.min(canvas.clientWidth, canvas.clientHeight);
    const offsetX = (canvas.clientWidth - c) / 2.0;
    const offsetY = (canvas.clientHeight - c) / 2.0;
    const normX = (mouseX - offsetX) / c;
    const normY = (mouseY - offsetY) / c;

    if (normX < 0.0 || normX >= 1.0 || normY < 0.0 || normY >= 1.0) {
      return [-1, -1];
    }
    const gx = Math.floor(normX * 9);
    const gy = Math.floor(normY * 9);
    return [gx, gy];
  }

  function brush(gx, gy) {
    if (gx < 0 || gx >= 9 || gy < 0 || gy >= 9) return;
    const cell_idx = gy * 9 + gx;

    input_digits[cell_idx] = 0;
    input_masks[cell_idx] = 0;

    const base_x = cell_idx * CHN;
    const ie = model.input_embed;

    if (params.set_mode === "correct") {
      const correct_digit = target_digits[cell_idx];
      // First 32 channels (unit 0) set to token embedding of the correct digit
      for (let c = 0; c < INPUT_EMB_D; c++) {
        x_state[base_x + c] = ie[correct_digit * INPUT_EMB_D + c];
      }
      // Hidden channels (units 1..7, channels 32..255) reset to random noise (as at t=0)
      for (let c = INPUT_EMB_D; c < CHN; c++) {
        x_state[base_x + c] = randGaussian(0, 0.15);
      }
    } else {
      // Random vector: all 256 channels initialized with Gaussian noise (as at t=0)
      for (let c = 0; c < CHN; c++) {
        x_state[base_x + c] = randGaussian(0, 0.15);
      }
    }

    // Group norm across all 8 units (each unit is 32 channels: 8 * 32 = 256, as at t=0)
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

    predict();
    const v = countViolatingCells(pred_digits);
    if (plot_history.length > 0 && plot_history[plot_history.length - 1][0] === step_count) {
      plot_history[plot_history.length - 1][1] = v;
    } else {
      plot_history.push([step_count, v]);
    }
    drawPlot();
  }

  function setSetMode(mode) {
    params.set_mode = mode;
    $$('input[name="set_mode"]').forEach((radio) => {
      radio.checked = radio.value === params.set_mode;
    });
    updateSetModeActiveStyle();
  }

  function updateSetModeActiveStyle() {
    $$("#set_mode .mode-btn").forEach((label) => {
      const radio = label.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        label.classList.add("active");
      } else {
        label.classList.remove("active");
      }
    });
  }

  function setDifficulty(diff) {
    params.difficulty = diff;
    $$("#difficulty-shelf .subcard").forEach((card) => {
      if (card.getAttribute("data-value") === diff) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });
    const diffSelect = $("#difficulty");
    if (diffSelect && diffSelect.value !== diff) {
      diffSelect.value = diff;
    }
    newSudoku();
  }

  function init_event_listeners() {
    let prevPos = [-1, -1];
    canvas.onmousedown = (e) => {
      e.preventDefault();
      if (e.buttons === 1) {
        const pos = getGridPos(e);
        if (pos[0] >= 0) {
          brush(pos[0], pos[1]);
          prevPos = pos;
        }
      }
    };

    canvas.onmousemove = (e) => {
      e.preventDefault();
      if (e.buttons === 1) {
        const pos = getGridPos(e);
        if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
          brush(pos[0], pos[1]);
          prevPos = pos;
        }
      }
    };

    canvas.onmouseup = () => {
      prevPos = [-1, -1];
    };

    canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const t = e.changedTouches[0];
      const pos = getGridPos(t);
      if (pos[0] >= 0) {
        brush(pos[0], pos[1]);
        prevPos = pos;
      }
    });

    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const t = e.changedTouches[0];
      const pos = getGridPos(t);
      if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
        brush(pos[0], pos[1]);
        prevPos = pos;
      }
    });

    canvas.addEventListener("touchend", () => {
      prevPos = [-1, -1];
    });

    // Control button wiring
    const playPauseBtn = $("#play-pause");
    if (playPauseBtn) {
      playPauseBtn.onclick = () => {
        params.runModel = !params.runModel;
        updatePlayPauseUI();
      };
    }

    const resetBtn = $("#reset");
    if (resetBtn) {
      resetBtn.onclick = () => reset();
    }

    const stepBtn = $("#step");
    if (stepBtn) {
      stepBtn.onclick = () => step();
    }

    const newSudokuBtn = $("#new-sudoku");
    if (newSudokuBtn) {
      newSudokuBtn.onclick = () => newSudoku();
    }

    const plotToggle = $("#plot-toggle");
    const plotContainer = $("#plot-container");
    const plotTitle = $("#plot-toggle-title");
    if (plotToggle && plotContainer) {
      plotToggle.onclick = () => {
        const isVisible = plotContainer.classList.toggle("visible");
        const icon = plotToggle.querySelector(".plot-toggle-icon");
        if (icon) {
          icon.classList.toggle("expanded", isVisible);
        }
        if (plotTitle) {
          plotTitle.textContent = isVisible ? "Hide #Violating Cells" : "Show #Violating Cells";
        }
        if (isVisible) {
          drawPlot();
        }
      };
    }

    const diffSelect = $("#difficulty");
    if (diffSelect) {
      diffSelect.onchange = (e) => {
        setDifficulty(e.target.value);
      };
    }

    const speedRange = $("#speed");
    const resetSpeedBtn = $("#reset-speed");
    if (speedRange) {
      speedRange.oninput = (e) => {
        const val = parseInt(e.target.value, 10);
        params.stepsPerSecond = val;
        const speedLabel = $("#speedLabel");
        if (speedLabel) {
          speedLabel.textContent = `${val} steps/s`;
        }
      };
      speedRange.ondblclick = () => {
        if (resetSpeedBtn) resetSpeedBtn.click();
      };
    }
    if (resetSpeedBtn) {
      resetSpeedBtn.onclick = () => {
        params.stepsPerSecond = 15;
        if (speedRange) speedRange.value = 15;
        const speedLabel = $("#speedLabel");
        if (speedLabel) {
          speedLabel.textContent = "15 steps/s";
        }
      };
    }

    const fireRateRange = $("#fire-rate");
    const resetFireRateBtn = $("#reset-fire-rate");
    if (fireRateRange) {
      fireRateRange.oninput = (e) => {
        const val = parseFloat(e.target.value);
        params.fireRate = val;
        const fireRateLabel = $("#fireRateLabel");
        if (fireRateLabel) {
          fireRateLabel.textContent = val.toFixed(2);
        }
      };
      fireRateRange.ondblclick = () => {
        if (resetFireRateBtn) resetFireRateBtn.click();
      };
    }
    if (resetFireRateBtn) {
      resetFireRateBtn.onclick = () => {
        params.fireRate = 0.8;
        if (fireRateRange) fireRateRange.value = 0.8;
        const fireRateLabel = $("#fireRateLabel");
        if (fireRateLabel) {
          fireRateLabel.textContent = "0.80";
        }
      };
    }

    $$('input[name="set_mode"]').forEach((radio) => {
      radio.onchange = (e) => {
        if (e.target.checked) {
          params.set_mode = e.target.value;
          updateSetModeActiveStyle();
        }
      };
    });
  }

  function frame(time_ms) {
    glsl.adjustCanvas();

    const time = (time_ms || performance.now()) / 1000.0;
    if (lastTime === null) {
      lastTime = time;
    }
    const dt = Math.min(Math.max(time - lastTime, 0.0), 0.2);
    lastTime = time;

    if (params.runModel) {
      stepAccumulator += dt * params.stepsPerSecond;
      const numSteps = Math.floor(stepAccumulator);
      for (let i = 0; i < numSteps; i++) {
        step();
      }
      stepAccumulator -= numSteps;
    } else {
      stepAccumulator = 0;
    }

    // Render Sudoku board with SwissGL
    glsl({
      ...digits,
      pred_grid: pred_grid_tex,
      input_grid: input_grid_tex,
      target_grid: target_grid_tex,
      H: 9,
      W: 9,
      Grid: [9, 9],
      Aspect: "fit",
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

    if (typeof onCanvasRendered === "function") {
      onCanvasRendered();
    }

    requestAnimationFrame(frame);
  }

  function reset_ui() {
    updatePlayPauseUI();
    updateStepCountUI();
    updateSetModeActiveStyle();

    const speedInput = $("#speed");
    if (speedInput) {
      speedInput.value = params.stepsPerSecond;
      const speedLabel = $("#speedLabel");
      if (speedLabel) {
        speedLabel.textContent = `${params.stepsPerSecond} steps/s`;
      }
    }

    const fireRateInput = $("#fire-rate");
    if (fireRateInput) {
      fireRateInput.value = params.fireRate;
      const fireRateLabel = $("#fireRateLabel");
      if (fireRateLabel) {
        fireRateLabel.textContent = params.fireRate.toFixed(2);
      }
    }
  }

  async function init() {
    init_event_listeners();
    reset_ui();

    // 1. Create digit textures
    digits = createDigitTextures();

    // 2. Load Reasoning NCA model
    const model_res = await fetch(model_path);
    const model_json = await model_res.json();

    model.input_embed = decodeBase64(model_json.input_embed.data, "float32");
    model.fixed_attention = decodeBase64(model_json.sensing.fixed_attention.data, "float32");
    model.v = decodeBase64(model_json.sensing.v.data, "float32");
    model.w1 = decodeBase64(model_json.update.w1.data, "float32");
    model.b1 = decodeBase64(model_json.update.b1.data, "float32");
    model.w2 = decodeBase64(model_json.update.w2.data, "float32");
    model.b2 = decodeBase64(model_json.update.b2.data, "float32");

    // Precalculate input_embed norms
    for (let t = 0; t < 10; t++) {
      let sum_sq = 0.0;
      for (let c = 0; c < INPUT_EMB_D; c++) {
        const val = model.input_embed[t * INPUT_EMB_D + c];
        sum_sq += val * val;
      }
      ie_norms[t] = Math.sqrt(sum_sq) + 1e-6;
    }

    // 3. Load Sudoku-Extreme dataset
    const dset_res = await fetch(dset_path);
    const dset_json = await dset_res.json();

    for (const diff_key in dset_json) {
      dsets[diff_key] = {
        input_masks: {
          data: decodeBase64(dset_json[diff_key].input_masks.data, "bool"),
          shape: dset_json[diff_key].input_masks.shape,
        },
        target_grids: {
          data: decodeBase64(dset_json[diff_key].target_grids.data, "uint8"),
          shape: dset_json[diff_key].target_grids.shape,
        },
      };
    }

    // Populate difficulty subcards & options
    const diffShelf = $("#difficulty-shelf");
    const diffSelect = $("#difficulty");
    const diff_keys = Object.keys(dsets);
    if (diff_keys.length > 0) {
      params.difficulty = diff_keys[0];
      if (diffShelf) {
        diffShelf.innerHTML = "";
        diff_keys.forEach((k) => {
          const card = document.createElement("div");
          card.className = `subcard ${k === params.difficulty ? "active" : ""}`;
          card.setAttribute("data-value", k);
          card.textContent = k;
          card.onclick = () => setDifficulty(k);
          diffShelf.appendChild(card);
        });
      }
      if (diffSelect) {
        diffSelect.innerHTML = "";
        diff_keys.forEach((k) => {
          const opt = document.createElement("option");
          opt.value = k;
          opt.textContent = `Difficulty: ${k}`;
          diffSelect.appendChild(opt);
        });
        diffSelect.value = params.difficulty;
      }
    }

    // 4. Initialize first puzzle and launch animation
    newSudoku();
    requestAnimationFrame(frame);
  }

  init();

  return {
    reset,
    step,
    newSudoku,
    setSetMode,
    setBrush: setSetMode,
    setDifficulty,
  };
}

export const createDemo = createDemoSext;
