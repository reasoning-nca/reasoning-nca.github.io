// Target Output Embeddings for ARC tokens 0..11 into layers 40..47 (channels 160..191)
// Precomputed via pseudoinverse of pred_head (W, b) such that argmax logit is exactly token k
const TARGET_OUTPUT_EMBEDDINGS = [
  [-0.305163, -0.820224, 0.253624, 0.064689, -0.297673, 0.452004, -0.195183, -0.16288, 0.068411, 1.53111, -0.142586, -0.079979, 0.330559, 0.108267, -1.501145, -0.393623, 0.372879, -0.172546, 0.981702, 1.06198, 0.011406, 0.683075, -0.090166, 0.504762, 0.091631, 0.087938, 0.475401, -0.021741, 0.273894, -0.868803, 0.380194, 0.62152],
  [0.035763, -1.190087, 0.050468, -0.191828, 0.039987, 0.459976, -0.624459, 0.017706, -0.062742, 1.898281, 0.284667, -0.061441, 0.580481, 0.134846, -1.719146, 0.188843, 0.117101, -0.321704, 0.253931, 1.283934, -0.658321, 0.448012, -0.744076, -0.152598, -0.156972, -0.005216, -0.297097, -0.022698, -0.352336, -0.884061, 1.005075, 0.7542],
  [-0.301859, -0.924235, 0.032497, -0.167102, 0.066397, 0.311696, -0.149919, 0.097351, 0.220036, 1.340308, 0.049269, -0.388378, 0.307847, -0.019608, -1.420934, -0.168186, 0.517268, -0.242781, 1.009662, 1.164152, -0.785872, 0.297499, -0.639774, 0.229238, 0.392018, -0.079277, -0.061024, 0.619875, -0.114037, -0.0385, 0.566778, 1.494911],
  [-0.417494, -0.958601, -0.201704, 0.105937, -0.301596, 0.753418, -0.603797, -0.225409, 0.372923, 1.262155, -0.230281, -0.094536, 0.22974, -0.037038, -1.24525, 0.099228, 0.305864, -0.912918, 0.238156, 1.110243, -0.273622, 0.710851, -0.241376, -0.095656, 0.651465, 0.526776, -0.041871, -0.223227, -0.054427, -0.81587, 0.514506, 1.503625],
  [-0.395477, -1.192277, -0.263797, -0.21055, 0.016021, 0.764427, -0.000728, -0.897665, 0.540671, 1.712511, 0.362605, 0.021878, 0.147749, 0.386128, -1.530501, -0.433242, 0.0153, -0.382976, 0.485138, 0.638375, -0.30902, 0.531054, -0.679809, 0.284775, 0.440236, 0.090417, -0.038565, -0.038467, -0.083084, -0.348449, 0.65185, 0.934662],
  [-0.418331, -0.791464, -0.379063, 0.288387, -0.499421, 0.71187, 0.095344, 0.162127, -0.157327, 1.859993, 0.421917, -0.657344, 0.717446, 0.400098, -1.379494, 0.048832, 0.235537, -0.644779, 0.192601, 0.723469, -0.347361, 0.64271, -0.31692, 0.348181, -0.102898, 0.279545, 0.435938, 0.318519, -0.050247, -0.309655, 0.816781, 1.391689],
  [-0.096171, -0.819582, -0.459049, -0.264687, -0.523367, 0.75177, -0.710279, 0.261522, 0.568154, 1.532763, 0.382844, -0.503546, 0.100557, -0.0585, -2.203264, -0.586921, 0.204724, -0.81734, 0.454986, 0.820259, -0.805784, 0.429691, -0.364295, 0.250056, -0.048416, -0.010508, 0.127629, 0.498179, 0.330498, -0.820179, 0.780078, 1.017066],
  [0.269401, -1.003733, -0.195665, 0.22985, 0.295669, 0.710412, -0.748164, 0.26817, 0.538268, 1.655657, -0.337003, -0.088951, 0.767369, 0.37115, -1.033508, -0.473048, 0.071724, -0.725219, 0.802788, 0.790502, -0.312975, 0.47286, -0.367104, -0.018046, 0.475884, -0.05091, 0.291277, 0.231611, 0.200905, -0.184793, 0.866846, 0.867969],
  [0.062921, -0.89473, 0.627548, 0.562146, 0.130264, 0.987584, -0.204992, -0.01047, 0.498134, 1.502386, 0.476224, -0.111631, 0.204477, -0.050517, -2.090361, 0.1211, 0.108618, -0.407385, 0.585796, 0.857171, -0.390578, 0.686033, -0.490009, -0.027074, 0.390919, -0.015481, 0.473959, 0.347711, -0.17112, -0.773736, 0.934068, 1.365142],
  [0.136386, -1.091332, -0.017305, 0.108646, 0.006813, 0.409823, -0.31533, -0.260505, 0.011153, 1.411603, -0.040352, -0.444176, 0.162135, -0.036177, -1.368662, -0.305075, -0.529032, -0.286459, 0.679935, 0.500407, -0.643637, 0.465146, -0.635783, -0.123737, -0.148829, 0.41249, 0.591696, -0.16559, 0.234866, -0.455347, 0.698148, 1.454555],
  [0.043866, -0.94803, 0.04477, 0.217265, -0.286132, 0.444992, -0.395708, -0.376465, 0.614424, 1.455945, 0.109563, -0.570827, 0.680092, 0.415195, -1.937572, -0.004151, 0.384086, -0.620452, 0.587138, 1.018657, -0.699751, 0.769646, -0.694008, 0.092762, 0.150515, 0.276333, 0.200954, 0.013585, 0.227142, -0.329286, 0.727611, 1.045739],
  [-0.09633, -1.98709, -0.107278, -0.04136, -0.288845, 1.125258, -0.475347, -0.110393, 0.277973, 3.154022, -0.167604, -0.570128, 0.601952, 0.060044, -3.02012, -0.266539, 0.046937, -1.244804, 1.215251, 1.720926, -0.619283, 1.296307, -1.1241, 0.468053, 0.508821, 0.152448, 0.174944, 0.278397, 0.064958, -0.993627, 1.504275, 2.184133]
];

export function createDemoArc1(glsl, divId) {
  const root = document.getElementById(divId);
  const $ = (q) => root.querySelector(q);
  const $$ = (q) => root.querySelectorAll(q);

  const canvas = $("#demo-canvas");
  const inputCanvas = $("#input-canvas");
  const targetCanvas = $("#target-canvas");

  // Model Architecture & Grid Dimensions
  const H = 32, W = 32;
  const CHN = 512;
  const INPUT_EMB_D = 32;
  const TASK_EMB_D = 128;
  const OUTPUT_EMB_D = 32;
  const NUM_UNITS = 32;
  const MUT_UNITS = 30; // 30 mutable groups of 16 channels each
  const MAX_STEPS = 64;

  const ARC_PALETTE = [
    "#000000", // 0: Black
    "#0074D9", // 1: Blue
    "#FF4136", // 2: Red
    "#2ECC40", // 3: Green
    "#FFDC00", // 4: Yellow
    "#AAAAAA", // 5: Grey
    "#F012BE", // 6: Magenta
    "#FF851B", // 7: Orange
    "#7FDBFF", // 8: Teal
    "#870E25", // 9: Maroon
    "#333333", // 10: Border
    "#f8f9fa", // 11: Pad
  ];

  const dset_path = "../assets/json/arc1data.json";
  const model_path = "../assets/json/arc1nca.json";

  let dset_json = null;
  let task_ids = [];
  const model_raw = {};

  // Model Weight Textures for GPU
  let v_tex, wg_tex, w1_tex, w2_tex;
  let bg_tex, b1_tex, b2_tex;
  let step_emb_tex, fa_tex, pw_tex, pb_tex;

  // GPU Target Textures
  let nca_grid = null;        // [512, 256] rgba32f, double-buffered
  let val_grid = null;        // [512, 256] rgba32f
  let z_grid = null;          // [512, 256] rgba32f
  let h_grid = null;          // [1024, 512] rgba32f
  let dx_grid = null;         // [512, 256] rgba32f
  let pred_grid_tex = null;   // [32, 32] rgba32f
  let input_grid_tex = null;  // [32, 32] r32f
  let target_grid_tex = null; // [32, 32] r32f

  // CPU-side state tracking
  const input_digits = new Int32Array(H * W);
  const target_digits = new Int32Array(H * W);
  const pred_digits = new Int32Array(H * W);

  let step_count = 0;
  let lastTime = null;
  let stepAccumulator = 0;
  let isExactMatch = false;
  let plot_history = [];

  const params = {
    task_id: "1a07d186",
    split: "test",
    sample_idx: 0,
    runModel: false,
    stepsPerSecond: 20,
    fireRate: 1.0,
    cropPadding: true,
    brush_mode: "input", // "input" or "output"
    brush: "1",          // color 0..9
  };

  function updatePlayPauseUI() {
    const playBtn = $("#play");
    const pauseBtn = $("#pause");
    if (playBtn && pauseBtn) {
      playBtn.style.display = params.runModel ? "none" : "inline";
      pauseBtn.style.display = params.runModel ? "inline" : "none";
    }
  }

  function updateHeaderUI() {
    const stepCountEl = $("#step-count");
    if (stepCountEl) {
      stepCountEl.textContent = `Step: ${step_count} / ${MAX_STEPS}`;
    }
    const taskIdBadge = $("#task-id-badge");
    if (taskIdBadge) {
      taskIdBadge.textContent = `Task: ${params.task_id}`;
    }
  }

  function decodeBase64(dataStr, dtype) {
    const binary = atob(dataStr);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    if (dtype === "float32" || dtype === "<f4") {
      return new Float32Array(bytes.buffer);
    } else if (dtype === "int32" || dtype === "<i4") {
      return new Int32Array(bytes.buffer);
    } else if (dtype === "uint8" || dtype === "|u1" || dtype === "bool") {
      return bytes;
    }
    return new Float32Array(bytes.buffer);
  }

  function randGaussian(mean = 0.0, std = 1.0) {
    let u1 = Math.random();
    let u2 = Math.random();
    while (u1 <= 1e-15) u1 = Math.random();
    return mean + std * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  function getTargetBounds(target) {
    let r_min = 32, r_max = -1, c_min = 32, c_max = -1;
    for (let i = 0; i < 1024; i++) {
      const t_val = target[i];
      if (t_val < 10) {
        const r = Math.floor(i / 32);
        const c = i % 32;
        if (r < r_min) r_min = r;
        if (r > r_max) r_max = r;
        if (c < c_min) c_min = c;
        if (c > c_max) c_max = c;
      }
    }
    if (r_max === -1) {
      return [0, 31, 0, 31];
    }
    return [r_min, r_max, c_min, c_max];
  }

  function getInputBounds(input) {
    let r_min = 32, r_max = -1, c_min = 32, c_max = -1;
    for (let i = 0; i < 1024; i++) {
      const in_val = input[i];
      if (in_val < 10) {
        const r = Math.floor(i / 32);
        const c = i % 32;
        if (r < r_min) r_min = r;
        if (r > r_max) r_max = r;
        if (c < c_min) c_min = c;
        if (c > c_max) c_max = c;
      }
    }
    if (r_max === -1) {
      return [0, 31, 0, 31];
    }
    return [r_min, r_max, c_min, c_max];
  }

  function getContentBounds(target, input) {
    const [tr0, tr1, tc0, tc1] = getTargetBounds(target);
    const [ir0, ir1, ic0, ic1] = getInputBounds(input);
    return [
      Math.min(tr0, ir0),
      Math.max(tr1, ir1),
      Math.min(tc0, ic0),
      Math.max(tc1, ic1),
    ];
  }

  const common_inc = `
  ivec2 get_cell_pos(ivec2 I) {
    return I % 32;
  }

  int get_layer(ivec2 I) {
    return (I.y / 32) * 16 + (I.x / 32);
  }

  ivec2 tex_coord(ivec2 p, int l) {
    return ivec2((l % 16) * 32 + p.x, (l / 16) * 32 + p.y);
  }

  vec4 get_cell(sampler2D tex, ivec2 p, int l) {
    return texelFetch(tex, tex_coord(p, l), 0);
  }

  const ivec2 NEIGHBORS[9] = ivec2[9](
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
    ivec2(-1,  0), ivec2(0,  0), ivec2(1,  0),
    ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1)
  );
  `;

  function computeAccuracy() {
    const [r0, r1, c0, c1] = getTargetBounds(target_digits);
    let num_correct = 0;
    let total_cells = 0;
    isExactMatch = true;

    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const idx = r * 32 + c;
        total_cells++;
        if (pred_digits[idx] === target_digits[idx]) {
          num_correct++;
        } else {
          isExactMatch = false;
        }
      }
    }

    return total_cells > 0 ? (num_correct / total_cells) * 100.0 : 0.0;
  }

  function updatePrediction() {
    if (!nca_grid || !pw_tex || !pb_tex) return;

    // Compute argmax prediction per cell on GPU: [32, 32]
    pred_grid_tex = glsl(
      {
        state: nca_grid[0],
        pw_tex: pw_tex,
        pb_tex: pb_tex,
        Inc: [common_inc],
        FP: `
        void fragment() {
          ivec2 p = I;
          
          vec4 o0 = get_cell(state, p, 40);
          vec4 o1 = get_cell(state, p, 41);
          vec4 o2 = get_cell(state, p, 42);
          vec4 o3 = get_cell(state, p, 43);
          vec4 o4 = get_cell(state, p, 44);
          vec4 o5 = get_cell(state, p, 45);
          vec4 o6 = get_cell(state, p, 46);
          vec4 o7 = get_cell(state, p, 47);

          float best_logit = -1e9;
          int best_k = 0;

          for (int k = 0; k < 12; k++) {
            float logit = texelFetch(pb_tex, ivec2(k, 0), 0).r;
            logit += dot(o0, texelFetch(pw_tex, ivec2(0, k), 0));
            logit += dot(o1, texelFetch(pw_tex, ivec2(1, k), 0));
            logit += dot(o2, texelFetch(pw_tex, ivec2(2, k), 0));
            logit += dot(o3, texelFetch(pw_tex, ivec2(3, k), 0));
            logit += dot(o4, texelFetch(pw_tex, ivec2(4, k), 0));
            logit += dot(o5, texelFetch(pw_tex, ivec2(5, k), 0));
            logit += dot(o6, texelFetch(pw_tex, ivec2(6, k), 0));
            logit += dot(o7, texelFetch(pw_tex, ivec2(7, k), 0));

            if (logit > best_logit) {
              best_logit = logit;
              best_k = k;
            }
          }

          FOut = vec4(float(best_k), 0.0, 0.0, 1.0);
        }
        `,
      },
      { size: [W, H], format: "rgba32f", tag: "pred_grid" },
    );

    // Read back 32x32 values to CPU for instant GUI metrics
    if (pred_grid_tex && typeof pred_grid_tex.readSync === "function") {
      try {
        const buf = pred_grid_tex.readSync();
        for (let i = 0; i < 1024; i++) {
          pred_digits[i] = Math.round(buf[i * 4]);
        }
      } catch (e) {
        console.warn("readSync fallback:", e);
      }
    }
  }

  function updateAccuracyPlot() {
    const acc = computeAccuracy();
    if (plot_history.length > 0 && plot_history[plot_history.length - 1][0] === step_count) {
      plot_history[plot_history.length - 1][1] = acc;
    } else {
      plot_history.push([step_count, acc]);
    }
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

    let maxT = MAX_STEPS;
    let maxY = 100;
    for (const [t, y] of plot_history) {
      if (t > maxT) maxT = t;
    }

    const mapX = (t) => padL + (t / maxT) * plotW;
    const mapY = (y) => padT + plotH - (y / maxY) * plotH;

    // Grid lines & Y ticks
    ctx.strokeStyle = "#e8eaed";
    ctx.lineWidth = 1;
    ctx.font = "11px 'Noto Sans', sans-serif";
    ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const yTicks = [0, 50];
    for (const yVal of yTicks) {
      const py = mapY(yVal);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.fillText(`${yVal}%`, padL - 6, py);
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
    ctx.fillText("Accuracy (%)", 0, 0);
    ctx.restore();

    // Target horizontal line: y = 100 (target accuracy 100%, dotted line only)
    const targetY = mapY(100);
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
      ctx.fillText(`${lastPt[1].toFixed(1)}%`, Math.min(lx + 5, padL + plotW - 30), ly - 6);
    }

    ctx.restore();
  }

  function reset() {
    step_count = 0;
    stepAccumulator = 0;
    updateHeaderUI();

    const tid = task_ids.indexOf(params.task_id);
    const active_tid = (tid >= 0 ? tid : 0);

    const te = model_raw.token_embeddings;
    const task_emb_offset = active_tid * TASK_EMB_D;
    const task_emb = model_raw.task_embedding.subarray(task_emb_offset, task_emb_offset + TASK_EMB_D);

    // Initialize stateData [512, 256, 4] for 128 layers of 32x32
    const stateData = new Float32Array(512 * 256 * 4);

    for (let r = 0; r < 32; r++) {
      for (let c = 0; c < 32; c++) {
        const cell_idx = r * 32 + c;
        const token = input_digits[cell_idx];
        const token_clamped = (token >= 0 && token < 12) ? token : 11;
        const base_token_offset = token_clamped * INPUT_EMB_D;

        // 1. Immutable token embeddings (channels 0..31, layers 0..7)
        for (let l = 0; l < 8; l++) {
          const dst = ((Math.floor(l / 16) * 32 + r) * 512 + ((l % 16) * 32 + c)) * 4;
          const src = base_token_offset + l * 4;
          stateData[dst + 0] = te[src + 0];
          stateData[dst + 1] = te[src + 1];
          stateData[dst + 2] = te[src + 2];
          stateData[dst + 3] = te[src + 3];
        }

        // 2. Mutable Task embeddings (channels 32..159, layers 8..39)
        for (let l = 8; l < 40; l++) {
          const dst = ((Math.floor(l / 16) * 32 + r) * 512 + ((l % 16) * 32 + c)) * 4;
          const src = (l - 8) * 4;
          stateData[dst + 0] = task_emb[src + 0];
          stateData[dst + 1] = task_emb[src + 1];
          stateData[dst + 2] = task_emb[src + 2];
          stateData[dst + 3] = task_emb[src + 3];
        }

        // 3. Mutable Output embeddings (channels 160..191, layers 40..47, all 0.0)
        for (let l = 40; l < 48; l++) {
          const dst = ((Math.floor(l / 16) * 32 + r) * 512 + ((l % 16) * 32 + c)) * 4;
          stateData[dst + 0] = 0.0;
          stateData[dst + 1] = 0.0;
          stateData[dst + 2] = 0.0;
          stateData[dst + 3] = 0.0;
        }

        // 4. Mutable Hidden channels (channels 192..511, layers 48..127, initialized to Gaussian noise)
        for (let l = 48; l < 128; l++) {
          const dst = ((Math.floor(l / 16) * 32 + r) * 512 + ((l % 16) * 32 + c)) * 4;
          stateData[dst + 0] = randGaussian(0, 0.1);
          stateData[dst + 1] = randGaussian(0, 0.1);
          stateData[dst + 2] = randGaussian(0, 0.1);
          stateData[dst + 3] = randGaussian(0, 0.1);
        }

        // 5. Initial Group Norm on mutable layers 8..127 (30 units of 4 layers = 16 channels)
        for (let u = 0; u < 30; u++) {
          const base_l = 8 + u * 4;
          let sum_sq = 0.0;
          for (let k = 0; k < 4; k++) {
            const l = base_l + k;
            const idx = ((Math.floor(l / 16) * 32 + r) * 512 + ((l % 16) * 32 + c)) * 4;
            sum_sq += stateData[idx + 0] * stateData[idx + 0];
            sum_sq += stateData[idx + 1] * stateData[idx + 1];
            sum_sq += stateData[idx + 2] * stateData[idx + 2];
            sum_sq += stateData[idx + 3] * stateData[idx + 3];
          }
          const norm = Math.sqrt(sum_sq) + 1e-6;
          for (let k = 0; k < 4; k++) {
            const l = base_l + k;
            const idx = ((Math.floor(l / 16) * 32 + r) * 512 + ((l % 16) * 32 + c)) * 4;
            stateData[idx + 0] /= norm;
            stateData[idx + 1] /= norm;
            stateData[idx + 2] /= norm;
            stateData[idx + 3] /= norm;
          }
        }
      }
    }

    // Upload initial state to GPU with double buffering
    nca_grid = glsl({}, { size: [512, 256], format: "rgba32f", data: stateData, story: 2, tag: "state" });
    if (Array.isArray(nca_grid)) {
      nca_grid[0].update([512, 256], stateData);
      nca_grid[1].update([512, 256], stateData);
    }

    // Update target and input textures
    input_grid_tex = glsl({}, { size: [W, H], format: "r32f", data: new Float32Array(input_digits), tag: "input_grid" });
    target_grid_tex = glsl({}, { size: [W, H], format: "r32f", data: new Float32Array(target_digits), tag: "target_grid" });

    updatePrediction();
    renderStaticCanvases();
    plot_history = [[0, computeAccuracy()]];
    drawPlot();
  }

  function step() {
    if (!nca_grid || !v_tex || !wg_tex || !w1_tex || !w2_tex) return;
    if (step_count >= MAX_STEPS) {
      if (params.runModel) {
        params.runModel = false;
        updatePlayPauseUI();
      }
      return;
    }

    try {
      const s_idx = Math.min(step_count, 127);
      const seed_val = Math.random() * 99999.0;
      const fire_r = params.fireRate;

      // Pass 1a: V Projection with Step Embedding [512, 256]
      val_grid = glsl(
        {
          state: nca_grid[0],
          v_tex: v_tex,
          step_emb_tex: step_emb_tex,
          s_idx: s_idx,
          Inc: [common_inc],
          FP: `
          void fragment() {
            ivec2 p = get_cell_pos(I);
            int l = get_layer(I);
            int base_row = 4 * l;

            vec4 val = vec4(0.0);
            for (int c = 0; c < 128; c++) {
              vec4 se = (c >= 48) ? texelFetch(step_emb_tex, ivec2(c - 48, int(s_idx)), 0) : vec4(0.0);
              vec4 xc = get_cell(state, p, c) + se;

              val.x += dot(xc, texelFetch(v_tex, ivec2(c, base_row + 0), 0));
              val.y += dot(xc, texelFetch(v_tex, ivec2(c, base_row + 1), 0));
              val.z += dot(xc, texelFetch(v_tex, ivec2(c, base_row + 2), 0));
              val.w += dot(xc, texelFetch(v_tex, ivec2(c, base_row + 3), 0));
            }
            FOut = val;
          }
          `,
        },
        { size: [512, 256], format: "rgba32f", tag: "val_grid" },
      );

      // Pass 1b: Perception z (Fixed Attention 3x3 Spatial Pool) [512, 256]
      z_grid = glsl(
        {
          val_grid: val_grid,
          fa_tex: fa_tex,
          Inc: [common_inc],
          FP: `
          void fragment() {
            ivec2 p = get_cell_pos(I);
            int l = get_layer(I);
            int m = l / 4;

            vec4 z = vec4(0.0);
            for (int n = 0; n < 9; n++) {
              ivec2 np = p + NEIGHBORS[n];
              if (np.x >= 0 && np.x < 32 && np.y >= 0 && np.y < 32) {
                float score = texelFetch(fa_tex, ivec2(n, m), 0).r;
                z += score * get_cell(val_grid, np, l);
              }
            }
            FOut = z;
          }
          `,
        },
        { size: [512, 256], format: "rgba32f", tag: "z_grid" },
      );

      // Pass 2: SwiGLU Hidden Layer [1024, 512] (512 tiles of 32x32 = 2048 units)
      h_grid = glsl(
        {
          z_grid: z_grid,
          wg_tex: wg_tex,
          w1_tex: w1_tex,
          bg_tex: bg_tex,
          b1_tex: b1_tex,
          Inc: [common_inc],
          FP: `
          void fragment() {
            ivec2 p = I % 32;
            int u = (I.y / 32) * 32 + (I.x / 32);
            int base_row = 4 * u;

            vec4 dot_g = texelFetch(bg_tex, ivec2(u, 0), 0);
            vec4 dot_v = texelFetch(b1_tex, ivec2(u, 0), 0);

            for (int c = 0; c < 128; c++) {
              vec4 zc = get_cell(z_grid, p, c);

              vec4 wgc0 = texelFetch(wg_tex, ivec2(c, base_row + 0), 0);
              vec4 wgc1 = texelFetch(wg_tex, ivec2(c, base_row + 1), 0);
              vec4 wgc2 = texelFetch(wg_tex, ivec2(c, base_row + 2), 0);
              vec4 wgc3 = texelFetch(wg_tex, ivec2(c, base_row + 3), 0);
              dot_g += vec4(dot(zc, wgc0), dot(zc, wgc1), dot(zc, wgc2), dot(zc, wgc3));

              vec4 w1c0 = texelFetch(w1_tex, ivec2(c, base_row + 0), 0);
              vec4 w1c1 = texelFetch(w1_tex, ivec2(c, base_row + 1), 0);
              vec4 w1c2 = texelFetch(w1_tex, ivec2(c, base_row + 2), 0);
              vec4 w1c3 = texelFetch(w1_tex, ivec2(c, base_row + 3), 0);
              dot_v += vec4(dot(zc, w1c0), dot(zc, w1c1), dot(zc, w1c2), dot(zc, w1c3));
            }

            vec4 swish_g = dot_g / (vec4(1.0) + exp(-clamp(dot_g, -20.0, 20.0)));
            FOut = swish_g * dot_v;
          }
          `,
        },
        { size: [1024, 512], format: "rgba32f", tag: "h_grid" },
      );

      // Pass 3: Output Projection dx & Stochastic Fire Mask [512, 256]
      dx_grid = glsl(
        {
          state: nca_grid[0],
          h_grid: h_grid,
          w2_tex: w2_tex,
          b2_tex: b2_tex,
          step_emb_tex: step_emb_tex,
          s_idx: s_idx,
          seed: seed_val,
          fire_rate: fire_r,
          Inc: [common_inc],
          FP: `
          void fragment() {
            ivec2 p = get_cell_pos(I);
            int l = get_layer(I);

            if (l < 8) {
              // Frozen input embedding channels 0..31
              FOut = get_cell(state, p, l);
              return;
            }

            int base_row = 4 * l;
            vec4 dx = texelFetch(b2_tex, ivec2(l, 0), 0);

            for (int u = 0; u < 512; u++) {
              ivec2 h_coord = ivec2((u % 32) * 32 + p.x, (u / 32) * 32 + p.y);
              vec4 hu = texelFetch(h_grid, h_coord, 0);

              vec4 w2_0 = texelFetch(w2_tex, ivec2(u, base_row + 0), 0);
              vec4 w2_1 = texelFetch(w2_tex, ivec2(u, base_row + 1), 0);
              vec4 w2_2 = texelFetch(w2_tex, ivec2(u, base_row + 2), 0);
              vec4 w2_3 = texelFetch(w2_tex, ivec2(u, base_row + 3), 0);
              dx += vec4(dot(hu, w2_0), dot(hu, w2_1), dot(hu, w2_2), dot(hu, w2_3));
            }

            if (fire_rate < 0.999) {
              if (hash(ivec3(p, int(seed))).x > fire_rate) {
                dx = vec4(0.0);
              }
            }

            vec4 se = (l >= 48) ? texelFetch(step_emb_tex, ivec2(l - 48, int(s_idx)), 0) : vec4(0.0);
            vec4 x_curr = get_cell(state, p, l) + se;
            FOut = x_curr + dx;
          }
          `,
        },
        { size: [512, 256], format: "rgba32f", tag: "dx_grid" },
      );

      // Pass 4: Group Normalization into nca_grid (double buffered) [512, 256]
      glsl(
        {
          dx_grid: dx_grid,
          Inc: [common_inc],
          FP: `
          void fragment() {
            ivec2 p = get_cell_pos(I);
            int l = get_layer(I);

            if (l < 8) {
              FOut = get_cell(dx_grid, p, l);
              return;
            }

            // 30 mutable units, each unit has 4 layers (16 channels)
            int u = (l - 8) / 4;
            int base_l = 8 + u * 4;

            vec4 v0 = get_cell(dx_grid, p, base_l + 0);
            vec4 v1 = get_cell(dx_grid, p, base_l + 1);
            vec4 v2 = get_cell(dx_grid, p, base_l + 2);
            vec4 v3 = get_cell(dx_grid, p, base_l + 3);

            float norm = sqrt(dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3)) + 1e-6;
            vec4 curr = (l == base_l + 0) ? v0 : ((l == base_l + 1) ? v1 : ((l == base_l + 2) ? v2 : v3));
            FOut = curr / norm;
          }
          `,
        },
        nca_grid,
      );

      step_count++;
      updateHeaderUI();
      updatePrediction();
      updateAccuracyPlot();

      if (step_count >= MAX_STEPS && params.runModel) {
        params.runModel = false;
        updatePlayPauseUI();
      }
    } catch (e) {
      console.error("Error executing NCA step:", e);
    }
  }

  function drawMiniGrid(canvas, digitsArr, isInput) {
    if (!canvas || !digitsArr) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bounds = isInput ? getInputBounds(digitsArr) : getTargetBounds(digitsArr);
    const [r0, r1, c0, c1] = bounds;
    const cropW = Math.max(1, c1 - c0 + 1);
    const cropH = Math.max(1, r1 - r0 + 1);
    const maxDim = Math.max(cropW, cropH);

    const size = 38;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(size * dpr) || canvas.height !== Math.round(size * dpr)) {
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(0, 0, size, size);

    const cellPixel = size / maxDim;
    const startX = (size - cropW * cellPixel) / 2.0;
    const startY = (size - cropH * cellPixel) / 2.0;

    for (let r = 0; r < cropH; r++) {
      for (let c = 0; c < cropW; c++) {
        const gridR = r0 + r;
        const gridC = c0 + c;
        const token = digitsArr[gridR * 32 + gridC];
        const color = ARC_PALETTE[token >= 0 && token < 12 ? token : 11];

        const x0 = startX + c * cellPixel;
        const x1 = startX + (c + 1) * cellPixel;
        const y0 = startY + r * cellPixel;
        const y1 = startY + (r + 1) * cellPixel;

        ctx.fillStyle = color;
        ctx.fillRect(x0, y0, x1 - x0 + 0.1, y1 - y0 + 0.1);

        if (cellPixel >= 4.5) {
          ctx.strokeStyle = "rgba(0, 0, 0, 0.25)";
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        }
      }
    }

    ctx.strokeStyle = isInput ? "#0074D9" : "#70757a";
    ctx.lineWidth = 1.0;
    ctx.strokeRect(startX, startY, cropW * cellPixel, cropH * cellPixel);

    ctx.restore();
  }

  function updatePairShelves(task) {
    const trainShelf = $("#train-shelf");
    const testShelf = $("#test-shelf");
    if (!trainShelf || !testShelf || !task) return;

    trainShelf.innerHTML = "";
    testShelf.innerHTML = "";

    const numTrain = task.train_inputs.shape[0];
    const numTest = task.test_inputs.shape[0];

    const train_in = decodeBase64(task.train_inputs.data, task.train_inputs.dtype);
    const train_tgt = decodeBase64(task.train_targets.data, task.train_targets.dtype);
    const test_in = decodeBase64(task.test_inputs.data, task.test_inputs.dtype);
    const test_tgt = decodeBase64(task.test_targets.data, task.test_targets.dtype);

    for (let i = 0; i < numTrain; i++) {
      const card = document.createElement("div");
      card.className = `subcard pair-card ${params.split === "train" && params.sample_idx === i ? "active" : ""}`;
      card.setAttribute("data-split", "train");
      card.setAttribute("data-idx", i);
      card.title = `Train Pair ${i + 1}`;

      const label = document.createElement("div");
      label.className = "pair-label";
      label.textContent = `Pair ${i + 1}`;
      card.appendChild(label);

      const thumbs = document.createElement("div");
      thumbs.className = "pair-thumbs";

      const inCanvas = document.createElement("canvas");
      inCanvas.className = "mini-grid mini-grid-in";
      thumbs.appendChild(inCanvas);

      const arrow = document.createElement("span");
      arrow.className = "pair-arrow";
      arrow.textContent = "→";
      thumbs.appendChild(arrow);

      const outCanvas = document.createElement("canvas");
      outCanvas.className = "mini-grid mini-grid-out";
      thumbs.appendChild(outCanvas);

      card.appendChild(thumbs);
      card.onclick = () => setPair("train", i);
      trainShelf.appendChild(card);

      const inSlice = train_in.subarray(i * 1024, (i + 1) * 1024);
      const outSlice = train_tgt.subarray(i * 1024, (i + 1) * 1024);
      drawMiniGrid(inCanvas, inSlice, true);
      drawMiniGrid(outCanvas, outSlice, false);
    }

    for (let i = 0; i < numTest; i++) {
      const card = document.createElement("div");
      card.className = `subcard pair-card ${params.split === "test" && params.sample_idx === i ? "active" : ""}`;
      card.setAttribute("data-split", "test");
      card.setAttribute("data-idx", i);
      card.title = `Test Pair ${i + 1}`;

      const label = document.createElement("div");
      label.className = "pair-label";
      label.textContent = `Test ${i + 1}`;
      card.appendChild(label);

      const thumbs = document.createElement("div");
      thumbs.className = "pair-thumbs";

      const inCanvas = document.createElement("canvas");
      inCanvas.className = "mini-grid mini-grid-in";
      thumbs.appendChild(inCanvas);

      const arrow = document.createElement("span");
      arrow.className = "pair-arrow";
      arrow.textContent = "→";
      thumbs.appendChild(arrow);

      const outCanvas = document.createElement("canvas");
      outCanvas.className = "mini-grid mini-grid-out";
      thumbs.appendChild(outCanvas);

      card.appendChild(thumbs);
      card.onclick = () => setPair("test", i);
      testShelf.appendChild(card);

      const inSlice = test_in.subarray(i * 1024, (i + 1) * 1024);
      const outSlice = test_tgt.subarray(i * 1024, (i + 1) * 1024);
      drawMiniGrid(inCanvas, inSlice, true);
      drawMiniGrid(outCanvas, outSlice, false);
    }
  }

  function setPair(split, idx) {
    params.split = split;
    params.sample_idx = idx;

    $$(".pair-shelf .subcard").forEach((card) => {
      const cSplit = card.getAttribute("data-split");
      const cIdx = parseInt(card.getAttribute("data-idx"), 10);
      if (cSplit === split && cIdx === idx) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });

    loadCurrentSample();
  }

  function loadCurrentSample() {
    if (!dset_json) return;
    const task = dset_json[params.task_id];
    if (!task) return;

    const isTrain = params.split === "train";
    const inputs_node = isTrain ? task.train_inputs : task.test_inputs;
    const targets_node = isTrain ? task.train_targets : task.test_targets;

    const num_demos = inputs_node.shape[0];
    params.sample_idx = Math.min(Math.max(params.sample_idx, 0), num_demos - 1);

    updatePairShelves(task);

    const in_arr = decodeBase64(inputs_node.data, inputs_node.dtype);
    const tgt_arr = decodeBase64(targets_node.data, targets_node.dtype);

    const slice_start = params.sample_idx * 1024;
    for (let i = 0; i < 1024; i++) {
      input_digits[i] = in_arr[slice_start + i];
      target_digits[i] = tgt_arr[slice_start + i];
    }

    reset();
  }

  function setTask(taskId) {
    if (!dset_json || !dset_json[taskId]) return;
    params.task_id = taskId;
    params.sample_idx = 0;

    const taskSelect = $("#task-select");
    if (taskSelect && taskSelect.value !== taskId) {
      taskSelect.value = taskId;
    }
    loadCurrentSample();
  }

  function randomTask() {
    if (task_ids.length === 0) return;
    const rand_idx = Math.floor(Math.random() * task_ids.length);
    setTask(task_ids[rand_idx]);
  }

  function brush(gx, gy, targetMode) {
    if (gx < 0 || gx >= W || gy < 0 || gy >= H) return;
    const cell_idx = gy * W + gx;
    const val = parseInt(params.brush, 10);
    const color = isNaN(val) ? 0 : val;
    const mode = targetMode || params.brush_mode;

    if (mode === "input") {
      input_digits[cell_idx] = color;
      reset();
    } else if (mode === "target") {
      target_digits[cell_idx] = color;
      target_grid_tex = glsl({}, { size: [W, H], format: "r32f", data: new Float32Array(target_digits), tag: "target_grid" });
      updateAccuracyPlot();
    } else {
      // "output" mode: modify the output channels directly on GPU without resetting
      pred_digits[cell_idx] = color;
      const targetVec = TARGET_OUTPUT_EMBEDDINGS[color];

      // Update layers 40..47 for cell (gx, gy) in nca_grid
      const te_data = new Float32Array(targetVec);
      const te_tex = glsl({}, { size: [8, 1], format: "rgba32f", data: te_data, tag: "out_token_emb" });

      glsl(
        {
          te_tex: te_tex,
          targetCell: [gx, gy],
          Inc: [common_inc],
          FP: `
          void fragment() {
            ivec2 p = get_cell_pos(I);
            int l = get_layer(I);
            if (p == ivec2(targetCell) && l >= 40 && l < 48) {
              int sub_l = l - 40;
              FOut = texelFetch(te_tex, ivec2(sub_l, 0), 0);
            } else {
              FOut = Src(I);
            }
          }
          `,
        },
        nca_grid,
      );

      updatePrediction();
      updateAccuracyPlot();
    }
  }

  function getGridPosFromCanvas(cElem, e) {
    if (!cElem) return [-1, -1];
    const rect = cElem.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const w = cElem.clientWidth;
    const h = cElem.clientHeight;

    const isInput = (cElem === inputCanvas) || (!inputCanvas && params.brush_mode === "input");
    const [r0, r1, c0, c1] = isInput
      ? getInputBounds(input_digits)
      : getTargetBounds(target_digits);
    const cropW = c1 - c0 + 1;
    const cropH = r1 - r0 + 1;
    const maxDim = Math.max(cropW, cropH);

    const size = Math.min(w, h);
    const cellPixel = size / maxDim;

    const startX = (w - cropW * cellPixel) / 2.0;
    const startY = (h - cropH * cellPixel) / 2.0;

    const px = Math.floor((mouseX - startX) / cellPixel);
    const py = Math.floor((mouseY - startY) / cellPixel);

    if (px >= 0 && px < cropW && py >= 0 && py < cropH) {
      const gx = c0 + px;
      const gy = r0 + py;
      return [gx, gy];
    }

    return [-1, -1];
  }

  function getGridPos(e) {
    return getGridPosFromCanvas(canvas, e);
  }

  function setBrush(val) {
    params.brush = val.toString();
    $$('input[name="brush"]').forEach((radio) => {
      radio.checked = radio.value === params.brush;
    });
    updateBrushActiveStyle();
  }

  function updateBrushActiveStyle() {
    $$(".color-btn").forEach((label) => {
      const radio = label.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        label.classList.add("active");
      } else {
        label.classList.remove("active");
      }
    });
  }

  function updateBrushModeActiveStyle() {
    $$(".mode-btn").forEach((label) => {
      const radio = label.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        label.classList.add("active");
      } else {
        label.classList.remove("active");
      }
    });
  }

  // 2D rendering for static Input and Target preview canvases

  function drawGrid2D(cElem, digitsArr, borderColor = "#bdc1c6") {
    if (!cElem) return;
    const ctx = cElem.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = cElem.getBoundingClientRect();
    const w = rect.width > 0 ? rect.width : (cElem.width || 200);
    const h = rect.height > 0 ? rect.height : (cElem.height || 200);

    if (cElem.width !== Math.round(w * dpr) || cElem.height !== Math.round(h * dpr)) {
      cElem.width = Math.round(w * dpr);
      cElem.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const [r0, r1, c0, c1] = (cElem === inputCanvas)
      ? getInputBounds(digitsArr)
      : getTargetBounds(digitsArr);
    const cropW = c1 - c0 + 1;
    const cropH = r1 - r0 + 1;
    const maxDim = Math.max(cropW, cropH);

    const size = Math.min(w, h);
    const cellPixel = size / maxDim;

    const startX = (w - cropW * cellPixel) / 2.0;
    const startY = (h - cropH * cellPixel) / 2.0;

    // Fill background (blank)
    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(0, 0, w, h);

    for (let r = 0; r < cropH; r++) {
      for (let c = 0; c < cropW; c++) {
        const gridR = r0 + r;
        const gridC = c0 + c;
        const token = digitsArr[gridR * 32 + gridC];
        const color = ARC_PALETTE[token >= 0 && token < 12 ? token : 11];

        const x0 = startX + c * cellPixel;
        const x1 = startX + (c + 1) * cellPixel;
        const y0 = startY + r * cellPixel;
        const y1 = startY + (r + 1) * cellPixel;

        ctx.fillStyle = color;
        ctx.fillRect(x0, y0, x1 - x0 + 0.1, y1 - y0 + 0.1);

        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
    }

    // Outer border (thick, prominent 4px stroke)
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 4.0;
    const bHalf = 2.0;
    const bx = startX > 0 ? startX + bHalf : bHalf;
    const by = startY > 0 ? startY + bHalf : bHalf;
    const bw = startX > 0 ? cropW * cellPixel - 2 * bHalf : w - 2 * bHalf;
    const bh = startY > 0 ? cropH * cellPixel - 2 * bHalf : h - 2 * bHalf;
    ctx.strokeRect(bx, by, bw, bh);

    ctx.restore();
  }

  function updateActiveCardThumb() {
    const activeCard = $(`#${params.split}-shelf .subcard.active`);
    if (!activeCard) return;
    const inCanvas = activeCard.querySelector(".mini-grid-in");
    const outCanvas = activeCard.querySelector(".mini-grid-out");
    if (inCanvas) drawMiniGrid(inCanvas, input_digits, true);
    if (outCanvas) drawMiniGrid(outCanvas, target_digits, false);
  }

  function renderStaticCanvases() {
    if (inputCanvas) {
      drawGrid2D(inputCanvas, input_digits, "#0074D9");
    }
    if (targetCanvas) {
      drawGrid2D(targetCanvas, target_digits, "#70757a");
    }
    updateActiveCardThumb();
  }

  function init_event_listeners() {
    let prevPos = [-1, -1];

    // Main demo-canvas listeners
    canvas.onmousedown = (e) => {
      e.preventDefault();
      if (e.buttons === 1) {
        // If inputCanvas exists, demo-canvas clicking acts directly as output mode (or brush_mode)
        const targetMode = inputCanvas ? "output" : params.brush_mode;
        const pos = getGridPos(e);
        if (pos[0] >= 0) {
          brush(pos[0], pos[1], targetMode);
          prevPos = pos;
        }
      }
    };

    canvas.onmousemove = (e) => {
      e.preventDefault();
      if (e.buttons === 1) {
        const targetMode = inputCanvas ? "output" : params.brush_mode;
        const pos = getGridPos(e);
        if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
          brush(pos[0], pos[1], targetMode);
          prevPos = pos;
        }
      }
    };

    canvas.onmouseup = () => {
      prevPos = [-1, -1];
    };

    canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const targetMode = inputCanvas ? "output" : params.brush_mode;
      const t = e.changedTouches[0];
      const pos = getGridPos(t);
      if (pos[0] >= 0) {
        brush(pos[0], pos[1], targetMode);
        prevPos = pos;
      }
    });

    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const targetMode = inputCanvas ? "output" : params.brush_mode;
      const t = e.changedTouches[0];
      const pos = getGridPos(t);
      if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
        brush(pos[0], pos[1], targetMode);
        prevPos = pos;
      }
    });

    canvas.addEventListener("touchend", () => {
      prevPos = [-1, -1];
    });

    // Optional input-canvas listeners (for 3-column horizontal layout)
    if (inputCanvas) {
      inputCanvas.onmousedown = (e) => {
        e.preventDefault();
        if (e.buttons === 1) {
          const pos = getGridPosFromCanvas(inputCanvas, e);
          if (pos[0] >= 0) {
            brush(pos[0], pos[1], "input");
            renderStaticCanvases();
            prevPos = pos;
          }
        }
      };
      inputCanvas.onmousemove = (e) => {
        e.preventDefault();
        if (e.buttons === 1) {
          const pos = getGridPosFromCanvas(inputCanvas, e);
          if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
            brush(pos[0], pos[1], "input");
            renderStaticCanvases();
            prevPos = pos;
          }
        }
      };
      inputCanvas.onmouseup = () => {
        prevPos = [-1, -1];
      };
      inputCanvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        const t = e.changedTouches[0];
        const pos = getGridPosFromCanvas(inputCanvas, t);
        if (pos[0] >= 0) {
          brush(pos[0], pos[1], "input");
          renderStaticCanvases();
          prevPos = pos;
        }
      });
      inputCanvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        const t = e.changedTouches[0];
        const pos = getGridPosFromCanvas(inputCanvas, t);
        if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
          brush(pos[0], pos[1], "input");
          renderStaticCanvases();
          prevPos = pos;
        }
      });
      inputCanvas.addEventListener("touchend", () => {
        prevPos = [-1, -1];
      });
    }

    // Optional target-canvas listeners
    if (targetCanvas) {
      targetCanvas.onmousedown = (e) => {
        e.preventDefault();
        if (e.buttons === 1) {
          const pos = getGridPosFromCanvas(targetCanvas, e);
          if (pos[0] >= 0) {
            brush(pos[0], pos[1], "target");
            renderStaticCanvases();
            prevPos = pos;
          }
        }
      };
      targetCanvas.onmousemove = (e) => {
        e.preventDefault();
        if (e.buttons === 1) {
          const pos = getGridPosFromCanvas(targetCanvas, e);
          if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
            brush(pos[0], pos[1], "target");
            renderStaticCanvases();
            prevPos = pos;
          }
        }
      };
      targetCanvas.onmouseup = () => {
        prevPos = [-1, -1];
      };
      targetCanvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        const t = e.changedTouches[0];
        const pos = getGridPosFromCanvas(targetCanvas, t);
        if (pos[0] >= 0) {
          brush(pos[0], pos[1], "target");
          renderStaticCanvases();
          prevPos = pos;
        }
      });
      targetCanvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        const t = e.changedTouches[0];
        const pos = getGridPosFromCanvas(targetCanvas, t);
        if (pos[0] >= 0 && (pos[0] !== prevPos[0] || pos[1] !== prevPos[1])) {
          brush(pos[0], pos[1], "target");
          renderStaticCanvases();
          prevPos = pos;
        }
      });
      targetCanvas.addEventListener("touchend", () => {
        prevPos = [-1, -1];
      });
    }

    // Control buttons wiring
    const playPauseBtn = $("#play-pause");
    if (playPauseBtn) {
      playPauseBtn.onclick = () => {
        if (step_count >= MAX_STEPS) {
          reset();
        }
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

    const newBtn = $("#new-task");
    if (newBtn) {
      newBtn.onclick = () => randomTask();
    }

    const taskSelect = $("#task-select");
    if (taskSelect) {
      taskSelect.onchange = (e) => {
        setTask(e.target.value);
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
        params.stepsPerSecond = 20;
        if (speedRange) speedRange.value = 20;
        const speedLabel = $("#speedLabel");
        if (speedLabel) {
          speedLabel.textContent = "20 steps/s";
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
        params.fireRate = 1.0;
        if (fireRateRange) fireRateRange.value = 1.0;
        const fireRateLabel = $("#fireRateLabel");
        if (fireRateLabel) {
          fireRateLabel.textContent = "1.00";
        }
      };
    }

    $$('input[name="brush_mode"]').forEach((radio) => {
      radio.onchange = (e) => {
        if (e.target.checked) {
          params.brush_mode = e.target.value;
          updateBrushModeActiveStyle();
        }
      };
    });

    $$('input[name="brush"]').forEach((radio) => {
      radio.onchange = (e) => {
        if (e.target.checked) {
          params.brush = e.target.value;
          updateBrushActiveStyle();
        }
      };
    });

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
          plotTitle.textContent = isVisible ? "Hide Accuracy" : "Show Accuracy";
        }
        if (isVisible) {
          drawPlot();
        }
      };
    }
  }

  async function init() {
    init_event_listeners();

    // 1. Fetch & decode Reasoning NCA model weights
    const model_res = await fetch(model_path);
    const model_json = await model_res.json();

    model_raw.token_embeddings = decodeBase64(model_json.token_embeddings.data, "float32");
    model_raw.task_embedding = decodeBase64(model_json.task_embedding.data, "float32");

    // V: [128, 512] rgba32f
    v_tex = glsl({}, {
      size: [128, 512],
      format: "rgba32f",
      data: decodeBase64(model_json.sensing.v.data, "float32"),
      tag: "v_mat",
    });

    // Fixed attention: [9, 32] r32f
    fa_tex = glsl({}, {
      size: [9, 32],
      format: "r32f",
      data: decodeBase64(model_json.sensing.fixed_attention.data, "float32"),
      tag: "fa_mat",
    });

    // Wg & W1: [128, 2048] rgba32f
    wg_tex = glsl({}, {
      size: [128, 2048],
      format: "rgba32f",
      data: decodeBase64(model_json.update.wg.data, "float32"),
      tag: "wg_mat",
    });
    w1_tex = glsl({}, {
      size: [128, 2048],
      format: "rgba32f",
      data: decodeBase64(model_json.update.w1.data, "float32"),
      tag: "w1_mat",
    });

    // W2: [512, 512] rgba32f
    w2_tex = glsl({}, {
      size: [512, 512],
      format: "rgba32f",
      data: decodeBase64(model_json.update.w2.data, "float32"),
      tag: "w2_mat",
    });

    // Biases:
    bg_tex = glsl({}, {
      size: [512, 1],
      format: "rgba32f",
      data: decodeBase64(model_json.update.bg.data, "float32"),
      tag: "bg_vec",
    });
    b1_tex = glsl({}, {
      size: [512, 1],
      format: "rgba32f",
      data: decodeBase64(model_json.update.b1.data, "float32"),
      tag: "b1_vec",
    });
    b2_tex = glsl({}, {
      size: [128, 1],
      format: "rgba32f",
      data: decodeBase64(model_json.update.b2.data, "float32"),
      tag: "b2_vec",
    });

    // Step embedding: [80, 128] rgba32f
    step_emb_tex = glsl({}, {
      size: [80, 128],
      format: "rgba32f",
      data: decodeBase64(model_json.step_embedding.data, "float32"),
      tag: "step_emb",
    });

    // Pred head: pw [8, 12] rgba32f, pb [12, 1] r32f
    pw_tex = glsl({}, {
      size: [8, 12],
      format: "rgba32f",
      data: decodeBase64(model_json.pred_head.w.data, "float32"),
      tag: "pw_mat",
    });
    pb_tex = glsl({}, {
      size: [12, 1],
      format: "r32f",
      data: decodeBase64(model_json.pred_head.b.data, "float32"),
      tag: "pb_vec",
    });

    // 2. Fetch ARC-AGI-1 dataset
    const dset_res = await fetch(dset_path);
    dset_json = await dset_res.json();
    task_ids = Object.keys(dset_json).sort();

    // Populate task dropdown
    const taskSelect = $("#task-select");
    if (taskSelect) {
      taskSelect.innerHTML = "";
      for (const tid of task_ids) {
        const opt = document.createElement("option");
        opt.value = tid;
        opt.textContent = `Task ${tid}`;
        taskSelect.appendChild(opt);
      }
      taskSelect.value = params.task_id;
    }

    // 3. Initialize first sample and start loop
    loadCurrentSample();
    requestAnimationFrame(frame);
  }

  function frame(time_ms) {
    glsl.adjustCanvas();

    const time = (time_ms || performance.now()) / 1000.0;
    if (lastTime === null) {
      lastTime = time;
    }
    const dt = Math.min(Math.max(time - lastTime, 0.0), 0.2);
    lastTime = time;

    if (params.runModel && step_count < MAX_STEPS) {
      stepAccumulator += dt * params.stepsPerSecond;
      const numSteps = Math.floor(stepAccumulator);
      for (let i = 0; i < numSteps; i++) {
        if (step_count >= MAX_STEPS) break;
        step();
      }
      stepAccumulator -= numSteps;
    } else {
      stepAccumulator = 0;
    }

    const isInputMode = !inputCanvas && params.brush_mode === "input";
    const [r0, r1, c0, c1] = isInputMode
      ? getInputBounds(input_digits)
      : getTargetBounds(target_digits);
    const cropW = c1 - c0 + 1;
    const cropH = r1 - r0 + 1;
    const maxDim = Math.max(cropW, cropH);

    // Clear canvas (blank)
    glsl({ Clear: [248 / 255, 249 / 255, 250 / 255, 1.0] });

    // Render rollout prediction
    if (pred_grid_tex) {
      const predBorder = isExactMatch ? [0.10, 0.65, 0.25, 1.0] : [0.44, 0.46, 0.48, 1.0];
      glsl({
        grid_tex: pred_grid_tex,
        cropW: cropW,
        cropH: cropH,
        r0: r0,
        c0: c0,
        maxDim: maxDim,
        borderColor: predBorder,
        Aspect: "fit",
        VP: `
          vec2 halfSize = vec2(cropW, cropH) / float(maxDim);
          VPos.xy = XY * halfSize;
        `,
        FP: `
          vec2 p_cell = vec2(UV.x * float(cropW), (1.0 - UV.y) * float(cropH));
          ivec2 cell_id = clamp(ivec2(floor(p_cell)), ivec2(0), ivec2(int(cropW) - 1, int(cropH) - 1));
          ivec2 coord = ivec2(int(c0) + cell_id.x, int(r0) + cell_id.y);
          uint val = uint(round(grid_tex(coord).r));

          vec3 col;
          switch (val) {
            case 0u: col = vec3(0.000, 0.000, 0.000); break; // 0: Black #000000
            case 1u: col = vec3(0.000, 0.455, 0.851); break; // 1: Blue #0074D9
            case 2u: col = vec3(1.000, 0.255, 0.212); break; // 2: Red #FF4136
            case 3u: col = vec3(0.180, 0.800, 0.251); break; // 3: Green #2ECC40
            case 4u: col = vec3(1.000, 0.863, 0.000); break; // 4: Yellow #FFDC00
            case 5u: col = vec3(0.667, 0.667, 0.667); break; // 5: Grey #AAAAAA
            case 6u: col = vec3(0.941, 0.071, 0.745); break; // 6: Magenta #F012BE
            case 7u: col = vec3(1.000, 0.522, 0.106); break; // 7: Orange #FF851B
            case 8u: col = vec3(0.498, 0.859, 1.000); break; // 8: Teal #7FDBFF
            case 9u: col = vec3(0.529, 0.055, 0.145); break; // 9: Maroon #870E25
            case 10u: col = vec3(0.200, 0.200, 0.200); break; // 10: Border token
            case 11u: col = vec3(0.9725, 0.9765, 0.9804); break; // 11: Pad token (#f8f9fa)
            default: col = vec3(0.9725, 0.9765, 0.9804); break;
          }

          // 1-pixel cell gridline
          vec2 cell_fract = fract(p_cell);
          vec2 fw = fwidth(p_cell);
          vec2 dist_to_edge = min(cell_fract, 1.0 - cell_fract);
          float is_gridline = 1.0 - min(step(fw.x * 0.65, dist_to_edge.x), step(fw.y * 0.65, dist_to_edge.y));
          col = mix(col, vec3(0.20, 0.20, 0.20), is_gridline * 0.65);

          // Outer boundary border outline (thick 4px border)
          vec2 border_dist = min(UV, 1.0 - UV) * vec2(float(cropW), float(cropH));
          float is_border = 1.0 - min(step(fw.x * 4.0, border_dist.x), step(fw.y * 4.0, border_dist.y));
          col = mix(col, borderColor.rgb, is_border);

          FOut = vec4(col, 1.0);
        `,
      });
    }

    requestAnimationFrame(frame);
  }

  init();

  return {
    reset,
    step,
    randomTask,
    setTask,
    setPair,
    setBrush,
  };
}

export const createDemo = createDemoArc1;
