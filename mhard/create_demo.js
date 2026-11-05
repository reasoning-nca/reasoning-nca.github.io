export function createDemoMhard(glsl, divId, onCanvasRendered = null) {
  const root = document.getElementById(divId);
  const $ = (q) => root.querySelector(q);
  const $$ = (q) => root.querySelectorAll(q);

  const canvas = $("#demo-canvas");

  const CHN = 32; // Number of channels
  const C4 = 8;   // 32 / 4 = 8 RGBA layers
  let H, W, pad = 0; // Grid size and external wall padding
  const dset_path = "../assets/json/mharddata.json"; // Maze dataset path
  let dsets;
  const model_path = "../assets/json/mhardnca.json"; // Reasoning NCA model path
  let raw_models;
  let model;
  let input_grid, nca_grid;
  let input_state = null; // Mirrored Float32Array of cell tokens for move mode
  let draggingCell = null; // Currently dragged cell { token, x, y } in move mode

  let step_count = 0;
  let lastTime = null;
  let stepAccumulator = 0;
  let animation_id = null;

  let plot_history = [];
  let target_path_length = 0;

  const params = {
    trainedOn: "bfs",
    runModel: false,
    fireRate: 0.8,
    brush_mode: "move",
    brush: "wall",
    stepsPerSecond: 30,
  };

  let prevPos = [-1, -1];
  let prevRunMode = params.runModel;

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

  function updateBrushModeActiveStyle() {
    $$("#brush_mode .mode-btn").forEach((label) => {
      const radio = label.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        label.classList.add("active");
      } else {
        label.classList.remove("active");
      }
    });
  }

  function reset_ui() {
    updatePlayPauseUI();
    updateStepCountUI();
    updateBrushModeActiveStyle();
    updateVariantUI();

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

    $$('input[name="brush_mode"]').forEach((radio) => {
      radio.checked = radio.value === params.brush_mode;
    });

    $$('input[name="brush"]').forEach((radio) => {
      radio.checked = radio.value === params.brush;
    });
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
    const innerW = W - 2 * pad;
    const innerH = H - 2 * pad;
    const gx = pad + Math.floor(normX * innerW);
    const gy = pad + Math.floor(normY * innerH);
    return [gx, gy];
  }

  function init_event_listeners() {
    canvas.onmousedown = (e) => {
      e.preventDefault();
      if (e.buttons === 1) {
        const pos = getGridPos(e);
        if (pos[0] < 0) return;

        if (params.brush_mode === "move") {
          const gx = pos[0];
          const gy = pos[1];
          const token = input_state ? input_state[gy * W + gx] : 0.0;
          // Draggable if cell is wall (2), source (3), or target (4)
          if (token === 2.0 || token === 3.0 || token === 4.0) {
            draggingCell = { token, x: gx, y: gy };
            canvas.style.cursor = "grabbing";
            prevRunMode = params.runModel;
            params.runModel = false;
            updatePlayPauseUI();
          }
          return;
        }

        canvas.style.cursor = "crosshair";
        prevRunMode = params.runModel;
        params.runModel = false;
        updatePlayPauseUI();
        brush(pos[0], pos[1]);
        prevPos = pos;
      }
    };

    canvas.onmousemove = (e) => {
      e.preventDefault();
      if (e.buttons === 1) {
        const pos = getGridPos(e);
        if (pos[0] < 0) return;

        if (params.brush_mode === "move") {
          if (draggingCell) {
            const gx = pos[0];
            const gy = pos[1];
            if (gx !== draggingCell.x || gy !== draggingCell.y) {
              const moved = moveCell(draggingCell.x, draggingCell.y, gx, gy, draggingCell.token);
              if (moved) {
                draggingCell.x = gx;
                draggingCell.y = gy;
              }
            }
          }
          return;
        }

        if (pos[0] !== prevPos[0] || pos[1] !== prevPos[1]) {
          brush(pos[0], pos[1]);
          prevPos = pos;
        }
      } else if (params.brush_mode === "move") {
        const pos = getGridPos(e);
        if (pos[0] >= 0 && input_state) {
          const token = input_state[pos[1] * W + pos[0]];
          canvas.style.cursor = (token === 2.0 || token === 3.0 || token === 4.0) ? "grab" : "default";
        } else {
          canvas.style.cursor = "default";
        }
      }
    };

    canvas.onmouseup = (e) => {
      e.preventDefault();
      if (params.brush_mode === "move") {
        if (draggingCell) {
          draggingCell = null;
          params.runModel = prevRunMode;
          updatePlayPauseUI();
        }
        canvas.style.cursor = "grab";
        return;
      }
      canvas.style.cursor = "default";
      params.runModel = prevRunMode;
      updatePlayPauseUI();
      prevPos = [-1, -1];
    };

    window.addEventListener("mouseup", () => {
      if (draggingCell) {
        draggingCell = null;
        params.runModel = prevRunMode;
        updatePlayPauseUI();
        canvas.style.cursor = params.brush_mode === "move" ? "grab" : "default";
      }
    });

    canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const t = e.changedTouches[0];
      const pos = getGridPos(t);
      if (pos[0] < 0) return;

      if (params.brush_mode === "move") {
        const gx = pos[0];
        const gy = pos[1];
        const token = input_state ? input_state[gy * W + gx] : 0.0;
        if (token === 2.0 || token === 3.0 || token === 4.0) {
          draggingCell = { token, x: gx, y: gy };
          prevRunMode = params.runModel;
          params.runModel = false;
          updatePlayPauseUI();
        }
        return;
      }

      prevRunMode = params.runModel;
      params.runModel = false;
      updatePlayPauseUI();
      brush(pos[0], pos[1]);
      prevPos = pos;
    });

    canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      for (const t of e.touches) {
        const pos = getGridPos(t);
        if (pos[0] < 0) continue;

        if (params.brush_mode === "move") {
          if (draggingCell) {
            const gx = pos[0];
            const gy = pos[1];
            if (gx !== draggingCell.x || gy !== draggingCell.y) {
              const moved = moveCell(draggingCell.x, draggingCell.y, gx, gy, draggingCell.token);
              if (moved) {
                draggingCell.x = gx;
                draggingCell.y = gy;
              }
            }
          }
          continue;
        }

        if (pos[0] !== prevPos[0] || pos[1] !== prevPos[1]) {
          brush(pos[0], pos[1]);
          prevPos = pos;
        }
      }
    });

    canvas.addEventListener("touchend", (e) => {
      e.preventDefault();
      if (params.brush_mode === "move") {
        if (draggingCell) {
          draggingCell = null;
          params.runModel = prevRunMode;
          updatePlayPauseUI();
        }
        return;
      }
      params.runModel = prevRunMode;
      updatePlayPauseUI();
      prevPos = [-1, -1];
    });

    // Control buttons wiring
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

    const newMazeBtn = $("#new-maze");
    if (newMazeBtn) {
      newMazeBtn.onclick = () => newMaze();
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
        params.stepsPerSecond = 30;
        if (speedRange) speedRange.value = 30;
        const speedLabel = $("#speedLabel");
        if (speedLabel) {
          speedLabel.textContent = "30 steps/s";
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

    $$("#variant-shelf .subcard").forEach((card) => {
      card.onclick = () => {
        const val = card.getAttribute("data-value") || card.getAttribute("data-variant");
        if (val) setTrainedOn(val);
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
          plotTitle.textContent = isVisible ? "Hide Path Length" : "Show Path Length";
        }
        if (isVisible) {
          drawPlot();
        }
      };
    }

    const trainedOnSelect = $("#trained-on");
    if (trainedOnSelect) {
      trainedOnSelect.onchange = (e) => {
        setTrainedOn(e.target.value);
      };
    }

    $$('input[name="brush_mode"]').forEach((radio) => {
      radio.onchange = (e) => {
        if (e.target.checked) {
          params.brush_mode = e.target.value;
          canvas.style.cursor = params.brush_mode === "move" ? "grab" : "default";
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
  }

  function updateVariantUI() {
    $$("#variant-shelf .subcard").forEach((card) => {
      const val = card.getAttribute("data-value") || card.getAttribute("data-variant");
      if (val === params.trainedOn) {
        card.classList.add("active");
      } else {
        card.classList.remove("active");
      }
    });
    const trainedOnSelect = $("#trained-on");
    if (trainedOnSelect && trainedOnSelect.value !== params.trainedOn) {
      trainedOnSelect.value = params.trainedOn;
    }
  }

  function setTrainedOn(variant) {
    params.trainedOn = variant;
    load_model(variant);
    reset();
    updateVariantUI();
  }

  function setBrush(val) {
    params.brush = val;
    $$('input[name="brush"]').forEach((radio) => {
      radio.checked = radio.value === val;
    });
    updateBrushActiveStyle();
  }

  function updateBrushActiveStyle() {
    $$(".brush-option").forEach((label) => {
      const radio = label.querySelector('input[type="radio"]');
      if (radio && radio.checked) {
        label.classList.add("active");
      } else {
        label.classList.remove("active");
      }
    });
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
      } else if (typeof obj[k] === "object" && obj[k] !== null) {
        decodeBase64Model(obj[k]);
      }
    }
  }

  function load_model(variant) {
    const raw = raw_models[variant];

    // Transpose w2 from (32, 256) to w2t (256, 32)
    const w2 = raw.update.w2.data;
    const w2t_data = new Float32Array(256 * 32);
    for (let c = 0; c < 32; c++) {
      for (let h = 0; h < 256; h++) {
        w2t_data[h * 32 + c] = w2[c * 256 + h];
      }
    }

    model = {
      K: raw.sensing.kernels.data, // Float32Array of 36 weights (4 heads x 9 neighbors)
      input_embed: glsl(
        {},
        {
          size: [8, 5], // width=8 (layers), height=5 (tokens 0..4)
          format: "rgba32f",
          data: raw.input_embed.data,
          tag: "input_embed",
        },
      ),
      w1: glsl(
        {},
        {
          size: [32, 256], // width=128/4=32, height=256
          format: "rgba32f",
          data: raw.update.w1.data,
          tag: "w1",
        },
      ),
      b1: glsl(
        {},
        {
          size: [1, 256],
          format: "r32f",
          data: raw.update.b1.data,
          tag: "b1",
        },
      ),
      w2t: glsl(
        {},
        {
          size: [8, 256], // width=32/4=8, height=256
          format: "rgba32f",
          data: w2t_data,
          tag: "w2t",
        },
      ),
      b2: glsl(
        {},
        {
          size: [8, 1], // width=32/4=8, height=1
          format: "rgba32f",
          data: raw.update.b2.data,
          tag: "b2",
        },
      ),
    };
    return model;
  }

  function newMaze() {
    const dset = dsets.target_grids;
    const raw_H = dset.shape[1];
    const raw_W = dset.shape[2];
    pad = (raw_H === 30 && raw_W === 30) ? 1 : 0;
    H = raw_H + 2 * pad;
    W = raw_W + 2 * pad;

    // Sample random maze from dataset
    const max_idx = dset.shape[0] - 1;
    const maze_idx = Math.floor(Math.random() * (max_idx + 1));
    const maze = dset.data.slice(maze_idx * raw_H * raw_W, (maze_idx + 1) * raw_H * raw_W);

    // Compute target path length from target grid
    target_path_length = 0;
    for (let i = 0; i < maze.length; i++) {
      if (maze[i] === 1) target_path_length++;
    }

    // Remap mharddata.json and pad with one row/col of walls (token 2) on each side
    const start = new Float32Array(H * W);
    start.fill(2.0); // fill border with walls (token 2)

    for (let r = 0; r < raw_H; r++) {
      for (let c = 0; c < raw_W; c++) {
        const val = maze[r * raw_W + c];
        const target_idx = (r + pad) * W + (c + pad);
        if (val === 2) start[target_idx] = 2.0;       // wall
        else if (val === 3) start[target_idx] = 3.0;  // source
        else if (val === 4) start[target_idx] = 4.0;  // target
        else start[target_idx] = 0.0;                 // empty
      }
    }

    input_state = new Float32Array(start);

    input_grid = glsl(
      {},
      {
        size: [W, H],
        format: "r32f",
        data: start,
        tag: "input_grid",
        story: 2,
      },
    );

    reset();
  }

  function reset() {
    step_count = 0;
    stepAccumulator = 0;
    updateStepCountUI();
    plot_history = [[0, 0]];
    drawPlot();

    nca_grid = glsl(
      {
        ...model,
        seed: Math.random() * 26321,
        input_grid: input_grid[0],
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
                // Non-input cell: initialize all 32 channels with Gaussian noise N(0, 0.25^2)
                FOut  = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed)).xy,      hash(ivec3(I, iseed + 1)).xy);
                FOut1 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 2)).xy,  hash(ivec3(I, iseed + 3)).xy);
                FOut2 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 4)).xy,  hash(ivec3(I, iseed + 5)).xy);
                FOut3 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 6)).xy,  hash(ivec3(I, iseed + 7)).xy);
                FOut4 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 8)).xy,  hash(ivec3(I, iseed + 9)).xy);
                FOut5 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 10)).xy, hash(ivec3(I, iseed + 11)).xy);
                FOut6 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 12)).xy, hash(ivec3(I, iseed + 13)).xy);
                FOut7 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 14)).xy, hash(ivec3(I, iseed + 15)).xy);
            } else {
                // Input cells (wall = token 2, source = token 3, target = token 4):
                // Set all 32 channels to input_embed[token]
                FOut  = input_embed(ivec2(0, token));
                FOut1 = input_embed(ivec2(1, token));
                FOut2 = input_embed(ivec2(2, token));
                FOut3 = input_embed(ivec2(3, token));
                FOut4 = input_embed(ivec2(4, token));
                FOut5 = input_embed(ivec2(5, token));
                FOut6 = input_embed(ivec2(6, token));
                FOut7 = input_embed(ivec2(7, token));
            }
        }
        `,
      },
      { size: [W, H], layern: C4, format: "rgba32f", story: 2, tag: "nca_grid" },
    );
  }

  function brush(gx, gy) {
    if (gx < 0 || gx >= W || gy < 0 || gy >= H) {
      return;
    }

    let clicktype = 1;
    if (params.brush === "wall") clicktype = 1;
    else if (params.brush === "empty") clicktype = 2;
    else if (params.brush === "source") clicktype = 3;
    else if (params.brush === "target") clicktype = 4;
    else if (params.brush === "source/target") clicktype = 3;

    // 1. Update input_grid if changing cell type
    if (clicktype >= 1) {
      const newToken = (clicktype === 1) ? 2.0 : ((clicktype === 3) ? 3.0 : ((clicktype === 4) ? 4.0 : 0.0));
      if (input_state) {
        input_state[gy * W + gx] = newToken;
      }
      glsl(
        {
          clickX: gx,
          clickY: gy,
          newToken: newToken,
          FP: `
          void fragment() {
            if (I == ivec2(clickX, clickY)) {
              FOut.r = newToken;
            } else {
              FOut.r = Src(I).r;
            }
          }
          `,
        },
        input_grid,
      );
    }

    // 2. Update nca_grid
    glsl(
      {
        ...model,
        input_grid: input_grid[0],
        clicktype: clicktype,
        clickX: gx,
        clickY: gy,
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
            ivec2 I_click = ivec2(clickX, clickY);
            if (I != I_click) {
                discard;
            }
            int iseed = int(seed);
            int ctype = int(clicktype);

            // wall (token 2)
            if (ctype == 1) {
                FOut  = input_embed(ivec2(0, 2));
                FOut1 = input_embed(ivec2(1, 2));
                FOut2 = input_embed(ivec2(2, 2));
                FOut3 = input_embed(ivec2(3, 2));
                FOut4 = input_embed(ivec2(4, 2));
                FOut5 = input_embed(ivec2(5, 2));
                FOut6 = input_embed(ivec2(6, 2));
                FOut7 = input_embed(ivec2(7, 2));
            }
            // empty (token 0)
            else if (ctype == 2) {
                FOut  = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed)).xy,      hash(ivec3(I, iseed + 1)).xy);
                FOut1 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 2)).xy,  hash(ivec3(I, iseed + 3)).xy);
                FOut2 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 4)).xy,  hash(ivec3(I, iseed + 5)).xy);
                FOut3 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 6)).xy,  hash(ivec3(I, iseed + 7)).xy);
                FOut4 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 8)).xy,  hash(ivec3(I, iseed + 9)).xy);
                FOut5 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 10)).xy, hash(ivec3(I, iseed + 11)).xy);
                FOut6 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 12)).xy, hash(ivec3(I, iseed + 13)).xy);
                FOut7 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 14)).xy, hash(ivec3(I, iseed + 15)).xy);
            }
            // source (token 3)
            else if (ctype == 3) {
                FOut  = input_embed(ivec2(0, 3));
                FOut1 = input_embed(ivec2(1, 3));
                FOut2 = input_embed(ivec2(2, 3));
                FOut3 = input_embed(ivec2(3, 3));
                FOut4 = input_embed(ivec2(4, 3));
                FOut5 = input_embed(ivec2(5, 3));
                FOut6 = input_embed(ivec2(6, 3));
                FOut7 = input_embed(ivec2(7, 3));
            }
            // target (token 4)
            else if (ctype == 4) {
                FOut  = input_embed(ivec2(0, 4));
                FOut1 = input_embed(ivec2(1, 4));
                FOut2 = input_embed(ivec2(2, 4));
                FOut3 = input_embed(ivec2(3, 4));
                FOut4 = input_embed(ivec2(4, 4));
                FOut5 = input_embed(ivec2(5, 4));
                FOut6 = input_embed(ivec2(6, 4));
                FOut7 = input_embed(ivec2(7, 4));
            } else {
                discard;
            }
        }
        `,
      },
      nca_grid[0],
    );
  }

  function moveCell(fromX, fromY, toX, toY, token) {
    if (fromX < 0 || fromX >= W || fromY < 0 || fromY >= H) return false;
    if (toX < 0 || toX >= W || toY < 0 || toY >= H) return false;
    if (fromX === toX && fromY === toY) return false;

    // Disallow moving out-of-bounds or into outer padding border walls
    if (toX < pad || toX >= W - pad || toY < pad || toY >= H - pad) return false;

    // Only allow moving into empty cells (token 0) to avoid deleting other walls/source/target
    if (input_state && input_state[toY * W + toX] !== 0.0) return false;

    if (input_state) {
      input_state[fromY * W + fromX] = 0.0;
      input_state[toY * W + toX] = token;
    }

    // 1. Update input_grid: previous cell becomes empty (0.0), destination cell gets moved token
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
      input_grid,
    );

    // 2. Update nca_grid: previous cell replaced with Gaussian noise, destination gets token embedding
    glsl(
      {
        ...model,
        input_grid: input_grid[0],
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
                // Previous location replaced with empty cell (Gaussian noise)
                FOut  = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed)).xy,      hash(ivec3(I, iseed + 1)).xy);
                FOut1 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 2)).xy,  hash(ivec3(I, iseed + 3)).xy);
                FOut2 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 4)).xy,  hash(ivec3(I, iseed + 5)).xy);
                FOut3 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 6)).xy,  hash(ivec3(I, iseed + 7)).xy);
                FOut4 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 8)).xy,  hash(ivec3(I, iseed + 9)).xy);
                FOut5 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 10)).xy, hash(ivec3(I, iseed + 11)).xy);
                FOut6 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 12)).xy, hash(ivec3(I, iseed + 13)).xy);
                FOut7 = rand_normal(vec4(0.0), 0.25, hash(ivec3(I, iseed + 14)).xy, hash(ivec3(I, iseed + 15)).xy);
            } else if (I == ivec2(toX, toY)) {
                // New location set to token embedding
                int t = int(token);
                FOut  = input_embed(ivec2(0, t));
                FOut1 = input_embed(ivec2(1, t));
                FOut2 = input_embed(ivec2(2, t));
                FOut3 = input_embed(ivec2(3, t));
                FOut4 = input_embed(ivec2(4, t));
                FOut5 = input_embed(ivec2(5, t));
                FOut6 = input_embed(ivec2(6, t));
                FOut7 = input_embed(ivec2(7, t));
            } else {
                discard;
            }
        }
        `,
      },
      nca_grid[0],
    );
    return true;
  }

  function step() {
    const nca = model;
    glsl(
      {
        ...nca,
        input_grid: input_grid[0],
        seed: Math.random() * 26321,
        fire_rate: params.fireRate,
        dt: 1.0,
        W_grid: W,
        H_grid: H,
        FP: `
        const int C4 = 8;
        uniform float K[36];

        vec4 perc[32];

        void fragment() {
            // If input cell (wall or source/target), do not update
            if (input_grid(I).r > 0.5) {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
                FOut4 = Src(I, 4);
                FOut5 = Src(I, 5);
                FOut6 = Src(I, 6);
                FOut7 = Src(I, 7);
                return;
            }

            if (hash(ivec3(I, int(seed))).x > fire_rate) {
                FOut  = Src(I, 0);
                FOut1 = Src(I, 1);
                FOut2 = Src(I, 2);
                FOut3 = Src(I, 3);
                FOut4 = Src(I, 4);
                FOut5 = Src(I, 5);
                FOut6 = Src(I, 6);
                FOut7 = Src(I, 7);
                return;
            }

            // Perception: 4 heads x 9 neighbors convolution
            for (int i = 0; i < 32; ++i) {
                perc[i] = vec4(0.0);
            }

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int n = (dy + 1) * 3 + (dx + 1);
                    ivec2 p = I + ivec2(dx, dy);
                    for (int l = 0; l < 8; ++l) {
                        vec4 v = Src(p, l);
                        for (int m = 0; m < 4; ++m) {
                            perc[m * 8 + l] += K[m * 9 + n] * v;
                        }
                    }
                }
            }

            // MLP Update:
            vec4 dx_out[8];
            for (int l = 0; l < 8; ++l) {
                dx_out[l] = b2(ivec2(l, 0));
            }

            const int ci = 32;  // 128 / 4
            const int ch = 256; // hidden units

            for (int h = 0; h < ch; ++h) {
                float y = b1(ivec2(0, h)).x;
                for (int i = 0; i < ci; ++i) {
                    y += dot(perc[i], w1(ivec2(i, h)));
                }
                if (y <= 0.0) continue; // ReLU
                for (int l = 0; l < 8; ++l) {
                    dx_out[l] += y * w2t(ivec2(l, h));
                }
            }

            // Residual update: x = x + dt * dx
            FOut  = Src(I, 0) + dx_out[0] * dt;
            FOut1 = Src(I, 1) + dx_out[1] * dt;
            FOut2 = Src(I, 2) + dx_out[2] * dt;
            FOut3 = Src(I, 3) + dx_out[3] * dt;
            FOut4 = Src(I, 4) + dx_out[4] * dt;
            FOut5 = Src(I, 5) + dx_out[5] * dt;
            FOut6 = Src(I, 6) + dx_out[6] * dt;
            FOut7 = Src(I, 7) + dx_out[7] * dt;
        }
        `,
      },
      nca_grid,
    );

    step_count++;
    updateStepCountUI();
    updatePathPlot();
  }

  function computeCurrentPathLength() {
    if (!nca_grid || !input_grid) return 0;
    const innerW = W - 2 * pad;
    const innerH = H - 2 * pad;
    glsl(
      {
        state: nca_grid[0],
        input_grid: input_grid[0],
        uv_min: [pad / W, pad / H],
        uv_span: [innerW / W, innerH / H],
        size: [innerW, innerH],
        FP: `
          vec2 uv = uv_min + vec2(UV.x, 1.0 - UV.y) * uv_span;
          int token = int(input_grid(uv).r);
          if (token != 0) {
            FOut = vec4(0.0);
            return;
          }
          vec3 out_emb = state(uv, 0).rgb;
          vec3 path_emb = vec3(0.0, 0.0, 1.0);
          float dist = length(out_emb - path_emb);
          float logit = 1.0 / (1.0 + dist);
          float yellow = clamp(2.0 * logit - 1.0, 0.0, 1.0);
          FOut = vec4(yellow > 0.5 ? 1.0 : 0.0, 0.0, 0.0, 1.0);
        `,
      },
      { size: [innerW, innerH], tag: "path_mask", format: "rgba8" },
    );
    const path_buf = glsl.buffers["path_mask"].readSync();
    let current_len = 0;
    for (let i = 0; i < path_buf.length; i += 4) {
      if (path_buf[i] > 128) current_len++;
    }
    return current_len;
  }

  function updatePathPlot() {
    const cur_len = computeCurrentPathLength();
    plot_history.push([step_count, cur_len]);
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
    let maxY = 10;
    for (const [t, y] of plot_history) {
      if (t > maxT) maxT = t;
      if (y > maxY) maxY = y;
    }
    if (target_path_length > maxY) {
      maxY = target_path_length;
    }
    maxT = Math.ceil(maxT * 1.05);
    maxY = Math.ceil(maxY * 1.15);

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
    ctx.fillText("Path length", 0, 0);
    ctx.restore();

    // Target horizontal line: y = target_path_length
    if (target_path_length > 0) {
      const targetY = mapY(target_path_length);
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#1b5e20"; // dark green
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, targetY);
      ctx.lineTo(padL + plotW, targetY);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = "#1b5e20";
      ctx.font = "11px 'Noto Sans', sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(`${target_path_length}`, padL - 6, targetY);
    }

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

    const innerW = W - 2 * pad;
    const innerH = H - 2 * pad;

    glsl({
      state: nca_grid[0],
      input_grid: input_grid[0],
      uv_min: [pad / W, pad / H],
      uv_span: [innerW / W, innerH / H],
      Aspect: "fit",
      size: [innerW, innerH],
      FP: `
        vec2 uv_inner = vec2(UV.x, 1.0 - UV.y);
        vec2 uv = uv_min + uv_inner * uv_span;
        int token = int(input_grid(uv).r);

        if (token == 2) {
            // Wall: black
            FOut.rgb = vec3(0.0);
            FOut.a = 1.0;
            return;
        }

        if (token == 3) {
            // Source: purple
            FOut.rgb = vec3(0.9, 0.0, 0.9);
            FOut.a = 1.0;
            return;
        }

        if (token == 4) {
            // Target: orange
            FOut.rgb = vec3(1.0, 0.55, 0.0);
            FOut.a = 1.0;
            return;
        }

        // Empty corridor: predict path logits via l2_similarity
        vec3 out_emb = state(uv, 0).rgb;
        vec3 path_emb = vec3(0.0, 0.0, 1.0); // token 1 input_embed[:3]
        float dist = length(out_emb - path_emb);
        float logit = 1.0 / (1.0 + dist);
        float yellow = clamp(2.0 * logit - 1.0, 0.0, 1.0);

        vec3 rgb = vec3(0.5);
        rgb += vec3(0.5, 0.5, -0.5) * yellow;

        FOut.rgb = clamp(rgb, 0.0, 1.0);
        FOut.a = 1.0;
      `,
      Clear: 1.0,
    });

    if (typeof onCanvasRendered === "function") {
      onCanvasRendered();
    }

    animation_id = requestAnimationFrame(frame);
  }

  async function init() {
    init_event_listeners();
    reset_ui();
    updateBrushActiveStyle();

    // 1. Load Maze dataset
    const response = await fetch(dset_path);
    dsets = await response.json();
    const src = dsets.target_grids;
    const byteChars = atob(src.data64);
    const byteNumbers = new Uint8Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) {
      byteNumbers[i] = byteChars.charCodeAt(i);
    }
    src.data = byteNumbers;
    delete src.data64;

    // 2. Load Reasoning NCA model
    const response2 = await fetch(model_path);
    raw_models = await response2.json();
    decodeBase64Model(raw_models);
    load_model(params.trainedOn);

    // 3. Load first maze
    newMaze();

    // 4. Start animation loop
    frame(0);
  }

  init();

  return {
    reset,
    step,
    newMaze,
    setBrush,
    setTrainedOn,
  };
}

export const createDemo = createDemoMhard;
