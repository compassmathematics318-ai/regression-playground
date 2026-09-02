"use strict";

// ===========================================================================
// DOM references
// ===========================================================================
// Site-level navigation
const homeView = document.querySelector("#homeView");
const quadraticPlayground = document.querySelector("#quadraticPlayground");
const linearChoiceButton = document.querySelector("#linearChoiceButton");
const quadraticChoiceButton = document.querySelector("#quadraticChoiceButton");
const backHomeButton = document.querySelector("#backHomeButton");

const trainModeButton = document.querySelector("#trainModeButton");
const testModeButton = document.querySelector("#testModeButton");
const exportModeButton = document.querySelector("#exportModeButton");
const testReadyDot = document.querySelector("#testReadyDot");
const exportReadyDot = document.querySelector("#exportReadyDot");
const trainView = document.querySelector("#trainView");
const testView = document.querySelector("#testView");
const exportView = document.querySelector("#exportView");

// Train mode
const trainingForm = document.querySelector("#trainingForm");
const trainButton = document.querySelector("#trainButton");
const trainButtonText = document.querySelector("#trainButtonText");
const formMessage = document.querySelector("#formMessage");
const resetButton = document.querySelector("#resetButton");

const trueAInput = document.querySelector("#trueA");
const trueBInput = document.querySelector("#trueB");
const trueCInput = document.querySelector("#trueC");
const initialAInput = document.querySelector("#initialA");
const initialBInput = document.querySelector("#initialB");
const initialCInput = document.querySelector("#initialC");
const randomInitInput = document.querySelector("#randomInit");
const learningRateInput = document.querySelector("#learningRate");
const weightDecayInput = document.querySelector("#weightDecay");
const epochsInput = document.querySelector("#epochs");
const seedInput = document.querySelector("#seed");

const lossFunctionSelect = document.querySelector("#lossFunction");
const lossFunctionHelp = document.querySelector("#lossFunctionHelp");
const smoothL1Options = document.querySelector("#smoothL1Options");
const smoothL1BetaInput = document.querySelector("#smoothL1Beta");
const huberOptions = document.querySelector("#huberOptions");
const huberDeltaInput = document.querySelector("#huberDelta");

const optimizerSelect = document.querySelector("#optimizerSelect");
const optimizerHelp = document.querySelector("#optimizerHelp");
const sgdOptions = document.querySelector("#sgdOptions");
const sgdMomentumInput = document.querySelector("#sgdMomentum");
const sgdNesterovInput = document.querySelector("#sgdNesterov");
const adamOptions = document.querySelector("#adamOptions");
const adamBeta1Input = document.querySelector("#adamBeta1");
const adamBeta2Input = document.querySelector("#adamBeta2");
const adamAmsgradInput = document.querySelector("#adamAmsgrad");
const rmspropOptions = document.querySelector("#rmspropOptions");
const rmspropAlphaInput = document.querySelector("#rmspropAlpha");
const rmspropMomentumInput = document.querySelector("#rmspropMomentum");
const rmspropCenteredInput = document.querySelector("#rmspropCentered");
const adagradOptions = document.querySelector("#adagradOptions");
const adagradLrDecayInput = document.querySelector("#adagradLrDecay");

const fitCanvas = document.querySelector("#fitCanvas");
const lossCanvas = document.querySelector("#lossCanvas");
const fitPlaceholder = document.querySelector("#fitPlaceholder");
const lossPlaceholder = document.querySelector("#lossPlaceholder");

const epochLabel = document.querySelector("#epochLabel");
const statusBadge = document.querySelector("#statusBadge");
const targetAValue = document.querySelector("#targetAValue");
const targetBValue = document.querySelector("#targetBValue");
const targetCValue = document.querySelector("#targetCValue");
const currentAValue = document.querySelector("#currentAValue");
const currentBValue = document.querySelector("#currentBValue");
const currentCValue = document.querySelector("#currentCValue");
const errorAValue = document.querySelector("#errorAValue");
const errorBValue = document.querySelector("#errorBValue");
const errorCValue = document.querySelector("#errorCValue");
const trainLossValue = document.querySelector("#trainLossValue");

const epochSlider = document.querySelector("#epochSlider");
const timelineStart = document.querySelector("#timelineStart");
const timelineEnd = document.querySelector("#timelineEnd");
const firstButton = document.querySelector("#firstButton");
const previousButton = document.querySelector("#previousButton");
const playPauseButton = document.querySelector("#playPauseButton");
const nextButton = document.querySelector("#nextButton");
const lastButton = document.querySelector("#lastButton");
const playPauseIcon = document.querySelector("#playPauseIcon");
const playPauseText = document.querySelector("#playPauseText");
const speedSelect = document.querySelector("#speedSelect");

// Training-ready cue targets
const trainingGraphCard = document.querySelector("#trainView .graph-card");
const trainingStatusCard = document.querySelector("#trainView .status-card");
const optimizationCard = document.querySelector("#trainView .loss-card");
const playbackCard = document.querySelector("#trainView .playback-card");

// Test mode
const noModelNotice = document.querySelector("#noModelNotice");
const goTrainButton = document.querySelector("#goTrainButton");
const testForm = document.querySelector("#testForm");
const clearTestButton = document.querySelector("#clearTestButton");
const testModelA = document.querySelector("#testModelA");
const testModelB = document.querySelector("#testModelB");
const testModelC = document.querySelector("#testModelC");
const testSnapshotHelp = document.querySelector("#testSnapshotHelp");

const manualInputButton = document.querySelector("#manualInputButton");
const randomInputButton = document.querySelector("#randomInputButton");
const manualInputPanel = document.querySelector("#manualInputPanel");
const randomInputPanel = document.querySelector("#randomInputPanel");
const manualXValues = document.querySelector("#manualXValues");
const randomCount = document.querySelector("#randomCount");
const randomXMin = document.querySelector("#randomXMin");
const randomXMax = document.querySelector("#randomXMax");
const testSeed = document.querySelector("#testSeed");
const runTestButton = document.querySelector("#runTestButton");
const runTestButtonText = document.querySelector("#runTestButtonText");
const testMessage = document.querySelector("#testMessage");

const testCanvas = document.querySelector("#testCanvas");
const testPlaceholder = document.querySelector("#testPlaceholder");
const testPlaceholderText = document.querySelector("#testPlaceholderText");
const testCountLabel = document.querySelector("#testCountLabel");
const testStatusBadge = document.querySelector("#testStatusBadge");
const maeValue = document.querySelector("#maeValue");
const rmseValue = document.querySelector("#rmseValue");
const mseValue = document.querySelector("#mseValue");
const maxErrorValue = document.querySelector("#maxErrorValue");

const testTargetA = document.querySelector("#testTargetA");
const testTargetB = document.querySelector("#testTargetB");
const testTargetC = document.querySelector("#testTargetC");
const testLearnedA = document.querySelector("#testLearnedA");
const testLearnedB = document.querySelector("#testLearnedB");
const testLearnedC = document.querySelector("#testLearnedC");
const testErrorA = document.querySelector("#testErrorA");
const testErrorB = document.querySelector("#testErrorB");
const testErrorC = document.querySelector("#testErrorC");

const tableCount = document.querySelector("#tableCount");
const resultsTableBody = document.querySelector("#resultsTableBody");

// Export mode
const noExportNotice = document.querySelector("#noExportNotice");
const goTrainFromExportButton = document.querySelector("#goTrainFromExportButton");
const exportSummary = document.querySelector("#exportSummary");
const exportSummaryStatus = document.querySelector("#exportSummaryStatus");
const exportTargetSummary = document.querySelector("#exportTargetSummary");
const exportInitialSummary = document.querySelector("#exportInitialSummary");
const exportLossSummary = document.querySelector("#exportLossSummary");
const exportOptimizerSummary = document.querySelector("#exportOptimizerSummary");
const exportEpochsSummary = document.querySelector("#exportEpochsSummary");
const exportCodeStatus = document.querySelector("#exportCodeStatus");
const exportCode = document.querySelector("#exportCode");
const exportFilename = document.querySelector("#exportFilename");
const exportLineCount = document.querySelector("#exportLineCount");
const copyCodeButton = document.querySelector("#copyCodeButton");
const downloadCodeButton = document.querySelector("#downloadCodeButton");

// ===========================================================================
// State and colors
// ===========================================================================
const state = {
  activeMode: "train",
  testInputMode: "manual",
  trainResult: null,
  testResult: null,
  testModelSnapshot: null,
  exportCode: null,
  exportFilename: null,
  exportLoading: false,
  frameIndex: 0, // 0 = before training. history[0] is frame 1.
  isPlaying: false,
  timerId: null,
  trainYBounds: null,
};

// ===========================================================================
// TRAINING-READY CUE
// Every successful Train Model run reframes the training workspace and briefly
// highlights the graph, training status, loss history, and playback controls.
// ===========================================================================

let trainingCueRunning = false;

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function nextPaint() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(resolve);
    });
  });
}

async function runTrainingReadyCue() {
  if (trainingCueRunning || !state.trainResult) {
    return;
  }

  const targets = [
    trainingGraphCard,
    trainingStatusCard,
    optimizationCard,
    playbackCard,
  ].filter(Boolean);

  if (targets.length !== 4) {
    return;
  }

  trainingCueRunning = true;

  try {
    // Ensure the new training result, graph, metrics, and enabled playback
    // controls have all reached browser layout before we measure anything.
    await nextPaint();

    const reducedMotion =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // Browser zoom changes the size of the visual viewport in CSS pixels.
    // Measure the four highlighted cards LIVE and treat them as one region,
    // rather than relying on fixed offsets that only work at one zoom level.
    const rects = targets.map((element) => element.getBoundingClientRect());

    const regionTop =
      Math.min(...rects.map((rect) => rect.top)) + window.scrollY;
    const regionBottom =
      Math.max(...rects.map((rect) => rect.bottom)) + window.scrollY;
    const regionHeight = regionBottom - regionTop;

    const viewportHeight =
      window.visualViewport?.height || window.innerHeight;

    // Keep a modest breathing margin, but shrink it on shorter/zoomed-in
    // viewports so we preserve as much useful training UI as possible.
    const viewportMargin = Math.max(
      10,
      Math.min(24, viewportHeight * 0.025),
    );
    const availableHeight = Math.max(
      1,
      viewportHeight - viewportMargin * 2,
    );

    let targetScrollTop;

    if (regionHeight <= availableHeight) {
      // Best case: all four highlighted areas physically fit. Center the
      // complete region so graph + metrics + playback are visible together.
      targetScrollTop =
        regionTop - viewportMargin - (availableHeight - regionHeight) / 2;
    } else {
      // At sufficiently high browser zoom no scroll position can make a region
      // taller than the viewport fully visible. Centering the oversized region
      // gives a balanced compromise instead of sacrificing only the graph or
      // only the playback controls.
      targetScrollTop =
        regionTop - (viewportHeight - regionHeight) / 2;
    }

    const documentHeight = Math.max(
      document.documentElement.scrollHeight,
      document.body.scrollHeight,
    );
    const maxScrollTop = Math.max(0, documentHeight - viewportHeight);

    targetScrollTop = Math.min(
      maxScrollTop,
      Math.max(0, targetScrollTop),
    );

    const needsScroll = Math.abs(window.scrollY - targetScrollTop) > 4;

    if (needsScroll) {
      window.scrollTo({
        top: targetScrollTop,
        behavior: reducedMotion ? "auto" : "smooth",
      });

      if (!reducedMotion) {
        // Follow the real scroll position instead of guessing with a fixed
        // timeout. This keeps the glow synchronized across zoom levels.
        const scrollStartedAt = performance.now();

        while (
          Math.abs(window.scrollY - targetScrollTop) > 10 &&
          performance.now() - scrollStartedAt < 650
        ) {
          await wait(20);
        }
      }
    }

    await nextPaint();

    const animations = targets.map((element, index) => {
      const computed = window.getComputedStyle(element);
      const normalBorder = computed.borderColor;
      const normalShadow =
        computed.boxShadow === "none"
          ? "0 0 0 rgba(0,0,0,0)"
          : computed.boxShadow;

      return element.animate(
        [
          {
            offset: 0,
            borderColor: normalBorder,
            boxShadow: normalShadow,
          },
          {
            offset: 0.24,
            borderColor: "rgba(255, 216, 112, 1)",
            boxShadow:
              "0 0 0 2px rgba(255, 216, 112, 0.24), " +
              "0 0 34px rgba(255, 197, 76, 0.42)",
          },
          {
            offset: 0.5,
            borderColor: normalBorder,
            boxShadow: normalShadow,
          },
          {
            offset: 0.76,
            borderColor: "rgba(255, 216, 112, 1)",
            boxShadow:
              "0 0 0 2px rgba(255, 216, 112, 0.24), " +
              "0 0 34px rgba(255, 197, 76, 0.42)",
          },
          {
            offset: 1,
            borderColor: normalBorder,
            boxShadow: normalShadow,
          },
        ],
        {
          duration: reducedMotion ? 900 : 2400,
          delay: reducedMotion ? 0 : index * 45,
          easing: "ease-in-out",
          fill: "none",
        },
      );
    });

    // Keep the Play button as the final small behavioral hint: the highlighted
    // workspace is ready, and this is how the user starts the epoch replay.
    let playAnimation = null;

    if (playPauseButton && !reducedMotion) {
      playAnimation = playPauseButton.animate(
        [
          { transform: "scale(1)", filter: "brightness(1)" },
          { transform: "scale(1.06)", filter: "brightness(1.18)" },
          { transform: "scale(1)", filter: "brightness(1)" },
        ],
        {
          duration: 760,
          delay: 320,
          iterations: 2,
          easing: "ease-in-out",
        },
      );
    }

    const finished = animations.map((animation) =>
      animation.finished.catch(() => undefined),
    );

    if (playAnimation) {
      finished.push(playAnimation.finished.catch(() => undefined));
    }

    await Promise.all(finished);
  } finally {
    trainingCueRunning = false;
  }
}

const COLORS = {};

function refreshCanvasColors() {
  const css = getComputedStyle(document.documentElement);
  COLORS.grid = css.getPropertyValue("--canvas-grid").trim() || "rgba(148, 163, 184, 0.10)";
  COLORS.axis = css.getPropertyValue("--canvas-axis").trim() || "rgba(148, 163, 184, 0.30)";
  COLORS.residual = css.getPropertyValue("--canvas-residual").trim() || "rgba(203, 213, 225, 0.28)";
  COLORS.text = css.getPropertyValue("--muted").trim() || "#92a4bb";
  COLORS.trueCurve = css.getPropertyValue("--true").trim() || "#8ea7ff";
  COLORS.prediction = css.getPropertyValue("--prediction").trim() || "#f07cff";
  COLORS.train = css.getPropertyValue("--train").trim() || "#56d6a1";
  COLORS.test = css.getPropertyValue("--test").trim() || "#ffbf69";
}

refreshCanvasColors();

window.addEventListener("themechange", () => {
  refreshCanvasColors();

  if (state.activeMode === "train" && state.trainResult) {
    renderCurrentFrame();
  } else if (state.activeMode === "test" && state.testResult) {
    drawTestChart(state.testResult);
  }
});

// ===========================================================================
// Site-level model navigation
// ===========================================================================
quadraticChoiceButton.addEventListener("click", openQuadraticPlayground);
linearChoiceButton.addEventListener("click", () => { window.location.href = "/linear"; });
backHomeButton.addEventListener("click", showModelSelection);

function showModelSelection() {
  stopPlayback();
  homeView.classList.remove("hidden");
  quadraticPlayground.classList.add("hidden");
  document.title = "Regression Playground";
  window.scrollTo(0, 0);
}

function openQuadraticPlayground() {
  homeView.classList.add("hidden");
  quadraticPlayground.classList.remove("hidden");
  document.title = "Quadratic Regression | Regression Playground";

  // Every entrance into the quadratic playground begins on Train Model.
  switchMode("train");
  window.scrollTo(0, 0);
}

// ===========================================================================
// Quadratic Train / Test / Export mode switching
// ===========================================================================
trainModeButton.addEventListener("click", () => switchMode("train"));
testModeButton.addEventListener("click", () => switchMode("test"));
exportModeButton.addEventListener("click", () => switchMode("export"));
goTrainButton.addEventListener("click", () => switchMode("train"));
goTrainFromExportButton.addEventListener("click", () => switchMode("train"));

function switchMode(mode) {
  if (!["train", "test", "export"].includes(mode)) return;

  if (mode !== "train") {
    stopPlayback();
  }

  // Test mode evaluates exactly what the user is currently looking at.
  if (mode === "test" && state.trainResult) {
    captureCurrentModelForTesting();
  }

  state.activeMode = mode;

  const isTrain = mode === "train";
  const isTest = mode === "test";
  const isExport = mode === "export";

  trainView.classList.toggle("hidden", !isTrain);
  testView.classList.toggle("hidden", !isTest);
  exportView.classList.toggle("hidden", !isExport);

  trainModeButton.classList.toggle("active", isTrain);
  testModeButton.classList.toggle("active", isTest);
  exportModeButton.classList.toggle("active", isExport);

  trainModeButton.setAttribute("aria-selected", String(isTrain));
  testModeButton.setAttribute("aria-selected", String(isTest));
  exportModeButton.setAttribute("aria-selected", String(isExport));

  window.requestAnimationFrame(() => {
    if (isTrain && state.trainResult) {
      renderCurrentFrame();
    } else if (isTest) {
      refreshTestAvailability();
      if (state.testResult) drawTestChart(state.testResult);
    } else if (isExport) {
      refreshExportAvailability();
      ensureExportCode();
    }
  });
}

// ===========================================================================
// TRAIN MODE: form and API
// ===========================================================================
trainingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPlayback();

  if (!trainingForm.reportValidity()) return;

  setTrainLoading(true);
  setTrainMessage("Training in PyTorch…", false);
  setTrainStatus("training", "Training");

  try {
    const response = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildTrainPayload()),
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body.detail || `Backend returned HTTP ${response.status}.`);
    }

    validateTrainResult(body);

    state.trainResult = body;
    state.testResult = null; // A newly trained model invalidates old test results.
    state.testModelSnapshot = null;
    state.exportCode = null; // Export must always represent the latest completed run.
    state.exportFilename = null;
    state.frameIndex = 0;
    state.trainYBounds = computeTrainingYBounds(body);

    preparePlaybackControls();
    fitPlaceholder.classList.add("hidden");
    lossPlaceholder.classList.add("hidden");
    renderCurrentFrame();

    testReadyDot.classList.add("visible");
    exportReadyDot.classList.add("visible");
    refreshTestAvailability();
    refreshExportAvailability();
    clearTestResults(false);

    setTrainStatus("complete", "Ready");
    const lossLabel = body.training_configuration?.loss?.label || "selected loss";
    const optimizerLabel = body.training_configuration?.optimizer?.label || "selected optimizer";
    setTrainMessage(
      `Training complete with ${lossLabel} + ${optimizerLabel}. ${body.history.length.toLocaleString()} epochs are ready to replay. Test Model is now ready.`,
      false,
    );

    runTrainingReadyCue();
  } catch (error) {
    console.error(error);
    setTrainStatus("idle", "Error");
    setTrainMessage(
      `${error.message} Make sure the FastAPI server is running and load this page through it.`,
      true,
    );
  } finally {
    setTrainLoading(false);
  }
});

function buildTrainPayload() {
  const randomInitialization = randomInitInput.checked;
  const seedText = seedInput.value.trim();

  const payload = {
    true_a: Number(trueAInput.value),
    true_b: Number(trueBInput.value),
    true_c: Number(trueCInput.value),

    initial_a: randomInitialization ? null : Number(initialAInput.value),
    initial_b: randomInitialization ? null : Number(initialBInput.value),
    initial_c: randomInitialization ? null : Number(initialCInput.value),
    seed: randomInitialization && seedText !== "" ? Number(seedText) : null,

    loss_function: lossFunctionSelect.value,
    loss_beta: Number(smoothL1BetaInput.value),
    loss_delta: Number(huberDeltaInput.value),

    optimizer: optimizerSelect.value,
    learning_rate: Number(learningRateInput.value),
    weight_decay: Number(weightDecayInput.value),

    momentum: Number(sgdMomentumInput.value),
    nesterov: sgdNesterovInput.checked,

    beta1: Number(adamBeta1Input.value),
    beta2: Number(adamBeta2Input.value),
    amsgrad: adamAmsgradInput.checked,

    rmsprop_alpha: Number(rmspropAlphaInput.value),
    rmsprop_momentum: Number(rmspropMomentumInput.value),
    rmsprop_centered: rmspropCenteredInput.checked,

    adagrad_lr_decay: Number(adagradLrDecayInput.value),

    epochs: Number(epochsInput.value),
  };

  if (payload.optimizer === "sgd" && payload.nesterov && payload.momentum <= 0) {
    throw new Error("Nesterov requires SGD momentum greater than 0.");
  }

  return payload;
}

function validateTrainResult(result) {
  if (!result || !result.dataset || !Array.isArray(result.history)) {
    throw new Error("The backend response is missing expected training data.");
  }
  if (!Array.isArray(result.dataset.train_x) || !Array.isArray(result.dataset.train_y)) {
    throw new Error("The backend response is missing the training dataset.");
  }
  if (!result.final_parameters || !result.target_parameters) {
    throw new Error("The backend response is missing model parameters.");
  }
}

function setTrainLoading(isLoading) {
  trainButton.disabled = isLoading;
  trainButton.classList.toggle("loading", isLoading);
  trainButtonText.textContent = isLoading ? "Training…" : "Train model";
}

function setTrainMessage(message, isError) {
  formMessage.textContent = message;
  formMessage.classList.toggle("error", isError);
}

function setTrainStatus(kind, text) {
  statusBadge.className = `status-badge ${kind}`;
  statusBadge.textContent = text;
}

// ===========================================================================
// TRAIN MODE: initialization, loss, optimizer + reset
// ===========================================================================
let savedManualInitialValues = {
  a: initialAInput.value,
  b: initialBInput.value,
  c: initialCInput.value,
};
let savedRandomSeed = "";

const LOSS_HELP = {
  l1: "Mean absolute error (MAE): averages the absolute difference between each prediction and its target value.",
  mse: "Squares each error before averaging, so larger misses receive much more weight.",
  smooth_l1: "Uses squared-error behavior for small errors and becomes more L1-like for larger errors.",
  huber: "Uses squared-error behavior for small errors and linear, absolute-error-style growth for larger errors.",
};

const OPTIMIZER_HELP = {
  sgd: "Plain stochastic gradient descent.",
  adam: "Adaptive first-order optimizer using moving averages of gradients and squared gradients. A common starting learning rate is 0.001.",
  adamw: "Adam with decoupled weight decay. It is especially useful when weight decay is intended as regularization.",
  rmsprop: "A Root Mean Square Propagation optimizer that adapts each parameter's step size based on the recent size of its gradients.",
  adagrad: "Accumulates squared gradients so frequently updated parameters receive progressively smaller effective steps.",
};

randomInitInput.addEventListener("change", updateInitializationInputs);
lossFunctionSelect.addEventListener("change", updateLossOptions);
optimizerSelect.addEventListener("change", updateOptimizerOptions);
sgdMomentumInput.addEventListener("input", updateSgdNesterovAvailability);

function updateInitializationInputs() {
  const useRandom = randomInitInput.checked;
  const initialInputs = [initialAInput, initialBInput, initialCInput];

  if (useRandom) {
    savedManualInitialValues = {
      a: initialAInput.value || savedManualInitialValues.a,
      b: initialBInput.value || savedManualInitialValues.b,
      c: initialCInput.value || savedManualInitialValues.c,
    };

    initialInputs.forEach((input) => {
      input.value = "";
      input.placeholder = "Random";
      input.disabled = true;
    });

    seedInput.disabled = false;
    seedInput.value = savedRandomSeed;
    seedInput.placeholder = "Optional";
  } else {
    if (!seedInput.disabled) {
      savedRandomSeed = seedInput.value;
    }

    initialAInput.disabled = false;
    initialBInput.disabled = false;
    initialCInput.disabled = false;

    initialAInput.placeholder = "";
    initialBInput.placeholder = "";
    initialCInput.placeholder = "";

    initialAInput.value = savedManualInitialValues.a || "0";
    initialBInput.value = savedManualInitialValues.b || "0";
    initialCInput.value = savedManualInitialValues.c || "0";

    seedInput.value = "";
    seedInput.placeholder = "Enable Random to use a seed";
    seedInput.disabled = true;
  }
}

function updateLossOptions() {
  const selected = lossFunctionSelect.value;

  lossFunctionHelp.textContent = LOSS_HELP[selected] || "";
  smoothL1Options.classList.toggle("hidden", selected !== "smooth_l1");
  huberOptions.classList.toggle("hidden", selected !== "huber");
}

function updateOptimizerOptions() {
  const selected = optimizerSelect.value;

  optimizerHelp.textContent = OPTIMIZER_HELP[selected] || "";
  sgdOptions.classList.toggle("hidden", selected !== "sgd");
  adamOptions.classList.toggle("hidden", !["adam", "adamw"].includes(selected));
  rmspropOptions.classList.toggle("hidden", selected !== "rmsprop");
  adagradOptions.classList.toggle("hidden", selected !== "adagrad");

  updateSgdNesterovAvailability();
}

function updateSgdNesterovAvailability() {
  const sgdSelected = optimizerSelect.value === "sgd";
  const hasMomentum = Number(sgdMomentumInput.value) > 0;

  sgdNesterovInput.disabled = !sgdSelected || !hasMomentum;
  if (!hasMomentum) sgdNesterovInput.checked = false;
}

resetButton.addEventListener("click", () => {
  stopPlayback();
  trainingForm.reset();

  savedManualInitialValues = { a: "0", b: "0", c: "0" };
  savedRandomSeed = "";
  updateInitializationInputs();
  updateLossOptions();
  updateOptimizerOptions();

  state.trainResult = null;
  state.testResult = null;
  state.testModelSnapshot = null;
  state.exportCode = null;
  state.exportFilename = null;
  state.exportLoading = false;
  state.frameIndex = 0;
  state.trainYBounds = null;

  clearCanvas(fitCanvas);
  clearCanvas(lossCanvas);
  clearCanvas(testCanvas);
  fitPlaceholder.classList.remove("hidden");
  lossPlaceholder.classList.remove("hidden");
  disablePlaybackControls();
  resetTrainingMetrics();
  setTrainStatus("idle", "Idle");
  setTrainMessage("", false);

  testReadyDot.classList.remove("visible");
  exportReadyDot.classList.remove("visible");
  refreshTestAvailability();
  refreshExportAvailability();
  clearTestResults(false);
});

// ===========================================================================
// TRAIN MODE: playback controls
// ===========================================================================
epochSlider.addEventListener("input", () => {
  stopPlayback();
  goToFrame(Number(epochSlider.value));
});

firstButton.addEventListener("click", () => {
  stopPlayback();
  goToFrame(0);
});

previousButton.addEventListener("click", () => {
  stopPlayback();
  goToFrame(state.frameIndex - 1);
});

nextButton.addEventListener("click", () => {
  stopPlayback();
  goToFrame(state.frameIndex + 1);
});

lastButton.addEventListener("click", () => {
  stopPlayback();
  goToFrame(state.trainResult?.history.length ?? 0);
});

playPauseButton.addEventListener("click", () => {
  if (!state.trainResult) return;
  if (state.isPlaying) stopPlayback();
  else startPlayback();
});

speedSelect.addEventListener("change", () => {
  if (state.isPlaying) {
    stopPlayback();
    startPlayback();
  }
});

function preparePlaybackControls() {
  if (!state.trainResult) return;

  const finalFrame = state.trainResult.history.length;
  epochSlider.min = "0";
  epochSlider.max = String(finalFrame);
  epochSlider.value = "0";
  timelineStart.textContent = "0";
  timelineEnd.textContent = finalFrame.toLocaleString();

  [epochSlider, firstButton, previousButton, playPauseButton, nextButton, lastButton, speedSelect]
    .forEach((element) => { element.disabled = false; });
}

function disablePlaybackControls() {
  epochSlider.min = "0";
  epochSlider.max = "0";
  epochSlider.value = "0";
  timelineStart.textContent = "0";
  timelineEnd.textContent = "0";

  [epochSlider, firstButton, previousButton, playPauseButton, nextButton, lastButton, speedSelect]
    .forEach((element) => { element.disabled = true; });

  playPauseIcon.textContent = "▶";
  playPauseText.textContent = "Play";
}

function goToFrame(index) {
  if (!state.trainResult) return;
  const finalFrame = state.trainResult.history.length;
  state.frameIndex = Math.max(0, Math.min(index, finalFrame));
  renderCurrentFrame();
}

function startPlayback() {
  if (!state.trainResult || state.isPlaying) return;
  if (state.frameIndex >= state.trainResult.history.length) goToFrame(0);

  state.isPlaying = true;
  playPauseIcon.textContent = "Ⅱ";
  playPauseText.textContent = "Pause";
  setTrainStatus("training", "Replaying");
  scheduleNextFrame();
}

function scheduleNextFrame() {
  if (!state.isPlaying) return;

  const intervalMs = Number(speedSelect.value);
  state.timerId = window.setTimeout(() => {
    const finalFrame = state.trainResult.history.length;
    if (state.frameIndex >= finalFrame) {
      stopPlayback();
      setTrainStatus("complete", "Complete");
      return;
    }

    goToFrame(state.frameIndex + 1);
    scheduleNextFrame();
  }, intervalMs);
}

function stopPlayback() {
  if (state.timerId !== null) {
    clearTimeout(state.timerId);
    state.timerId = null;
  }

  state.isPlaying = false;
  playPauseIcon.textContent = "▶";
  playPauseText.textContent = "Play";

  if (state.trainResult) {
    const complete = state.frameIndex === state.trainResult.history.length;
    setTrainStatus(complete ? "complete" : "idle", complete ? "Complete" : "Paused");
  }
}

// ===========================================================================
// TRAIN MODE: rendering
// ===========================================================================
function renderCurrentFrame() {
  if (!state.trainResult) return;

  const beforeTraining = state.frameIndex === 0;
  const snapshot = beforeTraining
    ? {
        epoch: 0,
        ...state.trainResult.initial_parameters,
        train_loss: null,
        predictions: state.trainResult.initial_predictions,
      }
    : state.trainResult.history[state.frameIndex - 1];

  epochLabel.textContent = beforeTraining
    ? "Before training"
    : `Epoch ${snapshot.epoch.toLocaleString()} / ${state.trainResult.history.length.toLocaleString()}`;

  epochSlider.value = String(state.frameIndex);
  updateTrainingMetricPanel(snapshot);
  drawTrainingFitChart(snapshot.predictions);
  drawTrainingLossChart(state.frameIndex);
}

function updateTrainingMetricPanel(snapshot) {
  const target = state.trainResult.target_parameters;

  targetAValue.textContent = formatNumber(target.a);
  targetBValue.textContent = formatNumber(target.b);
  targetCValue.textContent = formatNumber(target.c);

  currentAValue.textContent = formatNumber(snapshot.a);
  currentBValue.textContent = formatNumber(snapshot.b);
  currentCValue.textContent = formatNumber(snapshot.c);

  errorAValue.textContent = formatSignedError(snapshot.a - target.a);
  errorBValue.textContent = formatSignedError(snapshot.b - target.b);
  errorCValue.textContent = formatSignedError(snapshot.c - target.c);

  trainLossValue.textContent = snapshot.train_loss == null ? "—" : formatLoss(snapshot.train_loss);
}

function resetTrainingMetrics() {
  epochLabel.textContent = "Ready";
  [
    targetAValue, targetBValue, targetCValue,
    currentAValue, currentBValue, currentCValue,
    errorAValue, errorBValue, errorCValue, trainLossValue,
  ].forEach((element) => { element.textContent = "—"; });
}

function drawTrainingFitChart(predictions) {
  const { ctx, width, height } = prepareCanvas(fitCanvas);
  const dataset = state.trainResult.dataset;
  const xValues = dataset.train_x;
  const yValues = dataset.train_y;

  if (!xValues.length || !yValues.length) return;

  const padding = { top: 24, right: 22, bottom: 38, left: 58 };
  const plot = getPlotRect(width, height, padding);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const [yMin, yMax] = state.trainYBounds;

  const xScale = (x) => mapRange(x, xMin, xMax, plot.left, plot.right);
  const yScale = (y) => mapRange(y, yMin, yMax, plot.bottom, plot.top);

  drawGridAndAxes(ctx, plot, xMin, xMax, yMin, yMax, xScale, yScale, {
    xLabel: "x",
    yLabel: "y",
  });

  drawScatter(ctx, xValues, yValues, xScale, yScale, COLORS.train, 3.4);
  drawLineSeries(ctx, xValues, yValues, xScale, yScale, COLORS.trueCurve, 2.2, 0.92);
  drawLineSeries(ctx, xValues, predictions, xScale, yScale, COLORS.prediction, 2.6, 1);
}

function computeTrainingYBounds(result) {
  const values = [...result.dataset.train_y, ...result.initial_predictions];
  const history = result.history;
  const stride = Math.max(1, Math.floor(history.length / 300));

  for (let i = 0; i < history.length; i += stride) {
    for (const value of history[i].predictions) {
      if (Number.isFinite(value)) values.push(value);
    }
  }

  return paddedBounds(values, 0.12);
}

function drawTrainingLossChart(frameIndex) {
  const { ctx, width, height } = prepareCanvas(lossCanvas);
  const padding = { top: 20, right: 18, bottom: 36, left: 56 };
  const plot = getPlotRect(width, height, padding);
  const visibleHistory = state.trainResult.history.slice(0, frameIndex);

  if (visibleHistory.length === 0) {
    drawEmptyPlot(ctx, plot, width, height, "Loss appears after epoch 1");
    return;
  }

  const epochs = visibleHistory.map((item) => item.epoch);
  const losses = visibleHistory.map((item) => item.train_loss).filter(Number.isFinite);
  let yMin = Math.min(...losses, 0);
  let yMax = Math.max(...losses);
  if (yMax === yMin) yMax = yMin + 1;
  yMax += (yMax - yMin) * 0.12;

  const maxEpoch = Math.max(1, state.trainResult.history.length);
  const xScale = (epoch) => mapRange(epoch, 0, maxEpoch, plot.left, plot.right);
  const yScale = (loss) => mapRange(loss, yMin, yMax, plot.bottom, plot.top);

  drawGridAndAxes(ctx, plot, 0, maxEpoch, yMin, yMax, xScale, yScale, {
    xLabel: "epoch",
    yLabel: "loss",
    xTicks: 5,
    yTicks: 4,
  });

  drawLineSeries(ctx, epochs, losses, xScale, yScale, COLORS.train, 2.1, 0.96);
  drawPoint(ctx, xScale(epochs.at(-1)), yScale(losses.at(-1)), COLORS.train, 3.3);
}

// ===========================================================================
// TEST MODE: input type and availability
// ===========================================================================
manualInputButton.addEventListener("click", () => setTestInputMode("manual"));
randomInputButton.addEventListener("click", () => setTestInputMode("random"));

function setTestInputMode(mode) {
  state.testInputMode = mode;
  const manual = mode === "manual";
  manualInputButton.classList.toggle("active", manual);
  randomInputButton.classList.toggle("active", !manual);
  manualInputPanel.classList.toggle("hidden", !manual);
  randomInputPanel.classList.toggle("hidden", manual);
  setTestMessage("", false);
}

function getCurrentTrainingSnapshot() {
  if (!state.trainResult) return null;

  if (state.frameIndex === 0) {
    return {
      epoch: 0,
      a: state.trainResult.initial_parameters.a,
      b: state.trainResult.initial_parameters.b,
      c: state.trainResult.initial_parameters.c,
    };
  }

  const historyItem = state.trainResult.history[state.frameIndex - 1];
  return {
    epoch: historyItem.epoch,
    a: historyItem.a,
    b: historyItem.b,
    c: historyItem.c,
  };
}

function snapshotsMatch(left, right) {
  if (!left || !right) return false;
  return left.epoch === right.epoch
    && left.a === right.a
    && left.b === right.b
    && left.c === right.c;
}

function captureCurrentModelForTesting() {
  const displayed = getCurrentTrainingSnapshot();
  if (!displayed) {
    state.testModelSnapshot = null;
    return;
  }

  // If the user selected a different training epoch, any previous test metrics
  // belong to a different model and must not remain on screen.
  if (!snapshotsMatch(state.testModelSnapshot, displayed)) {
    state.testResult = null;
    state.testModelSnapshot = { ...displayed };
    clearTestResults(false);
  } else {
    state.testModelSnapshot = { ...displayed };
  }
}

function refreshTestAvailability() {
  const ready = Boolean(state.trainResult);
  noModelNotice.classList.toggle("hidden", ready);
  testForm.classList.toggle("hidden", !ready);

  if (!ready) {
    testPlaceholder.classList.remove("hidden");
    testPlaceholderText.textContent = "Train a model, then evaluate it on new x-values.";
    resetTestSnapshot();
    return;
  }

  const target = state.trainResult.target_parameters;
  const model = state.testModelSnapshot ?? getCurrentTrainingSnapshot();

  if (!model) return;

  testSnapshotHelp.textContent = model.epoch === 0
    ? "Snapshot from before epoch 1. Testing uses these exact starting A, B, and C values."
    : `Snapshot from epoch ${model.epoch.toLocaleString()} / ${state.trainResult.history.length.toLocaleString()}. Testing uses these exact displayed A, B, and C values.`;

  testModelA.textContent = formatNumber(model.a);
  testModelB.textContent = formatNumber(model.b);
  testModelC.textContent = formatNumber(model.c);

  testTargetA.textContent = formatNumber(target.a);
  testTargetB.textContent = formatNumber(target.b);
  testTargetC.textContent = formatNumber(target.c);
  testLearnedA.textContent = formatNumber(model.a);
  testLearnedB.textContent = formatNumber(model.b);
  testLearnedC.textContent = formatNumber(model.c);
  testErrorA.textContent = formatSignedError(model.a - target.a);
  testErrorB.textContent = formatSignedError(model.b - target.b);
  testErrorC.textContent = formatSignedError(model.c - target.c);

  if (!state.testResult) {
    testPlaceholder.classList.remove("hidden");
    testPlaceholderText.textContent = "Choose new x-values and run a test.";
  }
}

// ===========================================================================
// TEST MODE: form and API
// ===========================================================================
testForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!state.trainResult) {
    setTestMessage("Train a model before running a test.", true);
    return;
  }

  if (!testForm.reportValidity()) return;

  let payload;
  try {
    payload = buildTestPayload();
  } catch (error) {
    setTestMessage(error.message, true);
    return;
  }

  setTestLoading(true);
  setTestMessage("Evaluating the trained model…", false);
  setTestStatus("training", "Testing");

  try {
    const response = await fetch("/api/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body.detail || `Backend returned HTTP ${response.status}.`);
    }

    validateTestResult(body);
    state.testResult = body;

    testPlaceholder.classList.add("hidden");
    renderTestResult(body);
    setTestStatus("complete", "Complete");
    setTestMessage(
      `Test complete on ${body.metrics.count.toLocaleString()} new x-value${body.metrics.count === 1 ? "" : "s"}.`,
      false,
    );
  } catch (error) {
    console.error(error);
    setTestStatus("idle", "Error");
    setTestMessage(error.message, true);
  } finally {
    setTestLoading(false);
  }
});

function buildTestPayload() {
  const target = state.trainResult.target_parameters;
  const model = state.testModelSnapshot;

  if (!model) {
    throw new Error("No training snapshot is selected. Return to Train Model, choose an epoch, then open Test Model again.");
  }

  const payload = {
    true_a: target.a,
    true_b: target.b,
    true_c: target.c,
    model_a: model.a,
    model_b: model.b,
    model_c: model.c,
    model_epoch: model.epoch,
    mode: state.testInputMode,
  };

  if (state.testInputMode === "manual") {
    payload.x_values = parseManualXValues(manualXValues.value);
  } else {
    const count = Number(randomCount.value);
    const xMin = Number(randomXMin.value);
    const xMax = Number(randomXMax.value);
    const seedText = testSeed.value.trim();
    const seed = seedText === "" ? null : Number(seedText);

    if (!Number.isInteger(count) || count < 1 || count > 1000) {
      throw new Error("Random count must be an integer from 1 to 1,000.");
    }
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMax <= xMin) {
      throw new Error("Random maximum x must be greater than minimum x.");
    }
    if (seed !== null && !Number.isInteger(seed)) {
      throw new Error("Random seed must be an integer.");
    }

    payload.random_count = count;
    payload.x_min = xMin;
    payload.x_max = xMax;
    payload.seed = seed;
  }

  return payload;
}

function parseManualXValues(text) {
  const tokens = text.trim().split(/[\s,;]+/).filter(Boolean);
  if (tokens.length === 0) {
    throw new Error("Enter at least one x-value.");
  }
  if (tokens.length > 1000) {
    throw new Error("Please test at most 1,000 x-values at once.");
  }

  const values = tokens.map((token) => Number(token));
  const invalidIndex = values.findIndex((value) => !Number.isFinite(value));
  if (invalidIndex !== -1) {
    throw new Error(`“${tokens[invalidIndex]}” is not a valid finite x-value.`);
  }
  return values;
}

function validateTestResult(result) {
  if (!result || !result.metrics || !result.test_points || !result.curve) {
    throw new Error("The backend response is missing expected test data.");
  }
  if (!Array.isArray(result.rows) || !Array.isArray(result.curve.x)) {
    throw new Error("The backend returned malformed test results.");
  }
}

function setTestLoading(isLoading) {
  runTestButton.disabled = isLoading;
  runTestButton.classList.toggle("loading", isLoading);
  runTestButtonText.textContent = isLoading ? "Testing…" : "Run test";
}

function setTestMessage(message, isError) {
  testMessage.textContent = message;
  testMessage.classList.toggle("error", isError);
}

function setTestStatus(kind, text) {
  testStatusBadge.className = `status-badge ${kind}`;
  testStatusBadge.textContent = text;
}

clearTestButton.addEventListener("click", () => clearTestResults(true));

function clearTestResults(keepMessage = true) {
  state.testResult = null;
  clearCanvas(testCanvas);
  testPlaceholder.classList.remove("hidden");
  testPlaceholderText.textContent = state.trainResult
    ? "Choose new x-values and run a test."
    : "Train a model, then evaluate it on new x-values.";

  testCountLabel.textContent = "Waiting for data";
  [maeValue, rmseValue, mseValue, maxErrorValue].forEach((el) => { el.textContent = "—"; });
  setTestStatus("idle", "Idle");
  tableCount.textContent = "No results yet";
  resultsTableBody.innerHTML = '<tr class="empty-table-row"><td colspan="5">Run a test to inspect predictions.</td></tr>';

  if (keepMessage) setTestMessage("Test results cleared. The trained model is unchanged.", false);
}

function resetTestSnapshot() {
  testSnapshotHelp.textContent = "Testing uses the exact A, B, and C from the training epoch you were viewing when you entered Test Model.";
  [
    testModelA, testModelB, testModelC,
    testTargetA, testTargetB, testTargetC,
    testLearnedA, testLearnedB, testLearnedC,
    testErrorA, testErrorB, testErrorC,
  ].forEach((el) => { el.textContent = "—"; });
}

// ===========================================================================
// TEST MODE: rendering
// ===========================================================================
function renderTestResult(result) {
  drawTestChart(result);

  testCountLabel.textContent = `${result.metrics.count.toLocaleString()} test point${result.metrics.count === 1 ? "" : "s"}`;
  maeValue.textContent = formatLoss(result.metrics.mae);
  rmseValue.textContent = formatLoss(result.metrics.rmse);
  mseValue.textContent = formatLoss(result.metrics.mse);
  maxErrorValue.textContent = formatLoss(result.metrics.max_abs_error);

  renderResultsTable(result.rows);
}

function drawTestChart(result) {
  const { ctx, width, height } = prepareCanvas(testCanvas);
  const curve = result.curve;
  const points = result.test_points;
  if (!curve.x.length) return;

  const padding = { top: 24, right: 22, bottom: 38, left: 58 };
  const plot = getPlotRect(width, height, padding);
  const xMin = Math.min(...curve.x);
  const xMax = Math.max(...curve.x);
  const [yMin, yMax] = paddedBounds(
    [...curve.true_y, ...curve.predicted_y, ...points.true_y, ...points.predicted_y],
    0.12,
  );

  const xScale = (x) => mapRange(x, xMin, xMax, plot.left, plot.right);
  const yScale = (y) => mapRange(y, yMin, yMax, plot.bottom, plot.top);

  drawGridAndAxes(ctx, plot, xMin, xMax, yMin, yMax, xScale, yScale, {
    xLabel: "x",
    yLabel: "y",
  });

  // Thin connectors make the prediction error visible at each tested x.
  drawResidualConnectors(ctx, points.x, points.true_y, points.predicted_y, xScale, yScale);

  drawLineSeries(ctx, curve.x, curve.true_y, xScale, yScale, COLORS.trueCurve, 2.3, 0.95);
  drawLineSeries(ctx, curve.x, curve.predicted_y, xScale, yScale, COLORS.prediction, 2.6, 1);
  drawScatter(ctx, points.x, points.true_y, xScale, yScale, COLORS.test, 3.6);
  drawScatter(ctx, points.x, points.predicted_y, xScale, yScale, COLORS.prediction, 2.7);
}

function drawResidualConnectors(ctx, xs, trueYs, predictedYs, xScale, yScale) {
  ctx.save();
  ctx.strokeStyle = COLORS.residual;
  ctx.lineWidth = 1;

  for (let i = 0; i < xs.length; i += 1) {
    const x = xScale(xs[i]);
    const y1 = yScale(trueYs[i]);
    const y2 = yScale(predictedYs[i]);
    if (![x, y1, y2].every(Number.isFinite)) continue;

    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();
  }
  ctx.restore();
}

function renderResultsTable(rows) {
  resultsTableBody.textContent = "";
  const fragment = document.createDocumentFragment();

  for (const rowData of rows) {
    const row = document.createElement("tr");
    const values = [
      formatTableNumber(rowData.x),
      formatTableNumber(rowData.true_y),
      formatTableNumber(rowData.predicted_y),
      formatSignedError(rowData.error),
      formatTableNumber(rowData.absolute_error),
    ];

    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;

      if (index === 3) {
        if (Math.abs(rowData.error) < 0.0001) cell.classList.add("error-small");
        else if (rowData.error > 0) cell.classList.add("error-positive");
        else cell.classList.add("error-negative");
      }

      row.appendChild(cell);
    });
    fragment.appendChild(row);
  }

  resultsTableBody.appendChild(fragment);
  tableCount.textContent = `${rows.length.toLocaleString()} row${rows.length === 1 ? "" : "s"}`;
}

// ===========================================================================
// EXPORT MODE
// ===========================================================================

function refreshExportAvailability() {
  const hasTrainingRun = Boolean(state.trainResult);

  noExportNotice.classList.toggle("hidden", hasTrainingRun);
  exportSummary.classList.toggle("hidden", !hasTrainingRun);

  if (!hasTrainingRun) {
    exportCode.textContent = "# Train a model first to generate a standalone PyTorch experiment.";
    exportFilename.textContent = "quadratic_regression_experiment.py";
    exportLineCount.textContent = "— lines";
    exportCodeStatus.textContent = "Train a model to generate its PyTorch code.";
    copyCodeButton.disabled = true;
    downloadCodeButton.disabled = true;
    return;
  }

  const result = state.trainResult;
  const target = result.target_parameters;
  const initial = result.initial_parameters;
  const config = result.training_configuration || {};
  const initConfig = config.initialization || {};
  const loss = config.loss || {};
  const optimizer = config.optimizer || {};

  exportTargetSummary.textContent =
    `A=${formatNumber(target.a)}, B=${formatNumber(target.b)}, C=${formatNumber(target.c)}`;

  const initPrefix = initConfig.mode === "random" ? "Random → " : "Manual → ";
  exportInitialSummary.textContent =
    `${initPrefix}A₀=${formatNumber(initial.a)}, B₀=${formatNumber(initial.b)}, C₀=${formatNumber(initial.c)}`;

  exportLossSummary.textContent = formatLossSummary(loss);
  exportOptimizerSummary.textContent = formatOptimizerSummary(optimizer);
  exportEpochsSummary.textContent = Number(config.epochs ?? result.history.length).toLocaleString();

  exportSummaryStatus.className = "status-badge complete";
  exportSummaryStatus.textContent = "Ready";

  if (state.exportCode) {
    exportCode.textContent = state.exportCode;
    exportFilename.textContent = state.exportFilename || "quadratic_regression_experiment.py";
    exportLineCount.textContent = `${countCodeLines(state.exportCode).toLocaleString()} lines`;
    exportCodeStatus.textContent = "Generated from your latest completed training run.";
    copyCodeButton.disabled = false;
    downloadCodeButton.disabled = false;
  } else {
    exportCode.textContent = "# Generating the PyTorch reproduction script…";
    exportFilename.textContent = "quadratic_regression_experiment.py";
    exportLineCount.textContent = "— lines";
    exportCodeStatus.textContent = "Preparing code from your latest completed training run…";
    copyCodeButton.disabled = true;
    downloadCodeButton.disabled = true;
  }
}

async function ensureExportCode() {
  if (!state.trainResult || state.exportCode || state.exportLoading) return;

  state.exportLoading = true;
  refreshExportAvailability();

  try {
    const response = await fetch("/api/export/code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildExportPayload()),
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body.detail || `Backend returned HTTP ${response.status}.`);
    }

    if (typeof body.code !== "string" || !body.code.trim()) {
      throw new Error("The backend returned an empty export script.");
    }

    state.exportCode = body.code;
    state.exportFilename =
      typeof body.filename === "string" && body.filename.trim()
        ? body.filename
        : "quadratic_regression_experiment.py";

    refreshExportAvailability();
  } catch (error) {
    console.error(error);
    exportCodeStatus.textContent = `Could not generate code: ${error.message}`;
    exportCode.textContent =
      "# Export generation failed.\n# Return to Train Model, retrain if needed, then try Export again.";
    exportLineCount.textContent = "— lines";
    copyCodeButton.disabled = true;
    downloadCodeButton.disabled = true;
  } finally {
    state.exportLoading = false;
  }
}

function buildExportPayload() {
  const result = state.trainResult;
  if (!result) {
    throw new Error("Train a model before exporting code.");
  }

  const target = result.target_parameters;
  const initial = result.initial_parameters;
  const config = result.training_configuration || {};
  const initConfig = config.initialization || {};
  const loss = config.loss || {};
  const lossParams = loss.parameters || {};
  const optimizer = config.optimizer || {};
  const optimizerParams = optimizer.parameters || {};

  if (!loss.key || !optimizer.key) {
    throw new Error("The training result is missing loss or optimizer configuration.");
  }

  if (!Array.isArray(result.dataset?.train_x) || result.dataset.train_x.length < 2) {
    throw new Error("The training result is missing the training x-values.");
  }

  return {
    target_a: target.a,
    target_b: target.b,
    target_c: target.c,

    initial_a: initial.a,
    initial_b: initial.b,
    initial_c: initial.c,
    initialization_mode: initConfig.mode === "random" ? "random" : "manual",
    seed: initConfig.seed ?? config.seed ?? null,

    loss_function: loss.key,
    loss_beta: lossParams.beta ?? 1.0,
    loss_delta: lossParams.delta ?? 1.0,

    optimizer: optimizer.key,
    learning_rate: optimizerParams.learning_rate ?? 0.01,
    weight_decay: optimizerParams.weight_decay ?? 0.0,

    momentum: optimizer.key === "sgd" ? (optimizerParams.momentum ?? 0.0) : 0.0,
    nesterov: optimizer.key === "sgd" ? Boolean(optimizerParams.nesterov) : false,

    beta1: ["adam", "adamw"].includes(optimizer.key)
      ? (optimizerParams.beta1 ?? 0.9)
      : 0.9,
    beta2: ["adam", "adamw"].includes(optimizer.key)
      ? (optimizerParams.beta2 ?? 0.999)
      : 0.999,
    amsgrad: ["adam", "adamw"].includes(optimizer.key)
      ? Boolean(optimizerParams.amsgrad)
      : false,

    rmsprop_alpha: optimizer.key === "rmsprop"
      ? (optimizerParams.alpha ?? 0.99)
      : 0.99,
    rmsprop_momentum: optimizer.key === "rmsprop"
      ? (optimizerParams.momentum ?? 0.0)
      : 0.0,
    rmsprop_centered: optimizer.key === "rmsprop"
      ? Boolean(optimizerParams.centered)
      : false,

    adagrad_lr_decay: optimizer.key === "adagrad"
      ? (optimizerParams.lr_decay ?? 0.0)
      : 0.0,

    epochs: config.epochs ?? result.history.length,
    train_x: result.dataset.train_x,
  };
}

function formatLossSummary(loss) {
  const label = loss.label || "Selected loss";
  const params = loss.parameters || {};

  if (loss.key === "smooth_l1") {
    return `${label} · β=${formatNumber(params.beta)}`;
  }
  if (loss.key === "huber") {
    return `${label} · δ=${formatNumber(params.delta)}`;
  }
  return label;
}

function formatOptimizerSummary(optimizer) {
  const label = optimizer.label || "Selected optimizer";
  const params = optimizer.parameters || {};
  const parts = [label];

  if (Number.isFinite(params.learning_rate)) {
    parts.push(`lr=${formatNumber(params.learning_rate)}`);
  }

  if (optimizer.key === "sgd" && Number(params.momentum) !== 0) {
    parts.push(`momentum=${formatNumber(params.momentum)}`);
  }

  if (["adam", "adamw"].includes(optimizer.key)) {
    parts.push(`β₁=${formatNumber(params.beta1)}`);
    parts.push(`β₂=${formatNumber(params.beta2)}`);
  }

  if (optimizer.key === "rmsprop") {
    parts.push(`α=${formatNumber(params.alpha)}`);
  }

  if (optimizer.key === "adagrad" && Number(params.lr_decay) !== 0) {
    parts.push(`lr decay=${formatNumber(params.lr_decay)}`);
  }

  return parts.join(" · ");
}

function countCodeLines(code) {
  return code === "" ? 0 : code.split(/\r?\n/).length;
}

copyCodeButton.addEventListener("click", async () => {
  if (!state.exportCode) return;

  try {
    await navigator.clipboard.writeText(state.exportCode);
    const originalText = copyCodeButton.textContent;
    copyCodeButton.textContent = "Copied!";
    window.setTimeout(() => {
      copyCodeButton.textContent = originalText;
    }, 1200);
  } catch (error) {
    console.error(error);
    exportCodeStatus.textContent =
      "Clipboard access was blocked. You can still select and copy the code manually.";
  }
});

downloadCodeButton.addEventListener("click", () => {
  if (!state.exportCode) return;

  const blob = new Blob([state.exportCode], {
    type: "text/x-python;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = state.exportFilename || "quadratic_regression_experiment.py";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  window.setTimeout(() => URL.revokeObjectURL(url), 0);
});

// ===========================================================================
// Canvas helpers
// ===========================================================================
function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);

  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getPlotRect(width, height, padding) {
  return {
    left: padding.left,
    right: width - padding.right,
    top: padding.top,
    bottom: height - padding.bottom,
  };
}

function drawGridAndAxes(ctx, plot, xMin, xMax, yMin, yMax, xScale, yScale, options = {}) {
  const xTicks = options.xTicks ?? 6;
  const yTicks = options.yTicks ?? 5;

  ctx.save();
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = COLORS.text;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;

  for (let i = 0; i <= xTicks; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / xTicks;
    const x = xScale(value);

    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, plot.bottom);
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(formatAxisNumber(value), x, plot.bottom + 9);
  }

  for (let i = 0; i <= yTicks; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / yTicks;
    const y = yScale(value);

    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(formatAxisNumber(value), plot.left - 9, y);
  }

  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath();
  ctx.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
  ctx.stroke();

  ctx.fillStyle = COLORS.text;
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(options.xLabel ?? "x", plot.right, plot.bottom + 33);

  ctx.save();
  ctx.translate(15, plot.top);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(options.yLabel ?? "y", 0, 0);
  ctx.restore();
  ctx.restore();
}

function drawScatter(ctx, xs, ys, xScale, yScale, color, radius) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.95;

  for (let i = 0; i < xs.length; i += 1) {
    const x = xScale(xs[i]);
    const y = yScale(ys[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawLineSeries(ctx, xs, ys, xScale, yScale, color, lineWidth, alpha) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();

  let started = false;
  for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
    const x = xScale(xs[i]);
    const y = yScale(ys[i]);

    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      started = false;
      continue;
    }

    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function drawPoint(ctx, x, y, color, radius) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawEmptyPlot(ctx, plot, width, height, message) {
  ctx.save();
  ctx.strokeStyle = COLORS.grid;
  ctx.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
  ctx.fillStyle = COLORS.text;
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, width / 2, height / 2);
  ctx.restore();
}

function mapRange(value, inMin, inMax, outMin, outMax) {
  if (inMax === inMin) return (outMin + outMax) / 2;
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

function paddedBounds(values, paddingFraction = 0.12) {
  const finiteValues = values.filter(Number.isFinite);
  if (!finiteValues.length) return [-1, 1];

  let min = Math.min(...finiteValues);
  let max = Math.max(...finiteValues);
  if (min === max) {
    const pad = Math.max(1, Math.abs(min) * 0.2);
    min -= pad;
    max += pad;
    return [min, max];
  }

  const span = max - min;
  return [min - span * paddingFraction, max + span * paddingFraction];
}

// ===========================================================================
// Formatting helpers
// ===========================================================================
function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if ((magnitude !== 0 && magnitude < 0.0001) || magnitude >= 10000) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}

function formatTableNumber(value) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if ((magnitude !== 0 && magnitude < 0.00001) || magnitude >= 100000) {
    return value.toExponential(4);
  }
  return value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
}

function formatSignedError(value) {
  if (!Number.isFinite(value)) return "—";
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatNumber(value)}`;
}

function formatLoss(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) < 0.0001 && value !== 0) return value.toExponential(3);
  return value.toFixed(5);
}

function formatAxisNumber(value) {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.01) || abs >= 1000) return value.toExponential(1);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/\.00$/, "");
}

// ===========================================================================
// Resize redraw
// ===========================================================================
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    if (state.activeMode === "train" && state.trainResult) {
      renderCurrentFrame();
    } else if (state.activeMode === "test" && state.testResult) {
      drawTestChart(state.testResult);
    }
  }, 80);
});

// Initial UI setup.
updateInitializationInputs();
updateLossOptions();
updateOptimizerOptions();
disablePlaybackControls();
resetTrainingMetrics();
refreshTestAvailability();
refreshExportAvailability();
clearTestResults(false);
setTestInputMode("manual");
