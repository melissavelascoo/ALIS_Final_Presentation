"use strict";

// La interfaz pública no expone parámetros del clasificador.
// Al abrir ALIS desde FastAPI (puerto 8000), el frontend y la API comparten origen.
const API_BASE = "";

const VOWELS = new Set(["A", "E", "I", "O", "U"]);
const SCAN_INTERVAL_MS = 600;
const PLAYBACK_INTERVAL_MS = 1500;
const SIGN_TRANSITION_MS = 260;

// Las ilustraciones provienen del material visual seleccionado para ALIS.
const SIGN_LIBRARY = {
  A: { type: "image", src: "/assets/img/a.png" },
  E: { type: "image", src: "/assets/img/e.png" },
  I: { type: "image", src: "/assets/img/i.png" },
  O: { type: "image", src: "/assets/img/o.png" },
  U: { type: "image", src: "/assets/img/u.png" },
};

const state = {
  direction: "lsb",
  stream: null,
  cameraStarting: false,
  modelAvailable: false,
  detectedLetter: null,
  analyzing: false,
  scanTimer: null,
  scanController: null,
  scanSession: 0,
  pendingLetter: null,
  consecutiveHits: 0,
  sequence: [],
  sequenceIndex: 0,
  autoplayTimer: null,
  toastTimer: null,
  switching: false,
};

const $ = (selector) => document.querySelector(selector);
const els = {
  switchDirection: $("#switchDirectionBtn"),
  fromLabel: $("#fromLabel"),
  toLabel: $("#toLabel"),
  lsbMode: $("#lsbMode"),
  spanishMode: $("#spanishMode"),
  modeCaption: $("#modeCaption"),

  video: $("#cameraVideo"),
  cameraView: $("#cameraView"),
  cameraPlaceholder: $("#cameraPlaceholder"),
  cameraToggle: $("#cameraToggleBtn"),
  cameraSelect: $("#cameraSelect"),
  cameraBadge: $("#cameraBadge"),
  cameraStatus: $("#cameraStatus"),
  liveTag: $("#liveCameraTag"),
  canvas: $("#captureCanvas"),
  autoRecognitionText: $("#autoRecognitionText"),
  recognizedChar: $("#recognizedChar"),
  recognitionMessage: $("#recognitionMessage"),
  translatedText: $("#translatedText"),
  addChar: $("#addCharBtn"),
  addSpace: $("#addSpaceBtn"),
  deleteChar: $("#deleteCharBtn"),
  clearText: $("#clearTextBtn"),
  copyText: $("#copyTextBtn"),

  sourceText: $("#sourceText"),
  charCount: $("#charCount"),
  processText: $("#processTextBtn"),
  signStage: $("#signStage"),
  signMessage: $("#signMessage"),
  sequenceCounter: $("#sequenceCounter"),
  sequenceStrip: $("#sequenceStrip"),
  previous: $("#prevBtn"),
  next: $("#nextBtn"),
  play: $("#playBtn"),
  toast: $("#toast"),
  translator: $(".translator"),
};

function announce(message) {
  window.clearTimeout(state.toastTimer);
  els.toast.textContent = message;
  els.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => { els.toast.hidden = true; }, 2600);
}

function setDirection(direction) {
  if (!["lsb", "es"].includes(direction)) return;
  if (direction !== "lsb") stopCamera();
  stopAutoplay();
  state.direction = direction;
  const fromLSB = direction === "lsb";
  els.lsbMode.hidden = !fromLSB;
  els.spanishMode.hidden = fromLSB;
  els.fromLabel.textContent = fromLSB ? "Lengua de Señas Boliviana" : "Español";
  els.toLabel.textContent = fromLSB ? "Español" : "Lengua de Señas Boliviana";
  els.modeCaption.textContent = fromLSB ? "De señas a texto" : "De texto a señas";
  els.switchDirection.setAttribute("aria-label", fromLSB
    ? "Cambiar a Español hacia Lengua de Señas Boliviana"
    : "Cambiar a Lengua de Señas Boliviana hacia Español");
}

function switchDirection() {
  if (state.switching) return;
  const next = state.direction === "lsb" ? "es" : "lsb";
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    setDirection(next);
    return;
  }
  state.switching = true;
  els.switchDirection.disabled = true;
  els.translator.classList.remove("is-entering");
  els.translator.classList.add("is-swapping");
  window.setTimeout(() => {
    setDirection(next);
    els.translator.classList.remove("is-swapping");
    els.translator.classList.add("is-entering");
    window.setTimeout(() => {
      els.translator.classList.remove("is-entering");
      state.switching = false;
      els.switchDirection.disabled = false;
    }, 300);
  }, 155);
}

function clearDetection(message = "Presenta una vocal ante la cámara para reconocerla.") {
  state.detectedLetter = null;
  els.recognizedChar.textContent = "—";
  els.recognitionMessage.textContent = message;
  els.addChar.disabled = true;
}

function refreshCameraState() {
  const active = Boolean(state.stream);
  els.cameraView.classList.toggle("camera-active", active);
  els.cameraPlaceholder.hidden = active;
  els.liveTag.hidden = !active;
  els.cameraBadge.textContent = active ? "Encendida" : "Apagada";
  els.cameraBadge.classList.toggle("is-on", active);
  els.cameraBadge.classList.remove("is-error");
  els.cameraToggle.textContent = active ? "Desactivar cámara" : "Activar cámara";
  els.cameraToggle.disabled = state.cameraStarting;
  els.autoRecognitionText.textContent = !active
    ? "El reconocimiento se inicia al activar la cámara."
    : state.modelAvailable
      ? "Reconocimiento automático activo"
      : "Reconocimiento no disponible";
  els.cameraView.classList.toggle("is-recognizing", active && state.modelAvailable);
  if (!active) {
    els.cameraStatus.textContent = "Puedes activar la cámara cuando quieras.";
  } else if (!state.modelAvailable) {
    els.cameraStatus.textContent = "La cámara está lista. El reconocimiento aún no está disponible.";
  } else {
    els.cameraStatus.textContent = "Cámara lista para reconocer una seña.";
  }
}

async function checkAvailability() {
  try {
    const [health, status] = await Promise.all([
      fetch(`${API_BASE}/api/health`, { cache: "no-store" }),
      fetch(`${API_BASE}/api/model-status`, { cache: "no-store" }),
    ]);
    if (!health.ok || !status.ok) throw new Error("Servicio no disponible");
    const model = await status.json();
    state.modelAvailable = model.loaded === true;
  } catch {
    state.modelAvailable = false;
  }
  if (!state.modelAvailable) {
    stopRecognition();
    clearDetection("El reconocimiento no está disponible en este momento.");
  } else if (state.stream && !document.hidden && state.direction === "lsb") {
    scheduleRecognition(0);
  } else if (!state.stream) {
    clearDetection("Activa la cámara para reconocer una vocal.");
  }
  refreshCameraState();
}

// Sin peticiones simultáneas; el siguiente fotograma se envía tras terminar el anterior.
function scheduleRecognition(delay = SCAN_INTERVAL_MS) {
  if (state.scanTimer !== null || state.analyzing || !state.stream ||
      !state.modelAvailable || state.direction !== "lsb" || document.hidden) return;
  state.scanTimer = window.setTimeout(() => {
    state.scanTimer = null;
    void analyzeFrame();
  }, delay);
}

function stopRecognition() {
  state.scanSession += 1;
  window.clearTimeout(state.scanTimer);
  state.scanTimer = null;
  state.scanController?.abort();
  state.scanController = null;
  state.analyzing = false;
  state.pendingLetter = null;
  state.consecutiveHits = 0;
}

async function populateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === "videoinput");
    els.cameraSelect.replaceChildren();
    if (!cameras.length) {
      els.cameraSelect.add(new Option("Predeterminada", ""));
    }
    for (const [index, camera] of cameras.entries()) {
      els.cameraSelect.add(new Option(camera.label || `Cámara ${index + 1}`, camera.deviceId));
    }
    els.cameraSelect.disabled = cameras.length <= 1;
    const settings = state.stream?.getVideoTracks()[0]?.getSettings();
    if (settings?.deviceId && cameras.some((device) => device.deviceId === settings.deviceId)) {
      els.cameraSelect.value = settings.deviceId;
    }
  } catch {
    els.cameraSelect.disabled = true;
  }
}

function stopCamera() {
  stopRecognition();
  if (state.stream) {
    const oldStream = state.stream;
    state.stream = null;
    for (const track of oldStream.getTracks()) track.stop();
  }
  els.video.srcObject = null;
  clearDetection(state.modelAvailable
    ? "Activa la cámara para reconocer una vocal."
    : "El reconocimiento no está disponible en este momento.");
  refreshCameraState();
}

async function startCamera(deviceId = "") {
  if (state.cameraStarting) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    els.cameraStatus.textContent = "La cámara requiere un navegador compatible y una conexión segura.";
    els.cameraBadge.textContent = "No disponible";
    els.cameraBadge.classList.add("is-error");
    return;
  }
  state.cameraStarting = true;
  let cameraError = null;
  stopCamera();
  refreshCameraState();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" },
    });
    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play();
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      if (state.stream === stream) stopCamera();
    });
    await populateCameras();
    if (state.modelAvailable && state.direction === "lsb") scheduleRecognition(0);
    if (!state.modelAvailable) {
      clearDetection("El reconocimiento no está disponible en este momento.");
    }
  } catch (error) {
    stopCamera();
    cameraError = error?.name === "NotAllowedError"
      ? "No se concedió permiso para acceder a la cámara."
      : "No fue posible abrir esta cámara. Revisa los permisos del dispositivo.";
  } finally {
    state.cameraStarting = false;
    refreshCameraState();
    if (cameraError) {
      els.cameraBadge.textContent = "Error";
      els.cameraBadge.classList.add("is-error");
      els.cameraStatus.textContent = cameraError;
    }
  }
}

async function toggleCamera() {
  if (state.cameraStarting) return;
  if (state.stream) stopCamera();
  else await startCamera(els.cameraSelect.value);
}

function applyPrediction(result) {
  const cls = String(result?.class_name ?? "").trim().toUpperCase();
  const serverState = String(result?.state ?? result?.status ?? "");
  if (serverState === "no_hand") {
    state.pendingLetter = null;
    state.consecutiveHits = 0;
    clearDetection("No se detectó una mano.");
    return;
  }
  if (serverState === "no_sign" || cls === "SIN_SENA") {
    state.pendingLetter = null;
    state.consecutiveHits = 0;
    clearDetection("No se detectó una seña reconocible.");
    return;
  }
  if (serverState === "hand_out_of_frame") {
    state.pendingLetter = null;
    state.consecutiveHits = 0;
    clearDetection("Coloca la mano dentro del encuadre.");
    return;
  }
  if (serverState === "uncertain" || serverState === "low_confidence" || result?.accepted === false) {
    state.pendingLetter = null;
    state.consecutiveHits = 0;
    clearDetection("No se pudo identificar la seña.");
    return;
  }
  if ((result?.accepted === true || serverState === "recognized") && VOWELS.has(cls)) {
    if (state.pendingLetter !== cls) {
      state.pendingLetter = cls;
      state.consecutiveHits = 1;
    } else {
      state.consecutiveHits += 1;
    }
    // Se confirman dos lecturas seguidas antes de habilitar «Añadir vocal».
    if (state.consecutiveHits < 2) {
      clearDetection("Identificando vocal…");
      return;
    }
    state.detectedLetter = cls;
    if (els.recognizedChar.textContent !== cls) els.recognizedChar.textContent = cls;
    if (els.recognitionMessage.textContent !== `Vocal ${cls} reconocida. Puedes añadirla al texto.`) {
      els.recognitionMessage.textContent = `Vocal ${cls} reconocida. Puedes añadirla al texto.`;
    }
    els.addChar.disabled = false;
    return;
  }
  state.pendingLetter = null;
  state.consecutiveHits = 0;
  clearDetection("No se recibió un reconocimiento válido.");
}

async function analyzeFrame() {
  if (!state.stream || !state.modelAvailable || state.analyzing ||
      state.direction !== "lsb" || document.hidden) return;
  if (!els.video.videoWidth || !els.video.videoHeight) {
    scheduleRecognition(250);
    return;
  }
  const session = state.scanSession;
  const controller = new AbortController();
  state.scanController = controller;
  state.analyzing = true;
  let nextDelay = SCAN_INTERVAL_MS;
  try {
    const canvas = els.canvas;
    const aspect = els.video.videoWidth / els.video.videoHeight;
    canvas.width = 640;
    canvas.height = Math.max(1, Math.round(640 / aspect));
    // No reflejar: se envía la orientación original del fotograma.
    const context = canvas.getContext("2d");
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.drawImage(els.video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", .85));
    if (session !== state.scanSession || controller.signal.aborted) return;
    if (!blob) throw new Error("No se pudo capturar el fotograma.");
    const data = new FormData();
    data.append("frame", blob, "captura.jpg");
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST", body: data, signal: controller.signal,
    });
    if (session !== state.scanSession || controller.signal.aborted) return;
    if (!response.ok) {
      if (response.status === 503) {
        state.modelAvailable = false;
        throw new Error("El reconocimiento no está disponible en este momento.");
      }
      throw new Error("No fue posible analizar la imagen.");
    }
    const result = await response.json();
    if (session === state.scanSession && !controller.signal.aborted) applyPrediction(result);
  } catch (error) {
    if (controller.signal.aborted || session !== state.scanSession) return;
    state.pendingLetter = null;
    state.consecutiveHits = 0;
    clearDetection(error?.message || "No fue posible reconocer la seña.");
    nextDelay = 1800; // Evita saturar la API si hay un problema de conexión.
  } finally {
    if (session === state.scanSession) {
      state.analyzing = false;
      state.scanController = null;
      refreshCameraState();
      scheduleRecognition(nextDelay);
    }
  }
}

async function copyTranslation() {
  const content = els.translatedText.value;
  if (!content) return announce("Todavía no hay texto para copiar.");
  try {
    await navigator.clipboard.writeText(content);
    announce("Texto copiado al portapapeles.");
  } catch {
    els.translatedText.focus();
    els.translatedText.select();
    if (document.execCommand("copy")) announce("Texto copiado al portapapeles.");
    else announce("No fue posible copiar el texto.");
  }
}

function stopAutoplay() {
  if (state.autoplayTimer !== null) {
    window.clearInterval(state.autoplayTimer);
    state.autoplayTimer = null;
  }
  els.play.textContent = "▶ Reproducir";
  els.play.setAttribute("aria-pressed", "false");
}

function replaceStage(element) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const previous = els.signStage.querySelector(".stage-slide:not(.stage-exiting)");
  const layer = document.createElement("div");
  layer.className = "stage-slide";
  layer.append(element);

  // Se superponen brevemente las dos señas para un fundido natural.
  for (const stale of els.signStage.querySelectorAll(".stage-exiting")) stale.remove();
  if (reduceMotion || !previous) {
    els.signStage.replaceChildren(layer);
  } else {
    previous.classList.add("stage-exiting");
    previous.setAttribute("aria-hidden", "true");
    els.signStage.append(layer);
    window.setTimeout(() => previous.remove(), SIGN_TRANSITION_MS + 60);
  }
}

function makeStage(title, description, largeChar = null) {
  const wrap = document.createElement("div");
  wrap.className = "sign-char-placeholder";
  if (largeChar !== null) {
    const char = document.createElement("span");
    char.className = "big-char";
    char.textContent = largeChar;
    wrap.append(char);
  }
  const heading = document.createElement("strong");
  heading.textContent = title;
  const message = document.createElement("span");
  message.textContent = description;
  wrap.append(heading, message);
  replaceStage(wrap);
}

function renderSequence() {
  const total = state.sequence.length;
  els.sequenceCounter.textContent = total ? `${state.sequenceIndex + 1} / ${total}` : "0 / 0";
  els.sequenceStrip.replaceChildren();
  if (!total) {
    makeStage("Sin contenido", "Escribe un texto y pulsa Mostrar señas.");
    els.signMessage.textContent = "Todavía no hay caracteres para mostrar.";
    els.previous.disabled = els.next.disabled = els.play.disabled = true;
    return;
  }

  const current = state.sequence[state.sequenceIndex];
  if (current === " ") {
    makeStage("Espacio", "Separación entre palabras.", "␣");
    els.signMessage.textContent = "Espacio en el texto.";
  } else if (VOWELS.has(current)) {
    const entry = SIGN_LIBRARY[current];
    if (entry?.type === "image" && entry.src) {
      const img = document.createElement("img");
      img.className = "sign-actual-image";
      img.alt = `Seña correspondiente a la vocal ${current} en LSB`;
      img.src = entry.src;
      img.onerror = () => {
        if (state.sequence[state.sequenceIndex] === current) {
          makeStage("Imagen no disponible", "No se pudo mostrar esta seña.", current);
        }
      };
      img.decoding = "async";
      replaceStage(img);
      els.signMessage.textContent = `Vocal ${current} en alfabeto dactilológico LSB.`;
    } else {
      makeStage(`Vocal ${current}`, "Imagen de LSB aún no disponible.", current);
      els.signMessage.textContent = "La imagen verificada de esta seña está pendiente de incorporación.";
    }
  } else {
    makeStage("Sin representación disponible", "Este carácter todavía no está incorporado.", current);
    els.signMessage.textContent = `El carácter «${current}» todavía no tiene una representación disponible.`;
  }

  for (const [index, char] of state.sequence.entries()) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "sequence-chip";
    chip.textContent = char === " " ? "␣" : char;
    chip.title = `Ir al carácter ${index + 1}`;
    chip.setAttribute("aria-label", `Mostrar carácter ${index + 1}: ${char === " " ? "espacio" : char}`);
    if (index === state.sequenceIndex) chip.setAttribute("aria-current", "step");
    chip.addEventListener("click", () => {
      stopAutoplay();
      state.sequenceIndex = index;
      renderSequence();
    });
    els.sequenceStrip.append(chip);
  }
  const selected = els.sequenceStrip.children[state.sequenceIndex];
  if (selected) {
    const left = selected.offsetLeft - els.sequenceStrip.offsetLeft;
    const desired = left - els.sequenceStrip.clientWidth / 2 + selected.clientWidth / 2;
    els.sequenceStrip.scrollTo({ left: desired, behavior: "smooth" });
  }
  els.previous.disabled = state.sequenceIndex === 0;
  els.next.disabled = state.sequenceIndex === total - 1;
  els.play.disabled = false;
  // Mantener el estado visual del botón durante la reproducción.
  els.play.textContent = state.autoplayTimer !== null ? "Ⅱ Pausar" : "▶ Reproducir";
}

function processText() {
  stopAutoplay();
  const normalized = els.sourceText.value.toUpperCase().replace(/\s/g, " ");
  els.sourceText.value = normalized;
  els.charCount.textContent = `${normalized.length} / 160`;
  state.sequence = Array.from(normalized);
  state.sequenceIndex = 0;
  renderSequence();
}

function goToCharacter(offset) {
  stopAutoplay();
  const next = state.sequenceIndex + offset;
  if (next < 0 || next >= state.sequence.length) return;
  state.sequenceIndex = next;
  renderSequence();
}

function toggleAutoplay() {
  if (state.autoplayTimer !== null) return stopAutoplay();
  if (!state.sequence.length) return;
  if (state.sequenceIndex >= state.sequence.length - 1) {
    state.sequenceIndex = 0;
    renderSequence();
  }
  els.play.textContent = "Ⅱ Pausar";
  els.play.setAttribute("aria-pressed", "true");
  state.autoplayTimer = window.setInterval(() => {
    if (state.sequenceIndex >= state.sequence.length - 1) {
      stopAutoplay();
    } else {
      state.sequenceIndex += 1;
      renderSequence();
    }
  }, PLAYBACK_INTERVAL_MS);
}

// Acciones y eventos
els.switchDirection.addEventListener("click", switchDirection);
els.cameraToggle.addEventListener("click", toggleCamera);
els.cameraSelect.addEventListener("change", () => { if (state.stream) startCamera(els.cameraSelect.value); });
els.addChar.addEventListener("click", () => {
  if (VOWELS.has(state.detectedLetter)) els.translatedText.value += state.detectedLetter;
});
els.addSpace.addEventListener("click", () => { els.translatedText.value += " "; });
els.deleteChar.addEventListener("click", () => { els.translatedText.value = els.translatedText.value.slice(0, -1); });
els.clearText.addEventListener("click", () => { els.translatedText.value = ""; });
els.copyText.addEventListener("click", copyTranslation);
els.sourceText.addEventListener("input", () => { els.charCount.textContent = `${els.sourceText.value.length} / 160`; });
els.sourceText.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") processText();
});
els.processText.addEventListener("click", processText);
els.previous.addEventListener("click", () => goToCharacter(-1));
els.next.addEventListener("click", () => goToCharacter(1));
els.play.addEventListener("click", toggleAutoplay);
window.addEventListener("pagehide", () => { stopCamera(); stopAutoplay(); });
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopRecognition();
    clearDetection("Reconocimiento en pausa.");
  } else {
    void checkAvailability();
  }
});

setDirection("lsb");
refreshCameraState();
checkAvailability();
