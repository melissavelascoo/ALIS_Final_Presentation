(() => {
  'use strict';

  const API_BASE = '';
  const SCAN_INTERVAL_MS = 180;
  const CAPTURE_WIDTH = 640;
  const IMAGE_QUALITY = 0.78;
  const STATIC_TOKEN_GAP_MS = 900;
  const DYNAMIC_TOKEN_GAP_MS = 1400;
  const PLAYBACK_INTERVAL_MS = 760;
  const WORD_PAUSE_MS = 2500;
  const $ = (id) => document.getElementById(id);

  const els = {
    switchDirection: $('switchDirectionBtn'),
    fromLabel: $('fromLabel'),
    toLabel: $('toLabel'),
    lsbMode: $('lsbMode'),
    spanishMode: $('spanishMode'),
    modeCaption: $('modeCaption'),
    video: $('cameraVideo'),
    cameraView: $('cameraView'),
    cameraPlaceholder: $('cameraPlaceholder'),
    cameraToggle: $('cameraToggleBtn'),
    cameraSelect: $('cameraSelect'),
    cameraBadge: $('cameraBadge'),
    cameraStatus: $('cameraStatus'),
    liveCameraTag: $('liveCameraTag'),
    canvas: $('captureCanvas'),
    recognitionMode: $('recognitionMode'),
    autoRecognitionText: $('autoRecognitionText'),
    recognizedChar: $('recognizedChar'),
    recognitionMessage: $('recognitionMessage'),
    translatedText: $('translatedText'),
    addChar: $('addCharBtn'),
    addSpace: $('addSpaceBtn'),
    deleteChar: $('deleteCharBtn'),
    clearText: $('clearTextBtn'),
    copyText: $('copyTextBtn'),
    sourceText: $('sourceText'),
    charCount: $('charCount'),
    processText: $('processTextBtn'),
    signStage: $('signStage'),
    signMessage: $('signMessage'),
    sequenceCounter: $('sequenceCounter'),
    sequenceStrip: $('sequenceStrip'),
    previous: $('prevBtn'),
    next: $('nextBtn'),
    play: $('playBtn'),
    toast: $('toast'),
  };

  const state = {
    direction: 'lsb',
    stream: null,
    modelStatus: null,
    modelAvailable: false,
    analyzing: false,
    scanTimer: null,
    scanController: null,
    scanSession: 0,
    pendingStatic: null,
    staticHits: 0,
    lastTokenKey: null,
    tokenQueue: [],
    tokenTimer: null,
    nextTokenAllowedAt: 0,
    sequence: [],
    sequenceIndex: 0,
    playbackTimer: null,
    playing: false,
    toastTimer: null,
    initialSignStage: els.signStage.innerHTML,
  };

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { els.toast.hidden = true; }, 2600);
  }

  function setDirection(direction) {
    if (state.direction === direction) return;
    state.direction = direction;
    stopPlayback();

    const isLsb = direction === 'lsb';
    els.fromLabel.textContent = isLsb ? 'Lengua de Señas Boliviana' : 'Español';
    els.toLabel.textContent = isLsb ? 'Español' : 'Lengua de Señas Boliviana';
    els.lsbMode.hidden = !isLsb;
    els.spanishMode.hidden = isLsb;
    els.modeCaption.textContent = isLsb ? 'De señas a texto' : 'Alfabeto ilustrado';

    if (isLsb) {
      clearSequence();
      if (state.stream && state.modelAvailable) scheduleRecognition(0);
    } else {
      stopCamera();
    }
  }

  async function resetBackendBuffer() {
    try {
      const response = await fetch(`${API_BASE}/api/reset`, { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    } catch (error) {
      console.warn('No se pudo reiniciar el búfer de reconocimiento:', error);
    }
  }

  function modelDescription(name, model) {
    if (model?.loaded) {
      const count = Array.isArray(model.classes) ? model.classes.length : 0;
      return `${name}: listo (${count} clases)`;
    }
    return `${name}: ${model?.message || 'no disponible'}`;
  }

  async function checkAvailability() {
    try {
      const [health, response] = await Promise.all([
        fetch(`${API_BASE}/api/health`, { cache: 'no-store' }),
        fetch(`${API_BASE}/api/model-status`, { cache: 'no-store' }),
      ]);
      if (!health.ok || !response.ok) throw new Error('El servicio de reconocimiento no está disponible.');

      const data = await response.json();
      state.modelStatus = data;
      state.modelAvailable = data.loaded === true &&
        Boolean(data.static?.loaded || data.dynamic?.loaded);
      els.recognitionMode.querySelector('[value="static"]').disabled = !data.static?.loaded;
      els.recognitionMode.querySelector('[value="dynamic"]').disabled = !data.dynamic?.loaded;
      if (els.recognitionMode.selectedOptions[0]?.disabled) {
        els.recognitionMode.value = 'both';
      }
      els.autoRecognitionText.textContent = [
        modelDescription('MLP estático', data.static),
        modelDescription('LSTM dinámico', data.dynamic),
      ].join(' · ');
      els.recognitionMessage.textContent = state.modelAvailable
        ? 'Selecciona un modo, activa la cámara y muestra una seña.'
        : 'No hay modelos SLR disponibles. Revisa los archivos model.h5 y labels.txt.';
    } catch (error) {
      state.modelStatus = null;
      state.modelAvailable = false;
      els.autoRecognitionText.textContent = 'No se pudo conectar con el backend de ALIS.';
      els.recognitionMessage.textContent = error.message;
      console.warn('No se pudo consultar el estado de los modelos:', error);
    }

    document.querySelector('.tiny-indicator')?.classList.toggle('is-offline', !state.modelAvailable);
    if (!state.modelAvailable) {
      stopRecognition();
      if (state.stream) els.cameraStatus.textContent = 'Cámara activa; modelos no disponibles.';
    } else if (state.stream && state.direction === 'lsb' && !document.hidden) {
      scheduleRecognition(0);
    }
  }

  function scheduleRecognition(delay = SCAN_INTERVAL_MS) {
    if (state.scanTimer !== null || state.analyzing || !state.stream ||
        !state.modelAvailable || state.direction !== 'lsb' || document.hidden) return;
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
    state.pendingStatic = null;
    state.staticHits = 0;
  }

  async function loadCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === 'videoinput');
      const selected = els.cameraSelect.value;
      els.cameraSelect.replaceChildren(new Option('Predeterminada', ''));
      cameras.forEach((device, index) => {
        els.cameraSelect.add(new Option(device.label || `Cámara ${index + 1}`, device.deviceId));
      });
      if (cameras.some((device) => device.deviceId === selected)) {
        els.cameraSelect.value = selected;
      }
      els.cameraSelect.disabled = !state.stream || cameras.length < 2;
    } catch (error) {
      console.warn('No se pudieron enumerar las cámaras:', error);
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showToast('La cámara requiere HTTPS o abrir el proyecto en localhost.');
      return;
    }
    stopCamera();
    els.cameraStatus.textContent = 'Solicitando acceso a la cámara…';
    const deviceId = els.cameraSelect.value;
    const video = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: CAPTURE_WIDTH }, height: { ideal: 360 } }
      : { facingMode: 'user', width: { ideal: CAPTURE_WIDTH }, height: { ideal: 360 } };

    try {
      state.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      els.video.srcObject = state.stream;
      await els.video.play();
      state.stream.getVideoTracks()[0]?.addEventListener('ended', () => stopCamera());
      els.cameraView.classList.add('camera-active');
      els.cameraPlaceholder.hidden = true;
      els.liveCameraTag.hidden = false;
      els.cameraBadge.textContent = 'Encendida';
      els.cameraStatus.textContent = state.modelAvailable
        ? 'Cámara activa. Muestra una seña frente a ella.'
        : 'Cámara activa; esperando que haya un modelo disponible.';
      els.cameraToggle.textContent = 'Desactivar cámara';
      await loadCameras();
      if (state.modelAvailable) scheduleRecognition(0);
    } catch (error) {
      console.error('No se pudo activar la cámara:', error);
      const message = error?.name === 'NotAllowedError'
        ? 'Permite el acceso a la cámara en tu navegador.'
        : error?.name === 'NotFoundError'
          ? 'No se encontró una cámara disponible.'
          : 'No fue posible activar la cámara.';
      els.cameraStatus.textContent = message;
      showToast(message);
      stopCamera();
    }
  }

  function stopCamera(resetBackend = true) {
    stopRecognition();
    clearPendingTokens();
    if (resetBackend && state.stream) void resetBackendBuffer();
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
    els.video.srcObject = null;
    els.cameraSelect.disabled = true;
    els.cameraView.classList.remove('camera-active', 'is-recognizing');
    els.cameraPlaceholder.hidden = false;
    els.liveCameraTag.hidden = true;
    els.cameraBadge.textContent = 'Apagada';
    els.cameraStatus.textContent = 'Puedes activar la cámara cuando quieras.';
    els.cameraToggle.textContent = 'Activar cámara';
  }

  function appendToken(token, model) {
    let translatedToken = token;
    if (model === 'dynamic' && token === 'SER') {
      const previousWords = els.translatedText.value.trim().toLocaleUpperCase('es').split(/\s+/);
      if (previousWords.slice(-3).join(' ') === 'HOLA MI NOMBRE') {
        translatedToken = 'ES';
      }
    }

    const current = els.translatedText.value;
    const separator = current && !/\s$/.test(current) && translatedToken.length > 1 ? ' ' : '';
    els.translatedText.value =
      `${current}${separator}${translatedToken}${translatedToken.length > 1 ? ' ' : ''}`;
    els.recognizedChar.textContent = translatedToken;
    els.addChar.disabled = false;
  }

  function enqueueToken(token, model) {
    if (!token) return;
    const key = `${model}:${token}`;
    if (key === state.lastTokenKey || state.tokenQueue.some((item) => item.key === key)) return;
    const waiting = state.tokenTimer !== null || Date.now() < state.nextTokenAllowedAt;
    state.tokenQueue.push({ key, token, model });
    processTokenQueue();
    if (waiting) {
      els.recognitionMessage.textContent = 'Resultado en espera; separando señas para evitar solapamientos.';
    }
  }

  function processTokenQueue() {
    if (state.tokenTimer !== null || !state.tokenQueue.length) return;
    const wait = state.nextTokenAllowedAt - Date.now();
    if (wait > 0) {
      state.tokenTimer = window.setTimeout(() => {
        state.tokenTimer = null;
        processTokenQueue();
      }, wait);
      return;
    }

    const item = state.tokenQueue.shift();
    appendToken(item.token, item.model);
    state.lastTokenKey = item.key;
    const gap = item.model === 'dynamic' ? DYNAMIC_TOKEN_GAP_MS : STATIC_TOKEN_GAP_MS;
    state.nextTokenAllowedAt = Date.now() + gap;
    state.tokenTimer = window.setTimeout(() => {
      state.tokenTimer = null;
      processTokenQueue();
    }, gap);
  }

  function clearPendingTokens() {
    window.clearTimeout(state.tokenTimer);
    state.tokenTimer = null;
    state.tokenQueue = [];
    state.nextTokenAllowedAt = 0;
  }

  function applyPrediction(result) {
    const staticResult = result?.static || {};
    const dynamicResult = result?.dynamic || {};
    const dynamicLabel = String(dynamicResult.label || '').trim().toUpperCase();
    const staticLabel = String(staticResult.label || '').trim().toUpperCase();
    const dynamicAccepted = dynamicResult.accepted === true && dynamicLabel.length > 0;
    const staticAccepted = staticResult.accepted === true && staticLabel.length > 0;
    const dynamicBusy = dynamicAccepted ||
      dynamicResult.activity === 'moving' ||
      ['uncertain', 'cooldown'].includes(dynamicResult.state);

    if (dynamicAccepted) {
      state.pendingStatic = null;
      state.staticHits = 0;
      enqueueToken(dynamicLabel, 'dynamic');
      els.cameraStatus.textContent = `LSTM dinámico: ${dynamicLabel} (${Math.round((dynamicResult.confidence || 0) * 100)}%).`;
      els.recognitionMessage.textContent = 'Seña dinámica reconocida.';
      return;
    }

    if (staticAccepted && !dynamicBusy) {
      if (state.pendingStatic === staticLabel) {
        state.staticHits += 1;
      } else {
        state.pendingStatic = staticLabel;
        state.staticHits = 1;
      }
      if (state.staticHits >= 2) {
        enqueueToken(staticLabel, 'static');
        els.cameraStatus.textContent = `MLP estático: letra ${staticLabel} (${Math.round((staticResult.confidence || 0) * 100)}%).`;
        els.recognitionMessage.textContent = state.tokenQueue.length
          ? 'Letra reconocida; esperando el intervalo entre señas.'
          : 'Letra reconocida.';
      } else {
        els.cameraStatus.textContent = `Verificando letra ${staticLabel}…`;
      }
      return;
    }

    state.pendingStatic = null;
    state.staticHits = 0;
    if (!dynamicBusy && (dynamicResult.state === 'no_hand' || staticResult.state === 'no_hand')) {
      state.lastTokenKey = null;
      els.cameraStatus.textContent = 'No se detecta una seña.';
    } else if (dynamicBusy) {
      els.cameraStatus.textContent = 'LSTM dinámico activo; evitando duplicados del MLP.';
    } else if (dynamicResult.state === 'collecting') {
      els.cameraStatus.textContent = 'LSTM dinámico: reuniendo fotogramas de la seña…';
    } else if (dynamicResult.state === 'cooldown') {
      els.cameraStatus.textContent = 'LSTM dinámico: esperando la siguiente seña.';
    } else if (staticResult.state === 'uncertain' || dynamicResult.state === 'uncertain') {
      els.cameraStatus.textContent = 'Seña poco clara; mantenla visible y estable.';
    } else {
      els.cameraStatus.textContent = 'Observando seña…';
    }
    els.recognitionMessage.textContent = [
      `MLP: ${staticResult.state || 'inactivo'}`,
      `LSTM: ${dynamicResult.state || 'inactivo'}`,
    ].join(' · ');
  }

  async function analyzeFrame() {
    if (!state.stream || !state.modelAvailable || state.analyzing ||
        state.direction !== 'lsb' || document.hidden) return;
    if (!els.video.videoWidth || !els.video.videoHeight) {
      scheduleRecognition(200);
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
      canvas.width = CAPTURE_WIDTH;
      canvas.height = Math.max(1, Math.round(CAPTURE_WIDTH / aspect));
      const context = canvas.getContext('2d');
      context.drawImage(els.video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', IMAGE_QUALITY));
      if (session !== state.scanSession || controller.signal.aborted) return;
      if (!blob) throw new Error('No se pudo capturar el fotograma de la cámara.');

      const form = new FormData();
      form.append('frame', blob, 'frame.jpg');
      form.append('mode', els.recognitionMode.value);
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });
      if (session !== state.scanSession || controller.signal.aborted) return;
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Error de reconocimiento (HTTP ${response.status}).`);
      }
      applyPrediction(await response.json());
    } catch (error) {
      if (controller.signal.aborted || session !== state.scanSession) return;
      console.error('Falló el reconocimiento del fotograma:', error);
      els.recognitionMessage.textContent = error.message;
      nextDelay = 1200;
    } finally {
      if (session === state.scanSession) {
        state.analyzing = false;
        state.scanController = null;
        scheduleRecognition(nextDelay);
      }
    }
  }

  function updateTextButtons() {
    els.addChar.disabled = !els.recognizedChar.textContent.trim() ||
      els.recognizedChar.textContent.trim() === '—';
  }

  function clearText() {
    clearPendingTokens();
    els.translatedText.value = '';
    els.recognizedChar.textContent = '—';
    state.lastTokenKey = null;
    updateTextButtons();
  }

  async function copyText() {
    if (!els.translatedText.value) return;
    try {
      await navigator.clipboard.writeText(els.translatedText.value);
      showToast('Texto copiado.');
    } catch (error) {
      console.warn('No se pudo usar el portapapeles:', error);
      els.translatedText.focus();
      els.translatedText.select();
      showToast('Selecciona el texto y cópialo con Ctrl+C.');
    }
  }

  function imageForLetter(letter) {
    if (!/^[A-I]$/.test(letter) && !/^[K-Z]$/.test(letter)) return null;
    const code = letter.charCodeAt(0);
    const imageNumber = code >= 75 ? code - 64 : code - 63;
    return `/assets/img/alfabeto%20lsb/${String(imageNumber).padStart(2, '0')}.png`;
  }

  function createSequence(text) {
    const normalized = text.trim().toLocaleUpperCase('es').replace(/\s+/g, ' ');
    return Array.from(normalized).map((character) => character === ' '
      ? { type: 'space', character }
      : { type: 'letter', character, image: imageForLetter(character) });
  }

  function processText() {
    if (!els.sourceText.value.trim()) {
      showToast('Escribe un mensaje para mostrar el alfabeto.');
      els.sourceText.focus();
      return;
    }
    stopPlayback();
    state.sequence = createSequence(els.sourceText.value);
    state.sequenceIndex = 0;
    renderSequence();
    els.signMessage.textContent = 'Deletreo con el alfabeto ilustrado; no es una traducción gramatical a LSB.';
  }

  function renderSequence() {
    const total = state.sequence.length;
    if (!total) {
      els.signStage.innerHTML = state.initialSignStage;
      els.sequenceCounter.textContent = '0 / 0';
      els.sequenceStrip.replaceChildren();
      els.previous.disabled = true;
      els.next.disabled = true;
      els.play.disabled = true;
      return;
    }

    const item = state.sequence[state.sequenceIndex];
    els.signStage.replaceChildren();
    if (item.type === 'space') {
      const pause = document.createElement('div');
      pause.className = 'sequence-pause-card';
      pause.setAttribute('role', 'img');
      pause.setAttribute('aria-label', 'Pausa entre palabras');
      pause.style.setProperty('--pause-duration', `${WORD_PAUSE_MS}ms`);
      pause.innerHTML = '<svg class="sequence-pause-icon" viewBox="0 0 72 72" aria-hidden="true"><circle class="pause-timer-track" cx="36" cy="36" r="29"></circle><circle class="pause-timer-progress" cx="36" cy="36" r="29"></circle><path class="pause-symbol" d="M30 27v18m12-18v18"></path></svg>';
      els.signStage.appendChild(pause);
      els.signMessage.textContent = '';
    } else if (item.image) {
      const image = document.createElement('img');
      image.className = 'sign-actual-image';
      image.alt = `Seña de la letra ${item.character}`;
      image.addEventListener('error', () => {
        image.remove();
        els.signMessage.textContent = `Ilustración no disponible para ${item.character}.`;
      }, { once: true });
      image.src = item.image;
      els.signStage.appendChild(image);
    } else {
      const letter = document.createElement('strong');
      letter.className = 'sign-fallback-letter';
      letter.textContent = item.character;
      els.signStage.appendChild(letter);
    }
    if (item.type !== 'space') {
      els.signMessage.textContent = item.image
        ? `Letra ${item.character} del alfabeto dactilológico.`
        : `No hay ilustración disponible para ${item.character}.`;
    }
    els.sequenceCounter.textContent = `${state.sequenceIndex + 1} / ${total}`;
    els.sequenceStrip.replaceChildren();

    state.sequence.forEach((entry, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `sequence-item sequence-chip${index === state.sequenceIndex ? ' active' : ''}`;
      button.textContent = entry.type === 'space' ? '···' : entry.character;
      button.title = entry.type === 'space' ? 'Pausa entre palabras' : `Letra ${entry.character}`;
      button.setAttribute('aria-label', button.title);
      button.setAttribute('aria-current', index === state.sequenceIndex ? 'step' : 'false');
      button.addEventListener('click', () => {
        stopPlayback();
        state.sequenceIndex = index;
        renderSequence();
      });
      els.sequenceStrip.appendChild(button);
    });

    els.previous.disabled = state.sequenceIndex === 0;
    els.next.disabled = state.sequenceIndex === total - 1;
    els.play.disabled = false;
  }

  function clearSequence() {
    stopPlayback();
    state.sequence = [];
    state.sequenceIndex = 0;
    renderSequence();
  }

  function stopPlayback() {
    window.clearTimeout(state.playbackTimer);
    state.playbackTimer = null;
    state.playing = false;
    els.play.textContent = '▶ Reproducir';
  }

  function schedulePlaybackStep() {
    if (!state.playing || !state.sequence.length) return;
    const currentItem = state.sequence[state.sequenceIndex];
    const delay = currentItem.type === 'space' ? WORD_PAUSE_MS : PLAYBACK_INTERVAL_MS;
    state.playbackTimer = window.setTimeout(() => {
      state.playbackTimer = null;
      if (state.sequenceIndex >= state.sequence.length - 1) {
        stopPlayback();
        return;
      }
      state.sequenceIndex += 1;
      renderSequence();
      schedulePlaybackStep();
    }, delay);
  }

  function startPlayback() {
    if (!state.sequence.length) return;
    if (state.playing) {
      stopPlayback();
      return;
    }
    if (state.sequenceIndex >= state.sequence.length - 1) state.sequenceIndex = 0;
    state.playing = true;
    els.play.textContent = 'Ⅱ Pausar';
    renderSequence();
    schedulePlaybackStep();
  }

  els.switchDirection.addEventListener('click', () => {
    setDirection(state.direction === 'lsb' ? 'es' : 'lsb');
  });
  els.cameraToggle.addEventListener('click', () => {
    if (state.stream) stopCamera();
    else void startCamera();
  });
  els.cameraSelect.addEventListener('change', () => {
    if (state.stream) void startCamera();
  });
  els.recognitionMode.addEventListener('change', () => {
    stopRecognition();
    state.pendingStatic = null;
    state.staticHits = 0;
    state.lastTokenKey = null;
    clearPendingTokens();
    void resetBackendBuffer().then(() => {
      if (state.stream && state.modelAvailable) scheduleRecognition(0);
    });
  });
  els.addChar.addEventListener('click', () => {
    const token = els.recognizedChar.textContent.trim();
    if (token && token !== '—') {
      const current = els.translatedText.value;
      els.translatedText.value = `${current}${current && !/\s$/.test(current) && token.length > 1 ? ' ' : ''}${token}`;
    }
  });
  els.addSpace.addEventListener('click', () => { els.translatedText.value += ' '; });
  els.deleteChar.addEventListener('click', () => {
    els.translatedText.value = els.translatedText.value.slice(0, -1);
  });
  els.clearText.addEventListener('click', clearText);
  els.copyText.addEventListener('click', () => void copyText());
  els.sourceText.addEventListener('input', () => {
    els.charCount.textContent = `${els.sourceText.value.length} / 160`;
  });
  els.processText.addEventListener('click', processText);
  els.previous.addEventListener('click', () => {
    stopPlayback();
    if (state.sequenceIndex > 0) {
      state.sequenceIndex -= 1;
      renderSequence();
    }
  });
  els.next.addEventListener('click', () => {
    stopPlayback();
    if (state.sequenceIndex < state.sequence.length - 1) {
      state.sequenceIndex += 1;
      renderSequence();
    }
  });
  els.play.addEventListener('click', startPlayback);

  window.addEventListener('pagehide', () => {
    stopCamera(false);
    stopPlayback();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopRecognition();
    else void checkAvailability();
  });

  updateTextButtons();
  void loadCameras();
  void checkAvailability();
})();
