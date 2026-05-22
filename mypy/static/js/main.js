'use strict';

// ── DOM 요소 ──────────────────────────────────────
const webcamEl      = document.getElementById('webcam');
const procFrameEl   = document.getElementById('proc-frame');
const placeholderEl = document.getElementById('video-placeholder');
const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const connBadge     = document.getElementById('conn-badge');
const statusCard    = document.getElementById('status-card');
const statusIcon    = document.getElementById('status-icon');
const statusLabel   = document.getElementById('status-label');
const statusSub     = document.getElementById('status-sub');
const phoneAlert    = document.getElementById('phone-alert');
const statStudy     = document.getElementById('stat-study');
const statFocus     = document.getElementById('stat-focus');
const statSleep     = document.getElementById('stat-sleep');
const statAway      = document.getElementById('stat-away');
const statPhone     = document.getElementById('stat-phone');
const gaugeFill     = document.getElementById('gauge-fill');
const gaugeLabel    = document.getElementById('gauge-label');
const toast         = document.getElementById('toast');

// ── 상태 ──────────────────────────────────────────
let socket      = null;
let isRunning   = false;
let studyTimer  = null;
let waiting     = false; // 서버 응답 대기 중 여부

let studySec    = 0;
let totalFrames = 0;
let studyFrames = 0;
let sleepCount  = 0;
let awayCount   = 0;
let phoneCount  = 0;
let lastStatus  = '';
let phoneWas    = false;

// 캡처 캔버스 (DOM에 붙이지 않음)
const canvas = document.createElement('canvas');
const ctx    = canvas.getContext('2d');

// ── 상태 메타 ─────────────────────────────────────
const STATUS_META = {
  STUDYING: { icon: '📚', label: '집중 중',  sub: '잘 하고 있어요!',     cls: 'status-studying' },
  SLEEPING: { icon: '😴', label: '졸음 감지', sub: '잠깐 깨어나세요!',   cls: 'status-sleeping' },
  AWAY:     { icon: '🚶', label: '자리 비움', sub: '자리를 비웠습니다',  cls: 'status-away'     },
  PHONE:    { icon: '📵', label: '폰 사용',   sub: '집중을 방해합니다!', cls: 'status-phone'    },
};

// ── WebSocket 연결 ────────────────────────────────
function connect() {
  setBadge('connecting', '연결 중...');
  socket = io({ transports: ['websocket'] });

  socket.on('connect',    ()  => setBadge('connecting', '모델 로딩 중...'));
  socket.on('connected',  ()  => setBadge('ready', '준비 완료'));
  socket.on('disconnect', ()  => { setBadge('idle', '오프라인'); stopMonitor(); });
  socket.on('error',      (e) => showToast('오류: ' + e.message));
  socket.on('result',     handleResult);
}

// ── 프레임 캡처 & 전송 ───────────────────────────
function sendFrame() {
  if (!isRunning || !socket?.connected || waiting) return;

  const vw = webcamEl.videoWidth;
  const vh = webcamEl.videoHeight;
  if (!vw || !vh) {
    // 비디오 스트림 아직 준비 안 됨 — 100ms 후 재시도
    setTimeout(sendFrame, 100);
    return;
  }

  canvas.width  = vw;
  canvas.height = vh;
  ctx.drawImage(webcamEl, 0, 0, vw, vh);

  waiting = true;
  socket.emit('frame', { image: canvas.toDataURL('image/jpeg', 0.9) });
}

// ── 결과 처리 → 다음 프레임 즉시 요청 ───────────
function handleResult(data) {
  waiting = false;
  if (!isRunning) return;

  // 처리된 프레임: 서버에서 받은 즉시 웹캠 위에 덮어 표시
  procFrameEl.src = data.frame;
  procFrameEl.style.display = 'block';

  const s = data.has_phone ? 'PHONE' : data.status;
  totalFrames++;
  if (s === 'STUDYING') studyFrames++;

  // 상태 전환 감지
  if (s !== lastStatus) {
    if (s === 'SLEEPING') { sleepCount++; statSleep.textContent = sleepCount; showToast('졸음이 감지되었습니다!'); }
    if (s === 'AWAY')     { awayCount++;  statAway.textContent  = awayCount; }
    lastStatus = s;
  }

  // 폰 감지
  if (data.has_phone && !phoneWas) {
    phoneCount++;
    statPhone.textContent = phoneCount;
    showToast('휴대폰 사용이 감지되었습니다!');
  }
  phoneWas = data.has_phone;
  phoneAlert.classList.toggle('hidden', !data.has_phone);

  // 상태 카드 업데이트
  const meta = STATUS_META[s] || STATUS_META.STUDYING;
  statusCard.className    = `card status-card ${meta.cls}`;
  statusIcon.textContent  = meta.icon;
  statusLabel.textContent = meta.label;
  statusSub.textContent   = meta.sub;

  // 집중도 업데이트
  const pct = totalFrames > 0 ? Math.round((studyFrames / totalFrames) * 100) : 0;
  statFocus.textContent = pct + '%';
  gaugeFill.style.width = pct + '%';
  gaugeLabel.textContent = pct + '%';

  // 서버 응답이 오면 바로 다음 프레임 전송
  sendFrame();
}

// ── 학습 타이머 ──────────────────────────────────
function startStudyTimer() {
  studyTimer = setInterval(() => {
    if (lastStatus === 'STUDYING') {
      studySec++;
      const h = String(Math.floor(studySec / 3600)).padStart(2, '0');
      const m = String(Math.floor((studySec % 3600) / 60)).padStart(2, '0');
      const s = String(studySec % 60).padStart(2, '0');
      statStudy.textContent = `${h}:${m}:${s}`;
    }
  }, 1000);
}

// ── 모니터링 시작 ─────────────────────────────────
async function startMonitor() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    webcamEl.srcObject = stream;

    // 비디오 메타데이터 로드 대기
    await new Promise((resolve, reject) => {
      webcamEl.onloadedmetadata = resolve;
      webcamEl.onerror = reject;
    });
    await webcamEl.play();

    isRunning = true;
    waiting   = false;
    btnStart.disabled = true;
    btnStop.disabled  = false;
    setBadge('running', '모니터링 중');

    // 실제 카메라 비율로 컨테이너 aspect-ratio 조정 (잘림 방지)
    const vw = webcamEl.videoWidth  || 1280;
    const vh = webcamEl.videoHeight || 720;
    document.querySelector('.video-wrap').style.aspectRatio = `${vw} / ${vh}`;

    // 즉시 원본 웹캠을 화면에 표시 (서버 응답 기다리지 않음)
    placeholderEl.style.display = 'none';
    webcamEl.style.display      = 'block';

    startStudyTimer();
    sendFrame(); // 첫 프레임 전송 → 이후 handleResult에서 연속 전송
  } catch (err) {
    showToast('웹캠 접근 실패: ' + err.message);
  }
}

// ── 모니터링 중지 ─────────────────────────────────
function stopMonitor() {
  isRunning = false;
  waiting   = false;
  clearInterval(studyTimer);
  studyTimer = null;

  if (webcamEl.srcObject) {
    webcamEl.srcObject.getTracks().forEach(t => t.stop());
    webcamEl.srcObject = null;
  }

  webcamEl.style.display      = 'none';
  procFrameEl.style.display   = 'none';
  placeholderEl.style.display = 'flex';
  btnStart.disabled = false;
  btnStop.disabled  = true;
  phoneAlert.classList.add('hidden');

  statusCard.className    = 'card status-card status-idle';
  statusIcon.textContent  = '⏸';
  statusLabel.textContent = '대기 중';
  statusSub.textContent   = '모니터링이 시작되면 상태가 표시됩니다';

  if (socket?.connected) setBadge('ready', '준비 완료');
}

// ── 토스트 ───────────────────────────────────────
let toastTimer = null;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 3000);
}

// ── 배지 ─────────────────────────────────────────
function setBadge(type, text) {
  connBadge.className   = `badge badge-${type}`;
  connBadge.textContent = text;
}

// ── 이벤트 ───────────────────────────────────────
btnStart.addEventListener('click', startMonitor);
btnStop.addEventListener('click',  stopMonitor);

// ── 초기화 ───────────────────────────────────────
connect();
