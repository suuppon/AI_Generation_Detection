// content.js — FINAL (Find Video blue w/black text, red "Synthetic" in analysis (except button),
// Synthetic segmented button emphasized, slideshow on Analyze, old Find Video logic kept)
(() => {
  if (window.__synsKillerMounted) return;
  window.__synsKillerMounted = true;

  // ===== Root mount =====
  const host = document.createElement("div");
  host.style.position = "fixed";
  host.style.zIndex = "2147483647";
  host.style.top = "90px";
  host.style.right = "16px";
  host.style.width = "360px";
  host.style.pointerEvents = "none";
  document.documentElement.appendChild(host);

  // Shadow DOM
  const shadow = host.attachShadow({ mode: "open" });

  // ===== Styles =====
  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
    :root {
      --panel-bg: rgba(255,255,255,.88);
      --panel-border: rgba(0,0,0,.10);
      --panel-border-soft: rgba(0,0,0,.06);
      --text: #111827;
      --muted: #6b7280;
      --ring: rgba(99,102,241,.35);
      --accent: #2b64ff;
      --success: #10b981;
      --danger: #ef6a39;
      --r: 40px;
    }
    .panel {
      pointer-events: auto;
      display: flex; flex-direction: column;
      width: 100%;
      max-height: 86vh; overflow: hidden;
      border-radius: var(--r);
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(17px);
      box-shadow: 0 18px 54px rgba(0,0,0,.22), inset 0 0 0 1px rgba(255,255,255,.25);
      color: var(--text);
    }
    .head {
      position: sticky; top: 0; z-index: 5;
      padding: 10px 12px;
      background: linear-gradient(180deg, rgba(255,255,255,.55), rgba(255,255,255,0));
      border-bottom: 1px solid var(--panel-border-soft);
      backdrop-filter: blur(8px);
    }
    .title-row{ display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .title{ font-weight: 700; font-size: 15px; letter-spacing:.2px; }
    .badge{ font-size: 11px; background:#eef2ff; color:#3730a3; padding:4px 8px; border-radius:999px; border:1px solid #c7d2fe; }

    .toolbar{ display:flex; gap:8px; margin-top:8px; }
    .btn{
      flex:1; border:1px solid var(--panel-border);
      border-radius:12px; padding:10px 12px;
      background:var(--accent); color:#fff; font-weight:700; cursor:pointer;
      box-shadow: inset 0 -4px 8px rgba(0,0,0,.15);
    }
    .btn.secondary{ background:#f3f4f6; color:#111827; border-color:var(--panel-border-soft); }
    .btn:focus-visible{ outline:none; box-shadow:0 0 0 3px var(--ring); }

    /* 🎨 Find Video 버튼만 파란 배경 + 검은 글씨 */
    #btnFind{
      background:#98B7FF;          /* blue-100 */
      color:#111827;               /* black-ish */
      border-color:#bfdbfe;        /* blue-200 */
    }
    #btnFind:hover{ background:#cfe3ff; }

    .body{ overflow:auto; padding:12px; -webkit-overflow-scrolling:touch; }
    .section{ margin-top:10px; }
    .section-title{ font-size:12px; font-weight:700; opacity:.9; letter-spacing:.5px; margin:0 2px 8px; }
    .divider{ height:1px; background:linear-gradient(90deg,transparent,var(--panel-border),transparent); margin:6px 0 12px; }

    /* Logs */
    details.logs{ border:1px dashed var(--panel-border-soft); border-radius:12px; padding:6px 8px; margin-bottom:10px; }
    details.logs summary{ cursor:pointer; font-size:12px; color:var(--muted); }
    .log{ margin-top:6px; font-size:12px; color:#374151; max-height:120px; overflow:auto; white-space:pre-line; }

    /* Images */
    .overlay{ display:none; position:relative; border-radius:16px; overflow:hidden; border:1px solid var(--panel-border-soft); background:#f9fafb; }
    .overlay.show{ display:block; }
    .overlay img{ width:100%; height:auto; display:block; max-height:32vh; object-fit:contain; }

    /* Analysis */
    .analysis-row{ margin:12px 0 16px; }
    .analysis-head{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; }
    .analysis-title{ font-weight:700; font-size:14px; }
    .analysis-sub{ font-size:12px; color: var(--muted); }
    .result-box{
      margin-top:10px; border:1px solid var(--panel-border);
      border-radius:14px; padding:12px 14px;
      background:linear-gradient(180deg, rgba(0,0,0,.02), rgba(0,0,0,.0));
      display:flex; align-items:center; justify-content:space-between; gap:14px;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.65);
    }
    .result-left{ font-size:13px; }
    .result-left b{ font-weight:800; }
    .result-right{ font-size:12px; text-align:right; color:#111827; }
    .label-synthetic{ color:var(--danger); font-weight:800; }
    .label-natural{ color:var(--success); font-weight:800; }

    .summary-line{ margin-top:12px; font-weight:800; font-size:13px; }

    /* 🔴 Analysis 내부 텍스트의 'synthetic' 강조용 */
    .em-syn{ color: var(--danger); font-weight:800; }

    /* Segmented */
    .segment{ margin-top:14px; display:flex; gap:16px; justify-content:center; }
    .seg-btn{
      min-width:140px; border-radius:999px; padding:12px 18px;
      border:1px solid var(--panel-border); background:#f3f4f6; color:#6b7280; font-weight:800;
      box-shadow: inset 0 -8px 16px rgba(0,0,0,.06), 0 10px 24px rgba(0,0,0,.08);
    }
    .seg-btn.active{ background:#9ca3af; color:#111827; border-color:#9ca3af; }

    /* 🔵 Synthetic 버튼은 항상 더 진하게(기본/활성 모두) */
    .segment .seg-btn.synthetic{ background:#9aa0aa; color:#111827; border-color:#8b939d; }
    .segment .seg-btn.synthetic.active{ background:#6b7280; color:#ffffff; border-color:#4b5563; }
  `;
  shadow.appendChild(style);

  // ===== Panel =====
  const wrap = document.createElement("div");
  wrap.className = "panel";
  wrap.innerHTML = `
    <div class="head">
      <div class="title-row">
        <div class="title">Syns-killer</div>
        <span class="badge" id="badge">idle</span>
      </div>
      <div class="toolbar">
        <button id="btnFind" class="btn">Find Video</button>
        <button id="btnAnalyze" class="btn secondary">Analyze</button>
      </div>
    </div>

    <div class="body">
      <details class="logs"><summary>Logs</summary><div class="log" id="log"></div></details>

      <div class="section" id="imgSection">
        <div class="section-title">IMAGES</div>
        <div class="overlay" id="overlay"><img id="overlayImg" alt="overlay"/></div>
      </div>

      <div class="section">
        <div>ㅤ</div>
        <div class="section-title">ANALYSIS</div>
        <div class="divider"></div>
        <div id="analysis"><div style="font-size:12px;color:#6b7280">Click <b>Analyze</b> to view results.</div></div>
      </div>
    </div>
  `;
  shadow.appendChild(wrap);

  // ===== Refs =====
  const $badge = shadow.getElementById("badge");
  const $log = shadow.getElementById("log");
  const $analysis = shadow.getElementById("analysis");
  const $overlay = shadow.getElementById("overlay");
  const $overlayImg = shadow.getElementById("overlayImg");

  const log = (m) => ($log.textContent = `[${new Date().toLocaleTimeString()}] ${m}\n` + $log.textContent);

  // ===== Background helpers =====
  const bgFetchList = () => new Promise((res) => chrome.runtime.sendMessage({ type: "FETCH_LIST" }, (r) => res(r)));
  const bgFetchJSON = () => new Promise((res) => chrome.runtime.sendMessage({ type: "FETCH_JSON" }, (r) => res(r)));
  const bgFetchImageDataUrl = (name) => new Promise((res) => chrome.runtime.sendMessage({ type: "FETCH_IMAGE", name }, (r) => res(r)));

  // ===== Ping background =====
  chrome.runtime.sendMessage({ type: "PING" }, (res) => {
    if (res?.ok) { $badge.textContent = "ready"; log("Background connected."); }
    else { $badge.textContent = "error"; log("Background not responding."); }
  });

  // ===== Find Video — previous version AS-IS =====
  function findVideoCandidate() {
    const selectors = [
      "video",
      "#movie_player video",
      "#shorts-player video",
      "ytd-player video",
      "ytd-reel-video-renderer video",
      ".html5-video-player video",
      "video[src]"
    ];
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    const containers = ["#movie_player", "#shorts-player", "ytd-player", "ytd-watch-flexy", ".html5-video-player"];
    for (const cs of containers) {
      const el = document.querySelector(cs);
      if (el) return el;
    }
    return null;
  }
  function centerElement(el) {
    try { el.scrollIntoView({ behavior: "smooth", block: "center" }); }
    catch { const r = el.getBoundingClientRect(); const top = r.top + pageYOffset - (innerHeight/2 - r.height/2); scrollTo({ top, behavior:"smooth" }); }
  }
  shadow.getElementById("btnFind").onclick = () => {
    const target = findVideoCandidate();
    if (target) { centerElement(target); log("Video element found and centered."); }
    else { scrollTo({ top: (document.body.scrollHeight - innerHeight)/2, behavior:"smooth" }); log("No video found. Scrolled to page center."); }
  };

  // ===== Analysis renderer =====
  function renderAnalysisUI(raw) {
    const findLike = (obj, key) => {
      const norm = (s) => String(s).toLowerCase().replace(/[\s_-]+/g,'');
      const k = Object.keys(obj || {}).find(x => norm(x) === norm(key));
      return k ? obj[k] : undefined;
    };
    const root = (() => {
      if (!raw || typeof raw !== "object") return {};
      return findLike(raw, "syns-killer result") ?? raw;
    })();

    const jsonDesc = findLike(root, "descriptions") || {};
    const DESCR = {
      "edge feature": jsonDesc["edge feature"] || "Boundaries and outlines of objects.",
      "texture feature": jsonDesc["texture feature"] || "Surface patterns and regularity.",
      "etc feature": jsonDesc["etc feature"] || "Color, brightness, and overall structure."
    };

    const normLabel = (s) => {
      const t = String(s || "").toLowerCase();
      if (["fake","synthetic","gen","syn"].some(x=>t.includes(x))) return "Synthetic";
      if (["real","natural","auth"].some(x=>t.includes(x))) return "Natural";
      return "Synthetic";
    };
    const fmt = (x) => (Number.isFinite(x) ? (Math.round(x*100)/100).toFixed(2) : "—");

    const edge = findLike(root, "edge feature");
    const text = findLike(root, "texture feature");
    const etc  = findLike(root, "etc feature");
    const summary = findLike(root, "summary") || raw?.summary;

    const finalLabel = findLike(root, "final label") || (() => {
      const arr = [edge, text, etc].filter(Boolean).map(x => normLabel(x?.label));
      const syn = arr.filter(x => x==="Synthetic").length;
      const nat = arr.filter(x => x==="Natural").length;
      return (syn >= nat) ? "Synthetic" : "Natural";
    })();

    // render
    $analysis.innerHTML = "";

    const addFeature = (title, desc, data) => {
      if (!data) return;
      const label = normLabel(data.label);
      const conf = fmt(data.confidence);
      const row = document.createElement("div");
      row.className = "analysis-row";
      row.innerHTML = `
        <div class="analysis-head">
          <div class="analysis-title">${title}</div>
          <div class="analysis-sub">${desc}</div>
        </div>
        <div class="result-box">
          <div class="result-left">
            Classification Result:
            <span class="${label==='Synthetic' ? 'label-synthetic':'label-natural'}">${label}</span>
          </div>
          <div class="result-right">confidence<br>(${conf})</div>
        </div>
      `;
      $analysis.appendChild(row);
    };

    addFeature("Edge Feature", DESCR["edge feature"], edge);
    addFeature("Texture Feature", DESCR["texture feature"], text);
    addFeature("ETC Feature", DESCR["etc feature"], etc);

    if (summary) {
      const s = document.createElement("div");
      s.className = "summary-line";
      s.textContent = ">>> Summary: " + String(summary);
      $analysis.appendChild(s);
    }

    // Segmented buttons (자체 클래스로 natural/synthetic 지정)
    const seg = document.createElement("div");
    seg.className = "segment";
    seg.innerHTML = `
      <button class="seg-btn natural ${finalLabel==='Natural' ? 'active' : ''}">Natural</button>
      <button class="seg-btn synthetic ${finalLabel==='Synthetic' ? 'active' : ''}">Synthetic</button>
    `;
    $analysis.appendChild(seg);

    // 🔴 버튼을 제외한 analysis 내 모든 'synthetic' 텍스트를 빨간색으로 하이라이트
    highlightSynthetic($analysis);
  }

  // 텍스트 노드에서 'synthetic' 단어만 span.em-syn 으로 감싸기 (segment 버튼 영역 제외)
  function highlightSynthetic(container) {
    const walker = document.createTreeWalker(
      container,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
          // 버튼 영역 제외
          if (node.parentElement && node.parentElement.closest('.segment')) {
            return NodeFilter.FILTER_REJECT;
          }
          return /synthetic/i.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      }
    );
    const toProcess = [];
    let n; while ((n = walker.nextNode())) toProcess.push(n);
    for (const textNode of toProcess) {
      const span = document.createElement('span');
      span.innerHTML = textNode.nodeValue.replace(/(synthetic)/gi, '<span class="em-syn">$1</span>');
      textNode.parentNode.replaceChild(span, textNode);
    }
  }

  // ===== Analyze: render + start slideshow (2s) =====
  let slideshowStarted = false;
  shadow.getElementById("btnAnalyze").onclick = async () => {
    log("Analyze clicked. Loading local JSON...");
    $badge.textContent = "working";
    try {
      const j = await bgFetchJSON();
      if (!j?.ok) throw new Error(j?.error || "bg fetch failed");
      renderAnalysisUI(j.data);
      log("Rendered ANALYSIS.");

      if (!slideshowStarted) {
        $overlay.classList.add("overlay", "show"); // show Images section
        await setupOverlaySlideshow();             // start slideshow
        slideshowStarted = true;
      }
      $badge.textContent = "ready";
    } catch (err) {
      log("Analyze failed: " + err.message);
      $badge.textContent = "error";
    }
  };

  // ===== Slideshow (same behavior as 이전 버전: every 2s) =====
  async function setupOverlaySlideshow() {
    let imgList = [];
    let idx = 0;
  
    async function loadList() {
      const r = await bgFetchList();
      if (!r?.ok) {
        log("overlay/list failed: " + r?.error);
        return;
      }
      imgList = Array.isArray(r.data.files) ? r.data.files : [];
      idx = 0;
      log(`Loaded ${imgList.length} images for overlay.`);
    }
  
    async function tick() {
      if (!imgList.length) return;
      const name = imgList[idx % imgList.length];
      const r = await bgFetchImageDataUrl(name);
      if (r?.ok) {
        $overlayImg.src = r.dataUrl;
      } else {
        log("image fetch failed: " + r?.error);
      }
      idx++;
    }
  
    await loadList();
    await tick();
    setInterval(tick, 1000);
    // 필요시: setInterval(loadList, 5000);
  }
  
})();
