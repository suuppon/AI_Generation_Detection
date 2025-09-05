// // background.js (MV3 service worker)

// // 로컬 Flask 서버 주소(필요시 127.0.0.1 ↔ localhost 중 하나로 통일)
// const BASE = "http://127.0.0.1:5000";

// // 선택: 설치/업데이트 시 로그
// chrome.runtime.onInstalled.addListener(() => {
//   console.log("[bg] installed");
// });

// // 메시지 라우터
// chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
//   // 헬스체크
//   if (msg?.type === "PING") {
//     sendResponse({ ok: true });
//     return; // sync
//   }

//   // ----- 프록시: 이미지 목록 -----
//   if (msg?.type === "FETCH_LIST") {
//     fetch(`${BASE}/overlay/list`)
//       .then((r) => {
//         if (!r.ok) throw new Error(`HTTP ${r.status}`);
//         return r.json();
//       })
//       .then((data) => sendResponse({ ok: true, data }))
//       .catch((err) => sendResponse({ ok: false, error: String(err) }));
//     return true; // async
//   }

//   // ----- 프록시: JSON -----
//   if (msg?.type === "FETCH_JSON") {
//     fetch(`${BASE}/read-json`)
//       .then((r) => {
//         if (!r.ok) throw new Error(`HTTP ${r.status}`);
//         return r.json();
//       })
//       .then((data) => sendResponse({ ok: true, data }))
//       .catch((err) => sendResponse({ ok: false, error: String(err) }));
//     return true; // async
//   }

//   // ----- 프록시: 이미지 -> data URL로 변환 후 전달 -----
//   if (msg?.type === "FETCH_IMAGE" && msg.name) {
//     fetch(`${BASE}/overlay/files/${encodeURIComponent(msg.name)}`)
//       .then((r) => {
//         if (!r.ok) throw new Error(`HTTP ${r.status}`);
//         return r.blob();
//       })
//       .then((blob) => {
//         return new Promise((resolve, reject) => {
//           const fr = new FileReader();
//           fr.onload = () => resolve(fr.result);
//           fr.onerror = reject;
//           fr.readAsDataURL(blob); // data URL 변환
//         });
//       })
//       .then((dataUrl) => sendResponse({ ok: true, dataUrl }))
//       .catch((err) => sendResponse({ ok: false, error: String(err) }));
//     return true; // async
//   }

//   // 다른 메시지는 무시
//   return false;
// });

// background.js (MV3 service worker)

// 로컬 Flask 서버 주소(필요시 localhost ↔ 127.0.0.1 중 하나로 통일)
const BASE = "http://127.0.0.1:5000";

chrome.runtime.onInstalled.addListener(() => {
  console.log("[bg] installed");
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "PING") {
    sendResponse({ ok: true });
    return;
  }

  if (msg?.type === "FETCH_LIST") {
    fetch(`${BASE}/overlay/list`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (msg?.type === "FETCH_JSON") {
    fetch(`${BASE}/read-json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (msg?.type === "FETCH_IMAGE" && msg.name) {
    fetch(`${BASE}/overlay/files/${encodeURIComponent(msg.name)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.blob();
      })
      .then((blob) => new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onload = () => resolve(fr.result);
        fr.onerror = reject;
        fr.readAsDataURL(blob);
      }))
      .then((dataUrl) => sendResponse({ ok: true, dataUrl }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  return false;
});
