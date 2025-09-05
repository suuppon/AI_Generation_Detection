const btn = document.getElementById("toggle");

async function getState() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ overlay: true }, (o) => resolve(o.overlay));
  });
}
async function setState(v) {
  return new Promise((resolve) => {
    chrome.storage.sync.set({ overlay: v }, resolve);
  });
}
async function updateUI() {
  const on = await getState();
  btn.textContent = on ? "On" : "Off";
}
btn.onclick = async () => {
  const on = await getState();
  await setState(!on);
  // 현재 탭 새로고침 없이도 content script가 읽도록 간단히 리로드 유도(선택)
  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    if (tab?.id) chrome.tabs.reload(tab.id);
  });
  updateUI();
};
updateUI();
