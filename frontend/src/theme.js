import { useEffect, useState } from "react";

const KEY = "kkh-theme";
const read = () => {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" || v === "system" ? v : "system";
  } catch {
    return "system";
  }
};

export function resolve(mode) {
  if (mode === "system") return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  return mode;
}

// mode: "light" | "dark" | "system". Sets <html data-theme="light|dark"> and follows the OS while in system mode.
export function useTheme() {
  const [mode, setMode] = useState(read);
  useEffect(() => {
    const apply = () => (document.documentElement.dataset.theme = resolve(mode));
    apply();
    try { localStorage.setItem(KEY, mode); } catch { /* storage blocked: still works for this visit */ }
    if (mode !== "system") return undefined;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [mode]);
  return [mode, setMode];
}
