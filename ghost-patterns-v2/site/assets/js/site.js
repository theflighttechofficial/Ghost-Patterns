/* Shared behaviour for every page: theme toggle, mobile menu, copy buttons,
   and redrawing charts when the theme changes. */
(function () {
  "use strict";
  const root = document.documentElement;

  // ---- theme: system by default, explicit choice remembered per browser
  const KEY = "ghostpatterns-theme";
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
  } catch (e) { /* storage blocked: follow the system theme */ }

  function currentTheme() {
    const set = root.getAttribute("data-theme");
    if (set) return set;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  const redrawers = [];
  window.onThemeChange = (fn) => { redrawers.push(fn); };
  function redraw() { redrawers.forEach((fn) => fn()); }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.querySelector(".theme-toggle");
    const label = () => {
      if (toggle) toggle.textContent = currentTheme() === "dark" ? "Light" : "Dark";
    };
    label();
    if (toggle) {
      toggle.addEventListener("click", () => {
        const next = currentTheme() === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
        label();
        redraw();
      });
    }

    // ---- mobile menu
    const nav = document.querySelector("nav");
    const menu = document.querySelector(".menu-toggle");
    if (menu && nav) {
      menu.addEventListener("click", () => {
        const open = nav.classList.toggle("open");
        menu.setAttribute("aria-expanded", String(open));
      });
    }

    // ---- copy buttons sit right before a <pre>
    document.querySelectorAll(".copy").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pre = btn.nextElementSibling;
        const done = () => {
          btn.textContent = "Copied";
          setTimeout(() => { btn.textContent = "Copy"; }, 1500);
        };
        const select = () => {
          const range = document.createRange();
          range.selectNodeContents(pre);
          const sel = getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        };
        try {
          navigator.clipboard.writeText(pre.innerText).then(done, select);
        } catch (e) { select(); }
      });
    });

    // ---- footer year
    const y = document.querySelector("[data-year]");
    if (y) y.textContent = new Date().getFullYear();
  });
})();
