(() => {
  const STORAGE_KEY = "regression-playground-theme";
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: light)");

  function savedTheme() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (_) {
      return null;
    }
  }

  function systemTheme() {
    return media.matches ? "light" : "dark";
  }

  function updateControls(theme) {
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      const nextTheme = theme === "dark" ? "light" : "dark";
      const label = button.querySelector("[data-theme-label]");
      const icon = button.querySelector("[data-theme-icon]");

      if (label) label.textContent = nextTheme === "light" ? "Light" : "Dark";
      if (icon) icon.textContent = nextTheme === "light" ? "☀" : "☾";

      button.setAttribute("aria-label", `Switch to ${nextTheme} theme`);
      button.setAttribute("title", `Switch to ${nextTheme} theme`);
      button.setAttribute("aria-pressed", theme === "light" ? "true" : "false");
    });
  }

  function applyTheme(theme, { persist = false, announce = true } = {}) {
    const normalized = theme === "light" ? "light" : "dark";
    root.dataset.theme = normalized;
    root.style.colorScheme = normalized;

    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, normalized);
      } catch (_) {
        // The theme still works even when browser storage is unavailable.
      }
    }

    updateControls(normalized);

    if (announce) {
      window.dispatchEvent(new CustomEvent("themechange", {
        detail: { theme: normalized },
      }));
    }
  }

  // Apply before the page paints so navigation does not flash the wrong theme.
  applyTheme(savedTheme() || systemTheme(), { announce: false });

  document.addEventListener("DOMContentLoaded", () => {
    updateControls(root.dataset.theme || "dark");

    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        const current = root.dataset.theme === "light" ? "light" : "dark";
        applyTheme(current === "dark" ? "light" : "dark", { persist: true });
      });
    });
  });

  // Follow OS changes only until the user has explicitly chosen a theme.
  media.addEventListener?.("change", () => {
    if (!savedTheme()) applyTheme(systemTheme());
  });

  window.addEventListener("storage", (event) => {
    if (event.key !== STORAGE_KEY) return;
    const value = event.newValue;
    applyTheme(value === "light" || value === "dark" ? value : systemTheme(), {
      announce: true,
    });
  });
})();
