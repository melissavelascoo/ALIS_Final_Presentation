"use strict";

/* Efectos de interfaz.
 * No accede a la cámara, fotogramas, canvas, imágenes ni red.
 * Toda la lógica funcional continúa intacta en app.js.
 */
(function enhanceALISAppearance() {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const reveals = Array.from(document.querySelectorAll("[data-reveal]"));
  if (!reduceMotion.matches && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries, io) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      }
    }, { threshold: 0.05, rootMargin: "0px 0px 100px 0px" });
    reveals.forEach((node) => observer.observe(node));
    document.body.classList.add("motion-ready");
  } else {
    reveals.forEach((node) => node.classList.add("is-visible"));
  }

  const switcher = document.querySelector("#switchDirectionBtn");
  switcher?.addEventListener("click", () => {
    if (reduceMotion.matches) return;
    switcher.classList.remove("switch-pop");
    void switcher.offsetWidth;
    switcher.classList.add("switch-pop");
  });
  switcher?.addEventListener("animationend", () => switcher.classList.remove("switch-pop"));

  // Progreso basado únicamente en el contador ya generado por app.js.
  const counter = document.querySelector("#sequenceCounter");
  const fill = document.querySelector("#sequenceProgressBar");
  if (counter && fill) {
    const refresh = () => {
      const [, step = "0", total = "0"] = counter.textContent.match(/(\d+)\s*\/\s*(\d+)/) || [];
      const current = Number(step);
      const length = Number(total);
      fill.style.width = `${length ? Math.min(100, current / length * 100) : 0}%`;
    };
    new MutationObserver(refresh).observe(counter, { childList: true, subtree: true, characterData: true });
    refresh();
  }

  const sourceText = document.querySelector("#sourceText");
  const charCount = document.querySelector("#charCount");
  sourceText?.addEventListener("input", () => {
    charCount?.classList.toggle("count-active", Boolean(sourceText.value.length));
  });

  const letter = document.querySelector("#recognizedChar");
  if (letter) {
    let previous = letter.textContent;
    new MutationObserver(() => {
      const current = letter.textContent;
      if (current === previous) return;
      previous = current;
      if (reduceMotion.matches || current === "—") return;
      letter.classList.remove("letter-celebrate");
      void letter.offsetWidth;
      letter.classList.add("letter-celebrate");
    }).observe(letter, { childList: true, characterData: true, subtree: true });
    letter.addEventListener("animationend", () => letter.classList.remove("letter-celebrate"));
  }

  if (!reduceMotion.matches && window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
    for (const [selector, x, y] of [
      [".translator", "--pointer-x", "--pointer-y"],
      ["--glow-x", "--glow-y"]
    ]) {
      for (const node of document.querySelectorAll(selector)) {
        node.addEventListener("pointermove", (event) => {
          const rect = node.getBoundingClientRect();
          node.style.setProperty(x, `${event.clientX - rect.left}px`);
          node.style.setProperty(y, `${event.clientY - rect.top}px`);
        }, { passive: true });
      }
    }
  }
})();
