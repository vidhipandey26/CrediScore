// -------------------------
// TruthLens UI Controller
// -------------------------

const root = document.documentElement;
const themeToggleBtn = document.getElementById("theme-toggle");
const langSelect = document.getElementById("lang");
const canvas = document.getElementById("signal-canvas");

// -------------------------
// THEME TOGGLER
// -------------------------
function applyTheme(theme) {
  if (theme === "light") root.classList.add("light");
  else root.classList.remove("light");

  localStorage.setItem("theme", theme);

  // Adjust text/icon contrast dynamically
  const chips = document.querySelectorAll(".chip, .btn, .mode, .icon-btn, .icon");
  chips.forEach((el) => {
    el.style.transition = "color 0.3s ease, background 0.3s ease, box-shadow 0.3s ease";
    if (theme === "light") {
      el.style.color = "#0B2B33";
    } else {
      el.style.color = "#E9F3F2";
    }
  });
}

function toggleTheme() {
  const current = root.classList.contains("light") ? "dark" : "light";
  applyTheme(current);
}

// Load preferred theme
document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("theme") || "dark";
  applyTheme(savedTheme);
});

themeToggleBtn?.addEventListener("click", toggleTheme);

// -------------------------
// BACKGROUND ANIMATION
// -------------------------
if (canvas) {
  const ctx = canvas.getContext("2d");
  let w, h;
  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  const particles = Array.from({ length: 100 }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.5,
    vy: (Math.random() - 0.5) * 0.5,
    r: 1.1 + Math.random() * 1.4,
  }));

  function drawParticles() {
    ctx.clearRect(0, 0, w, h);
    const theme = root.classList.contains("light") ? "light" : "dark";
    const teal = theme === "light" ? "rgba(0,126,135,0.3)" : "rgba(0,198,198,0.4)";
    const gold = theme === "light" ? "rgba(203,175,69,0.3)" : "rgba(255,216,87,0.35)";

    // Connection lines
    for (let i = 0; i < particles.length; i++) {
      const a = particles[i];
      for (let j = i + 1; j < particles.length; j++) {
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          const alpha = 1 - dist / 130;
          ctx.strokeStyle = Math.random() > 0.5 ? teal : gold;
          ctx.globalAlpha = alpha * 0.4;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    ctx.globalAlpha = 1;
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = Math.random() > 0.5 ? teal : gold;
      ctx.fill();
    }

    requestAnimationFrame(drawParticles);
  }

  drawParticles();
}

// -------------------------
// ANALYSIS SIMULATION
// -------------------------
const analyzeBtn = document.getElementById("analyze");
const queryInput = document.getElementById("query");

function setConfidence(val) {
  val = Math.max(0, Math.min(100, val));
  const g = document.querySelector(".gauge");
  if (!g) return;
  g.style.setProperty("--v", val);
  g.dataset.label = val + "%";

  // animate smooth gauge transition
  g.animate([{ transform: "scale(0.95)" }, { transform: "scale(1)" }], { duration: 250, easing: "ease-out" });
}

function analyzeInput() {
  const input = queryInput?.value.trim();
  if (!input) return;
  const score = Math.round(40 + (input.length % 50) + Math.random() * 30);
  setConfidence(score);
  const out = document.getElementById("analysis-out");
  out.innerHTML = `<b>AI Confidence:</b> ${score}% — <span style="color:${score > 60 ? 'var(--gold)' : 'var(--teal)'}">${score > 60 ? 'Likely credible' : 'Needs verification'}</span>`;
}

analyzeBtn?.addEventListener("click", analyzeInput);
queryInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    analyzeInput();
  }
});

// -------------------------
// MODES SWITCH HANDLER
// -------------------------
const modeButtons = document.querySelectorAll(".mode");
modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeButtons.forEach((m) => m.classList.remove("active"));
    btn.classList.add("active");
  });
});

// -------------------------
// UPLOAD / MIC / SETTINGS (stubs)
// -------------------------
document.getElementById("btn-upload")?.addEventListener("click", () => {
  alert("Upload stub: connect file input or backend upload.");
});
document.getElementById("btn-mic")?.addEventListener("click", () => {
  alert("Mic stub: use Web Speech API or Recorder.js integration.");
});
document.getElementById("btn-settings")?.addEventListener("click", () => {
  location.href = "account.html";
});

// -------------------------
// LANGUAGE SELECT (stub)
// -------------------------
langSelect?.addEventListener("change", (e) => {
  localStorage.setItem("lang", e.target.value);
  alert("Language changed to: " + e.target.value);
});

// -------------------------
// FOOTER YEAR
// -------------------------
const yearEl = document.getElementById("year");
if (yearEl) yearEl.textContent = new Date().getFullYear();
