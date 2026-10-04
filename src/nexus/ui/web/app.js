/* NEXUS -- Quick Menu web frontend logic.
   Talks to ui/server.py over fetch() + one WebSocket. No framework, no
   build step (V4.0 Phase 42). */

(() => {
  "use strict";

  const PLACEHOLDER = "Type a command...";

  const chat = document.getElementById("chat");
  const statusBar = document.getElementById("status-bar");
  const inputField = document.getElementById("input-field");
  const inputForm = document.getElementById("input-form");
  const sendBtn = document.getElementById("send-btn");
  const mainMenu = document.getElementById("main-menu");

  let processing = false;
  let currentNexusBubble = null;

  // ------------------------------------------------------------------
  // pywebview bridge helpers -- all no-ops when run in a plain browser
  // (e.g. during development, before Phase 43 wraps this in pywebview).
  // ------------------------------------------------------------------
  function hasPywebview() {
    return typeof window.pywebview !== "undefined" && window.pywebview.api;
  }

  async function pywebviewCall(method, ...args) {
    if (!hasPywebview() || typeof window.pywebview.api[method] !== "function") {
      return null;
    }
    try {
      return await window.pywebview.api[method](...args);
    } catch (e) {
      console.error(`pywebview.api.${method} failed:`, e);
      return null;
    }
  }

  // ------------------------------------------------------------------
  // Chat rendering
  // ------------------------------------------------------------------
  function appendMessage(label, text, variant) {
    const div = document.createElement("div");
    div.className = `msg msg--${variant}`;
    if (label) {
      const strong = document.createElement("span");
      strong.className = "msg__label";
      strong.textContent = label;
      div.appendChild(strong);
    }
    div.appendChild(document.createTextNode(text));
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  function setStatus(text, color) {
    statusBar.textContent = text;
    statusBar.style.color = color || "var(--thinking)";
  }

  // ------------------------------------------------------------------
  // WebSocket
  // ------------------------------------------------------------------
  let ws = null;

  function connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/api/ws`);

    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      handleWsMessage(msg);
    });

    ws.addEventListener("close", () => {
      setStatus("Disconnected -- retrying...", "var(--error)");
      setTimeout(connectWebSocket, 2000);
    });
  }

  function handleWsMessage(msg) {
    switch (msg.type) {
      case "token":
        if (!currentNexusBubble) {
          currentNexusBubble = appendMessage("NEXUS: ", "", "nexus");
        }
        currentNexusBubble.appendChild(document.createTextNode(msg.text));
        chat.scrollTop = chat.scrollHeight;
        break;

      case "done":
        currentNexusBubble = null;
        setProcessing(false);
        setStatus("Ready", "var(--success)");
        break;

      case "confirm":
        showConfirm(msg.id, msg.title, msg.details);
        break;

      case "train_complete":
        appendMessage("NEXUS: ", msg.reply, "nexus");
        break;

      default:
        console.warn("Unknown WS message type:", msg);
    }
  }

  function setProcessing(value) {
    processing = value;
    sendBtn.disabled = value;
  }

  // ------------------------------------------------------------------
  // Sending a command
  // ------------------------------------------------------------------
  inputForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendCommand();
  });

  function sendCommand() {
    const text = inputField.value.trim();
    if (!text || processing) return;

    inputField.value = "";
    appendMessage("You: ", text, "user");
    setProcessing(true);
    setStatus("Processing...", "var(--thinking)");

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "command", text }));
    } else {
      // Fallback to the plain HTTP endpoint if the socket isn't up yet.
      fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
        .then((r) => r.json())
        .then((data) => {
          appendMessage("NEXUS: ", data.reply, "nexus");
          setProcessing(false);
          setStatus("Ready", "var(--success)");
        })
        .catch((e) => {
          appendMessage("NEXUS: ", `Error: ${e}`, "error");
          setProcessing(false);
          setStatus("Error", "var(--error)");
        });
    }
  }

  // ------------------------------------------------------------------
  // Quick-action pills
  // ------------------------------------------------------------------
  document.querySelectorAll(".pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      inputField.value = pill.dataset.insert;
      inputField.focus();
    });
  });

  // ------------------------------------------------------------------
  // Window controls (real behaviour wired up once pywebview is present --
  // Phase 43's js_api exposes minimize/toggle_maximize/close)
  // ------------------------------------------------------------------
  document.getElementById("min-btn").addEventListener("click", () => pywebviewCall("minimize"));
  document.getElementById("max-btn").addEventListener("click", () => pywebviewCall("toggle_maximize"));
  document.getElementById("close-btn").addEventListener("click", () => pywebviewCall("close"));

  // ------------------------------------------------------------------
  // Dropdown menu
  // ------------------------------------------------------------------
  document.getElementById("menu-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    mainMenu.hidden = !mainMenu.hidden;
  });

  document.addEventListener("click", (e) => {
    if (!mainMenu.hidden && !mainMenu.contains(e.target) && e.target.id !== "menu-btn") {
      mainMenu.hidden = true;
    }
  });

  mainMenu.addEventListener("click", (e) => {
    const item = e.target.closest("[data-action]");
    if (item) {
      mainMenu.hidden = true;
      runMenuAction(item.dataset.action);
      return;
    }
    const themeItem = e.target.closest("[data-theme-name]");
    if (themeItem) {
      mainMenu.hidden = true;
      setTheme(themeItem.dataset.themeName);
    }
  });

  function runMenuAction(action) {
    switch (action) {
      case "settings": openSettingsModal(); break;
      case "reset-model": resetModel(); break;
      case "train": openTrainModal(); break;
      case "clear-chat": clearChat(); break;
      case "about": openAboutModal(); break;
    }
  }

  function clearChat() {
    chat.innerHTML = "";
    appendMessage("", "Chat cleared.", "info");
  }

  function resetModel() {
    appendMessage("NEXUS: ", "", "nexus");
    fetch("/api/model/reset", { method: "POST" })
      .then((r) => r.json())
      .then((data) => appendMessage("NEXUS: ", data.reply, "nexus"))
      .catch((e) => appendMessage("NEXUS: ", `Error: ${e}`, "error"));
  }

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------
  function setTheme(name) {
    document.documentElement.dataset.theme = name;
    fetch("/api/theme", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).catch(() => {});
    appendMessage("", `Theme changed to: ${name}`, "info");
  }

  function loadSavedTheme() {
    fetch("/api/theme")
      .then((r) => r.json())
      .then((data) => {
        document.documentElement.dataset.theme = data.name;
      })
      .catch(() => {});
  }

  // ------------------------------------------------------------------
  // Settings modal
  // ------------------------------------------------------------------
  const settingsModal = document.getElementById("settings-modal");
  let selectedFolder = null;

  function openSettingsModal() {
    Promise.all([
      fetch("/api/settings").then((r) => r.json()),
      fetch("/api/permissions").then((r) => r.json()),
      fetch("/api/deps").then((r) => r.json()),
    ]).then(([settings, perms, deps]) => {
      document.getElementById("setting-max_tokens").value = settings.max_tokens ?? "";
      document.getElementById("setting-n_ctx").value = settings.n_ctx ?? "";
      document.getElementById("setting-n_threads").value = settings.n_threads ?? "";

      const list = document.getElementById("folder-list");
      list.innerHTML = "";
      selectedFolder = null;
      (perms.paths || []).forEach((p) => {
        const li = document.createElement("li");
        li.textContent = p;
        li.addEventListener("click", () => {
          list.querySelectorAll("li").forEach((el) => el.classList.remove("selected"));
          li.classList.add("selected");
          selectedFolder = p;
        });
        list.appendChild(li);
      });

      const depsStatus = document.getElementById("deps-status");
      if (deps.missing_required && deps.missing_required.length) {
        depsStatus.textContent = `Missing required: ${deps.missing_required.map((d) => d[0]).join(", ")}`;
      } else if (deps.missing_optional && deps.missing_optional.length) {
        depsStatus.textContent = `Missing optional: ${deps.missing_optional.map((d) => d[0]).join(", ")}`;
      } else {
        depsStatus.textContent = "All packages installed.";
      }
    });

    settingsModal.showModal();
  }

  document.getElementById("add-folder-btn").addEventListener("click", async () => {
    const list = document.getElementById("folder-list");
    let folder = await pywebviewCall("pick_folder");
    if (!folder) folder = window.prompt("Folder path to allow:");
    if (!folder) return;

    const existing = [...list.querySelectorAll("li")].some((li) => li.textContent === folder);
    if (existing) return;

    const li = document.createElement("li");
    li.textContent = folder;
    li.addEventListener("click", () => {
      list.querySelectorAll("li").forEach((el) => el.classList.remove("selected"));
      li.classList.add("selected");
      selectedFolder = folder;
    });
    list.appendChild(li);
  });

  document.getElementById("remove-folder-btn").addEventListener("click", () => {
    const list = document.getElementById("folder-list");
    const selected = list.querySelector("li.selected");
    if (selected) selected.remove();
    selectedFolder = null;
  });

  document.getElementById("settings-save-btn").addEventListener("click", () => {
    const settingsBody = {
      max_tokens: parseInt(document.getElementById("setting-max_tokens").value, 10) || undefined,
      n_ctx: parseInt(document.getElementById("setting-n_ctx").value, 10) || undefined,
      n_threads: parseInt(document.getElementById("setting-n_threads").value, 10) || undefined,
    };
    const paths = [...document.querySelectorAll("#folder-list li")].map((li) => li.textContent);

    Promise.all([
      fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settingsBody),
      }),
      fetch("/api/permissions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths }),
      }),
    ])
      .then(() => {
        settingsModal.close();
        appendMessage("", "Settings saved.", "info");
      })
      .catch((e) => appendMessage("", `Settings not saved: ${e}`, "error"));
  });

  document.getElementById("install-deps-btn").addEventListener("click", () => {
    const status = document.getElementById("deps-status");
    status.textContent = "Installing... please wait";
    fetch("/api/deps/install", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        status.textContent = data.message;
      })
      .catch((e) => {
        status.textContent = `Error: ${e}`;
      });
  });

  // ------------------------------------------------------------------
  // About modal
  // ------------------------------------------------------------------
  const aboutModal = document.getElementById("about-modal");

  function openAboutModal() {
    fetch("/api/about")
      .then((r) => r.json())
      .then((data) => {
        const rows = document.getElementById("about-rows");
        rows.innerHTML = "";
        const entries = [
          ["Version", data.version],
          ["Theme", data.theme],
          ["Model loaded", data.model_loaded ? "Yes" : "No"],
          ["LoRA applied", data.lora_applied ? "Yes" : "No"],
          ["Memories", String(data.memories)],
        ];
        entries.forEach(([label, value]) => {
          const row = document.createElement("div");
          const dt = document.createElement("dt");
          dt.textContent = label;
          const dd = document.createElement("dd");
          dd.textContent = value;
          row.appendChild(dt);
          row.appendChild(dd);
          rows.appendChild(row);
        });
      });
    aboutModal.showModal();
  }

  // ------------------------------------------------------------------
  // Train modal
  // ------------------------------------------------------------------
  const trainModal = document.getElementById("train-modal");

  function openTrainModal() {
    document.getElementById("train-folder").value = "";
    trainModal.showModal();
  }

  document.getElementById("train-browse-btn").addEventListener("click", async () => {
    let folder = await pywebviewCall("pick_folder");
    if (!folder) folder = window.prompt("Dataset folder path:");
    if (folder) document.getElementById("train-folder").value = folder;
  });

  document.getElementById("train-start-btn").addEventListener("click", () => {
    const folder = document.getElementById("train-folder").value.trim();
    if (!folder) return;
    const quick = document.getElementById("train-quick").checked;

    trainModal.close();
    appendMessage("NEXUS: ", "", "nexus");
    appendMessage(
      "",
      "Experimental feature: training produces a real PEFT adapter, but NEXUS's " +
        "inference engine (llama.cpp) can't load it directly -- it needs a manual " +
        "GGUF conversion first. See README for details.",
      "info"
    );
    appendMessage("", `Training on: ${folder}`, "info");
    appendMessage("", "This may take a while...", "thinking");

    fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, quick }),
    }).catch((e) => appendMessage("NEXUS: ", `Training error: ${e}`, "error"));
  });

  // ------------------------------------------------------------------
  // Confirm dialog (destructive actions, pushed over the WebSocket)
  // ------------------------------------------------------------------
  const confirmModal = document.getElementById("confirm-modal");
  let pendingConfirmId = null;

  function showConfirm(id, title, details) {
    pendingConfirmId = id;
    document.getElementById("confirm-title").textContent = title;
    document.getElementById("confirm-details").textContent = details;
    confirmModal.showModal();
  }

  function answerConfirm(answer) {
    if (pendingConfirmId && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "confirm_response", id: pendingConfirmId, answer }));
    }
    pendingConfirmId = null;
    confirmModal.close();
  }

  document.getElementById("confirm-yes-btn").addEventListener("click", () => answerConfirm(true));
  document.getElementById("confirm-no-btn").addEventListener("click", () => answerConfirm(false));

  // ------------------------------------------------------------------
  // Modal close buttons (generic)
  // ------------------------------------------------------------------
  document.querySelectorAll("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => btn.closest("dialog").close());
  });

  // ------------------------------------------------------------------
  // Input placeholder + focus handling (native placeholder attribute,
  // not the old Tkinter UI's manual string-matching hack)
  // ------------------------------------------------------------------
  inputField.setAttribute("placeholder", PLACEHOLDER);

  // ------------------------------------------------------------------
  // Model-load status polling (no push channel for this yet -- cheap
  // enough to poll a couple of times a second until it flips)
  // ------------------------------------------------------------------
  function pollModelStatus() {
    fetch("/api/about")
      .then((r) => r.json())
      .then((data) => {
        if (data.model_loaded) {
          setStatus("Ready -- model loaded", "var(--success)");
        } else {
          setStatus("Loading model...", "var(--thinking)");
          setTimeout(pollModelStatus, 1500);
        }
      })
      .catch(() => setTimeout(pollModelStatus, 3000));
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------
  appendMessage("", "Welcome to NEXUS", "nexus");
  appendMessage("", "Use slash commands for quick actions:", "info");
  appendMessage("", "  /task  /remember  /recall  /help", "info");
  appendMessage("", "Or just type naturally and NEXUS will understand.", "info");

  loadSavedTheme();
  connectWebSocket();
  pollModelStatus();
  inputField.focus();
})();
