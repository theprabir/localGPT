/**
 * LocalGPT frontend Alpine.js components.
 * Kept intentionally small: theme, chat shell (with composer), status.
 * No build step; served as plain JS.
 *
 * app.js must be loaded (deferred) BEFORE the Alpine bundle so that
 * Alpine.data(...) registrations exist when Alpine initializes.
 */
(function () {
  "use strict";

  // Templates reference components as localgpt.theme() / localgpt.shell() /
  // localgpt.status(). Alpine evaluates x-data expressions with `with(scope)`;
  // names missing from the component scope fall through to the global object,
  // so exposing this BEFORE Alpine runs (app.js is deferred ahead of alpine.js)
  // makes those expressions resolve.
  window.localgpt = {
    theme: themeComponent,
    shell: shellComponent,
    status: statusComponent,
  };

  // Also register plain names (x-data="shell()") for robustness.
  document.addEventListener("alpine:init", function () {
    Alpine.data("theme", themeComponent);
    Alpine.data("shell", shellComponent);
    Alpine.data("status", statusComponent);
  });

  function themeComponent() {
    return {
      theme: "dark",
      font: "default",
      initTheme() {
        // The pre-paint script in <head> already set data-theme; read it back
        // so the toggle buttons reflect the active theme.
        try {
          this.theme = localStorage.getItem("localgpt-theme") || "dark";
        } catch (e) {
          this.theme = "dark";
        }
        try {
          this.font = localStorage.getItem("localgpt-font") || "default";
        } catch (e) {
          this.font = "default";
        }
        this.applyFont();
      },
      applyTheme() {
        document.documentElement.dataset.theme = this.theme;
        try {
          localStorage.setItem("localgpt-theme", this.theme);
        } catch (e) {
          // ignore
        }
      },
      setTheme(value) {
        this.theme = value;
        this.applyTheme();
      },
      // Font families are defined in app.css as [data-font="..."] blocks that
      // set --font-ui; the whole interface reads from that one variable.
      applyFont() {
        const allowed = ["default", "lato", "garamond"];
        const value = allowed.includes(this.font) ? this.font : "default";
        document.documentElement.dataset.font = value;
        try {
          localStorage.setItem("localgpt-font", value);
        } catch (e) {
          // ignore
        }
      },
      setFont(value) {
        this.font = value;
        this.applyFont();
      },
    };
  }

  function shellComponent() {
    return {
      // Sidebar / conversation state
      conversations: [],
      activeConversationId: null,
      messages: [],
      pageTitle: "LocalGPT",
      sidebarOpen: false,

      // Composer state (single scope owns it — no nested composer component)
      composerText: "",
      attachedFile: null,
      attachedPreview: null,
      composerError: null,

      // Streaming state
      streaming: false,
      abortController: null,

      // Backend availability
      vlmReady: true,
      vlmStatusText: "",

      // Modals
      imageModal: false,
      imageModalSrc: "",
      imageModalTitle: "",

      init() {
        this.loadConversations();
        this.checkVlm();
        // Keyboard support: Escape dismisses the topmost open overlay.
        // (The mobile drawer is otherwise closeable only via its backdrop.)
        window.addEventListener("keydown", (e) => {
          if (e.key !== "Escape") return;
          if (this.imageModal) {
            this.imageModal = false;
          } else if (this.sidebarOpen) {
            this.sidebarOpen = false;
          }
        });
      },

      // ---------- Backend status ----------

      checkVlm() {
        fetch("/health")
          .then((r) => r.json())
          .then((data) => {
            const reachable = data.vlm && data.vlm.llama_server_reachable;
            this.vlmReady = !!reachable;
            this.vlmStatusText = reachable
              ? ""
              : "Model not connected. Start llama-server, then reload. See docs/setup.md.";
          })
          .catch(() => {
            this.vlmReady = false;
            this.vlmStatusText = "Cannot reach the LocalGPT server.";
          });
      },

      // ---------- Conversations ----------

      loadConversations() {
        fetch("/api/conversations?limit=50")
          .then((r) => r.json())
          .then((data) => {
            this.conversations = data.conversations || [];
          })
          .catch(() => {
            this.conversations = [];
          });
      },

      // Conversation history grouped by day (Today / Yesterday / date),
      // preserving the API's most-recent-first ordering.
      get conversationGroups() {
        const groups = [];
        const byLabel = new Map();
        const startOfToday = new Date();
        startOfToday.setHours(0, 0, 0, 0);
        const startOfYesterday = new Date(startOfToday);
        startOfYesterday.setDate(startOfYesterday.getDate() - 1);

        (this.conversations || []).forEach((item) => {
          const when = new Date(item.updated_at || item.created_at);
          let label;
          if (when >= startOfToday) {
            label = "Today";
          } else if (when >= startOfYesterday) {
            label = "Yesterday";
          } else {
            label = when.toLocaleDateString(undefined, {
              year: "numeric",
              month: "short",
              day: "numeric",
            });
          }
          let group = byLabel.get(label);
          if (!group) {
            group = { label: label, items: [] };
            byLabel.set(label, group);
            groups.push(group);
          }
          group.items.push(item);
        });

        return groups;
      },

      selectConversation(item) {
        if (this.streaming) this.stop();
        this.activeConversationId = item.id;
        this.messages = [];
        this.composerError = null;
        this.sidebarOpen = false;
        this.loadMessages();
      },

      newChat() {
        if (this.streaming) this.stop();
        this.activeConversationId = null;
        this.messages = [];
        this.composerText = "";
        this.composerError = null;
        this.clearAttachment();
        this.resetComposerHeight();
        this.pageTitle = "New chat";
        this.sidebarOpen = false;
        this.$refs.composerInput && this.$refs.composerInput.focus();
      },

      toggleSidebar() {
        this.sidebarOpen = !this.sidebarOpen;
      },

      loadMessages() {
        if (!this.activeConversationId) return;
        const base = "/api/conversations/" + this.activeConversationId;
        Promise.all([
          fetch(base + "/messages").then((r) => r.json()),
          fetch(base + "/attachments")
            .then((r) => (r.ok ? r.json() : []))
            .catch(() => []),
        ])
          .then(([msgData, attachments]) => {
            // attachment_id -> served URL, so history re-displays uploaded images.
            const attachMap = {};
            (attachments || []).forEach((a) => {
              attachMap[a.id] = "/api/uploads/" + a.stored_filename;
            });
            this.messages = (msgData.messages || [])
              .filter((m) => m.content || m.attachment_id) // hide the empty streaming placeholder row
              .map((m) => ({
                id: m.id,
                role: m.role,
                content: m.content,
                attachment_id: m.attachment_id || null,
                html: renderMessageHtml(
                  m.content,
                  m.role,
                  m.attachment_id ? attachMap[m.attachment_id] : null
                ),
              }));
            const conv = this.conversations.find(
              (c) => c.id === this.activeConversationId
            );
            this.pageTitle = conv ? conv.title : "Conversation";
          })
          .catch(() => {
            this.messages = [];
          });
      },

      // ---------- Sending / streaming ----------

      // Enter sends, Shift+Enter inserts a newline.
      // Handled in JS rather than with `@keydown.enter.exact.prevent`, because
      // Alpine's key guard suppresses the event when a non-key modifier such as
      // `.exact` is combined with a key modifier (the handler would never run).
      onComposerKeydown(e) {
        if (e.key !== "Enter" || e.shiftKey) return;
        if (e.isComposing || e.keyCode === 229) return; // IME composition
        e.preventDefault();
        this.send();
      },

      async send(opts) {
        const options = opts && typeof opts === "object" ? opts : {};
        const isRegen = options.regenerate === true;
        const text = (this.composerText || "").trim();
        if (!text || this.streaming) return;

        // If this is the first message, create the conversation up-front so the
        // message lands in a real, persisted conversation.
        let conversationId = this.activeConversationId;
        if (!conversationId && !isRegen) {
          try {
            const res = await fetch("/api/conversations", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ title: text.slice(0, 80) || "New chat" }),
            });
            if (!res.ok) throw new Error("create failed");
            const conv = await res.json();
            conversationId = conv.id;
            this.activeConversationId = conversationId;
            this.conversations.unshift(conv);
          } catch (e) {
            this.composerError = "Could not start a new conversation.";
            return;
          }
        }

        const payload = { conversation_id: conversationId, content: text };
        let previewUrl = null;

        if (isRegen) {
          payload.regenerate = true;
          // Reuse the original attachment so image context survives regeneration.
          if (options.attachmentId) payload.attachment_id = options.attachmentId;
        }

        if (this.attachedFile) {
          try {
            const dataUrl = await toDataUrl(this.attachedFile);
            payload.image_base64 = dataUrl.split(",", 2)[1];
            payload.image_media_type =
              this.attachedFile.type || "image/png";
            previewUrl = dataUrl;

            // Persist the file as a conversation attachment so the image
            // survives a reload (POST /api/upload). Best-effort: if the upload
            // fails we still send the inline base64 image for this turn.
            try {
              const fd = new FormData();
              fd.append("file", this.attachedFile);
              fd.append("conversation_id", String(conversationId));
              const up = await fetch("/api/upload", {
                method: "POST",
                body: fd,
              });
              if (up.ok) {
                const att = await up.json();
                payload.attachment_id = att.id;
              }
            } catch (e) {
              // non-fatal; inline image still travels with the message
            }
          } catch (e) {
            this.composerError = "Could not read the selected image.";
            return;
          }
        }

        // Optimistic UI: show the user message and an assistant placeholder
        // that streaming will fill in.
        const placeholderId = "pending-" + Date.now();
        if (!isRegen) {
          this.messages.push({
            id: Date.now(),
            role: "user",
            content: text,
            attachment_id: payload.attachment_id || null,
            html: renderMessageHtml(text, "user", previewUrl),
          });
        }
        this.messages.push({
          id: placeholderId,
          role: "assistant",
          content: "",
          html: "",
        });
        this.composerText = "";
        this.composerError = null;
        this.resetComposerHeight();
        this.streaming = true;
        this.abortController = new AbortController();
        this.scrollDown();

        let assistantText = "";
        let hadError = false;

        try {
          const res = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: this.abortController.signal,
          });

          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // SSE frames are separated by a blank line; only parse complete ones.
            const frames = buffer.split("\n\n");
            buffer = frames.pop();
            for (const frame of frames) {
              for (const line of frame.split("\n")) {
                if (!line.startsWith("data: ")) continue;
                const raw = line.slice(6).trim();
                if (!raw || raw === "[DONE]") continue;
                let event;
                try {
                  event = JSON.parse(raw);
                } catch (e) {
                  continue;
                }
                if (event.event === "token" && typeof event.content === "string") {
                  assistantText += event.content;
                  this.updatePlaceholder(placeholderId, assistantText);
                  this.scrollDown();
                } else if (event.event === "error") {
                  hadError = true;
                  this.composerError =
                    event.message || "The assistant could not respond.";
                }
              }
            }
          }
        } catch (e) {
          if (e.name === "AbortError") {
            // User pressed stop; keep whatever streamed in.
            if (!assistantText) {
              this.removePlaceholder(placeholderId);
            }
          } else {
            hadError = true;
            this.composerError = "Connection error. Is the server running?";
          }
        } finally {
          this.streaming = false;
          this.abortController = null;

          const placeholder = this.messages.find((m) => m.id === placeholderId);
          if (placeholder) {
            if (assistantText) {
              placeholder.content = assistantText;
              placeholder.html = renderMessageHtml(assistantText, "assistant");
            } else {
              this.removePlaceholder(placeholderId);
            }
          }
          if (hadError && !assistantText) {
            this.removePlaceholder(placeholderId);
          }
          this.clearAttachment();
          this.refreshSidebar();
          this.scrollDown();
        }
      },

      stop() {
        if (this.abortController) {
          this.abortController.abort();
        }
      },

      regenerate() {
        if (this.streaming) return;
        let lastUserIdx = -1;
        for (let i = this.messages.length - 1; i >= 0; i--) {
          if (this.messages[i].role === "user") {
            lastUserIdx = i;
            break;
          }
        }
        // Need at least one user message followed by something to replace.
        if (lastUserIdx === -1 || lastUserIdx === this.messages.length - 1) return;
        const lastUser = this.messages[lastUserIdx];
        // Drop the stale assistant answer from view; the backend deletes it too.
        this.messages = this.messages.slice(0, lastUserIdx + 1);
        this.composerText = lastUser.content || "";
        this.composerError = null;
        return this.send({
          regenerate: true,
          attachmentId: lastUser.attachment_id || null,
        });
      },

      updatePlaceholder(id, text) {
        const msg = this.messages.find((m) => m.id === id);
        if (msg) {
          msg.content = text;
          msg.html =
            renderMarkdown(text) + '<span class="lg-streaming-cursor">▍</span>';
        }
      },

      removePlaceholder(id) {
        const idx = this.messages.findIndex((m) => m.id === id);
        if (idx !== -1) this.messages.splice(idx, 1);
      },

      refreshSidebar() {
        fetch("/api/conversations?limit=50")
          .then((r) => r.json())
          .then((data) => {
            this.conversations = data.conversations || [];
          })
          .catch(() => {});
      },

      scrollDown() {
        this.$nextTick(() => {
          const el = this.$refs.messages;
          if (el) el.scrollTop = el.scrollHeight;
        });
      },

      // ---------- Composer / attachments ----------

      autoGrow(event) {
        const el = event.target;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 180) + "px";
      },

      // Shrink the composer back to one line after the text is cleared, so the
      // attach/send icons stay vertically centred instead of floating in a
      // leftover multi-line box.
      resetComposerHeight() {
        const el = this.$refs.composerInput;
        if (el) el.style.height = "auto";
      },

      async handleFile(event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        if (!file.type.startsWith("image/")) {
          this.composerError = "Please select an image file.";
          return;
        }
        try {
          this.attachedFile = file;
          this.attachedPreview = await toDataUrl(file);
          this.composerError = null;
        } catch (e) {
          this.composerError = "Could not read image.";
        }
      },

      clearAttachment() {
        this.attachedFile = null;
        this.attachedPreview = null;
        const input = this.attachInput;
        if (input) input.value = "";
      },

      attachImageInput(formEl) {
        const input = formEl.querySelector(".lg-attach-input");
        if (input) this.attachInput = input;
      },

      // ---------- Modals ----------

      showImage(src, title) {
        this.imageModalSrc = src;
        this.imageModalTitle = title || "Generated image";
        this.imageModal = true;
      },
      closeImageModal() {
        this.imageModal = false;
      },
    };
  }

  function statusComponent() {
    return {
      statusRows: [],
      statusLoaded: false,
      loadStatus() {
        fetch("/api/system/status")
          .then((r) => r.json())
          .then((data) => {
            this.statusRows = [
              { label: "Running", value: String(data.running) },
              { label: "Model", value: data.vlm_model || "—" },
              {
                label: "Memory (RSS)",
                value: data.memory_mb != null ? data.memory_mb + " MB" : "—",
              },
              {
                label: "CPU (process)",
                value: data.cpu_percent != null ? data.cpu_percent + "%" : "—",
              },
            ];
            this.statusLoaded = true;
          })
          .catch(() => {
            this.statusRows = [];
            this.statusLoaded = true;
          });
      },
    };
  }

  // ---------- Helpers ----------

  function renderMessageHtml(content, role, imageUrl) {
    let html = content ? renderMarkdown(content) : "";
    if (imageUrl) {
      html +=
        '<img src="' +
        escapeHtml(imageUrl) +
        '" alt="Attached image" loading="lazy">';
    }
    return html;
  }

  // ---------- Markdown (escape-first, so output is XSS-safe) ----------

  function mdInline(s) {
    return s
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(
        /\[([^\]\n]+)\]\((https?:\/\/[^)\s]+|\/[^)\s]*)\)/g,
        '<a href="$2" rel="noopener noreferrer" target="_blank">$1</a>'
      );
  }

  function renderMarkdown(text) {
    if (!text) return "";

    // 1. Extract fenced code blocks before anything else (kept raw here,
    //    escaped at restore time).
    const blocks = [];
    let src = String(text).replace(
      /```([a-zA-Z0-9+#-]*)[ \t]*\n?([\s\S]*?)(?:\n```|$)/g,
      (m, lang, code) => {
        blocks.push({ lang: lang || "", code: code });
        return "\n\u0000B" + (blocks.length - 1) + "\u0000\n";
      }
    );

    // 2. Escape everything else — tags below are the only markup that survives.
    src = escapeHtml(src);

    const out = [];
    let list = null; // 'ul' | 'ol'
    let para = [];
    const closeList = () => {
      if (list) {
        out.push("</" + list + ">");
        list = null;
      }
    };
    const flushPara = () => {
      if (para.length) {
        out.push("<p>" + para.join("<br>") + "</p>");
        para = [];
      }
    };

    for (const line of src.split("\n")) {
      let m;
      if (/^\u0000B\d+\u0000$/.test(line)) {
        flushPara();
        closeList();
        out.push(line);
      } else if ((m = line.match(/^(#{1,6})\s+(.+)$/))) {
        flushPara();
        closeList();
        const lvl = m[1].length;
        out.push("<h" + lvl + ">" + mdInline(m[2]) + "</h" + lvl + ">");
      } else if (/^\s*(?:---|\*\*\*)\s*$/.test(line)) {
        flushPara();
        closeList();
        out.push("<hr>");
      } else if ((m = line.match(/^\s*[-*]\s+(.+)$/))) {
        flushPara();
        if (list !== "ul") {
          closeList();
          out.push("<ul>");
          list = "ul";
        }
        out.push("<li>" + mdInline(m[1]) + "</li>");
      } else if ((m = line.match(/^\s*\d+\.\s+(.+)$/))) {
        flushPara();
        if (list !== "ol") {
          closeList();
          out.push("<ol>");
          list = "ol";
        }
        out.push("<li>" + mdInline(m[1]) + "</li>");
      } else if ((m = line.match(/^\s*&gt;\s?(.*)$/))) {
        flushPara();
        closeList();
        out.push("<blockquote>" + mdInline(m[1]) + "</blockquote>");
      } else if (line.trim() === "") {
        flushPara();
        closeList();
      } else {
        closeList();
        para.push(mdInline(line));
      }
    }
    flushPara();
    closeList();

    // 3. Restore code blocks with a copy button on each.
    return out.join("").replace(/\u0000B(\d+)\u0000/g, (m, i) => {
      const b = blocks[Number(i)];
      const cls = b.lang ? ' class="language-' + escapeHtml(b.lang) + '"' : "";
      return (
        '<pre><code' +
        cls +
        ">" +
        escapeHtml(b.code.replace(/\n$/, "")) +
        "</code>" +
        '<button type="button" class="lg-copy-btn" aria-label="Copy code">Copy</button>' +
        "</pre>"
      );
    });
  }

  // Copy-to-clipboard for code blocks (delegated: x-html content is plain DOM).
  document.addEventListener("click", function (e) {
    const btn = e.target && e.target.closest && e.target.closest(".lg-copy-btn");
    if (!btn) return;
    const code = btn.parentElement && btn.parentElement.querySelector("code");
    if (!code) return;
    const done = () => {
      btn.textContent = "Copied!";
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1500);
    };
    const fallback = () => {
      try {
        const ta = document.createElement("textarea");
        ta.value = code.textContent;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        done();
      } catch (err) {
        /* clipboard unavailable */
      }
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(code.textContent).then(done, fallback);
    } else {
      fallback();
    }
  });

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Read error"));
      reader.readAsDataURL(file);
    });
  }
})();
