# LocalGPT — Phase 2 Audit Report

**Date:** 2026-10-09
**Auditor:** Buffy (Codebuff agent)
**Scope:** Phase 2 — Image Generation + Editing + Unified Routing (backend, frontend, tests, live verification)
**Status:** Complete and verified. Ready for Phase 3 after the creator's review/commit.

---

## 1. What Phase 2 delivers

| Capability | Where | Verified by |
|---|---|---|
| Intent router (CHAT / IMAGE_ANALYSIS / IMAGE_GENERATION / IMAGE_EDIT) | `app/core/router.py` | `tests/test_router.py` |
| Unified routing from chat (stream + non-stream) | `app/api/chat.py` (`_classify`, `_try_route_image`) | 7 routing tests |
| Structured `image_job` SSE event before VLM gate | `app/api/chat.py` | live curl + browser |
| Image job system (queue → run → poll → cancel) | `app/services/image_service.py` | 9 job tests + live E2E |
| External CPU engine: stable-diffusion.cpp (`sd-cli`) | `app/inference/image_engine.py` | live queue→cancel (real sd-cli) |
| Job/file/ownership endpoints | `app/api/images.py` | 11 endpoint tests |
| Conversation image history endpoint | `app/api/conversations.py` | `test_generated_images_listing_and_traversal_guard` |
| Engine status in `/api/models` + `/api/system/status` | `app/api/system.py` | live curl + tests |
| Inline image-job UI (progress, Cancel, figure, view modal, download) | `app/static/js/app.js`, `app/static/css/app.css` | browser E2E |
| History re-render of completed images | `loadMessages` + `imageFigureHtml` | covered by endpoint tests; see gaps |

## 2. Defects found and fixed during this audit

1. **Frontend file was syntactically broken on disk** (`node --check` failed):
   stray `}` before the `image_job` SSE branch; duplicated `this.streaming = false;`;
   a missing backslash in the Cancel-button string (from an earlier byte-level edit);
   and the whole image-job polling block had been inserted *outside* the Alpine
   component object. Repaired with an anchor-asserted script; `node --check` now exits 0.
2. **Job state lost between requests**: routers used `get_database()` which builds a
   *new* `Database` per request, so `get_image_service`'s identity check rebuilt the
   service on every poll — cancel events and progress vanished. Fixed by wiring
   `app.dependency_overrides[get_database] = _get_database` in `app/main.py`
   (the helper existed but was never registered) and keying the service cache on
   DB **path** in `get_image_service`.
3. **Cancel → failed race**: killing sd-cli makes it exit non-zero; the worker's
   failure handler overwrote `cancelled` with `failed` + a scary error. `_run_job`
   now checks the cancel flag before recording a failure. Verified live twice
   (status held `cancelled`, sd-cli terminated, no orphan processes).
4. **Test isolation bug (pre-existing)**: `database_path_path` returned the fixed
   default `data/localgpt.db` and ignored `LOCALGPT_DATA_DIR`, so every pytest run
   since Phase 1 wrote test conversations/jobs into the **real** database.
   Fixed in `app/config.py`; verified a full suite run leaves the real DB untouched
   (163 → 163 conversations).

## 3. Verification evidence

- **Tests:** `119 passed` (Phase 1 baseline 99 + 20 new in `tests/test_phase2_images.py`).
  New tests use a `FakeEngine` (no sd-cli, no llama-server) per AGENTS.md §33.
- **Syntax:** `node --check app/static/js/app.js` → OK; `ast.parse` OK on all touched Python.
- **Live E2E (real engine, `sd_cpp` ready, all capabilities true):**
  - `POST /api/image/generate` → queued → `generating` (sd-cli spawned) →
    `POST .../cancel` → `cancelled: true`, status held `cancelled`, process gone.
  - `POST /api/chat/stream` "Draw a red circle" →
    `data: {"event":"image_job","job_id":"img_...","status":"queued",...}` + `done`.
  - Browser: text streaming with markdown/code/copy; image request renders the
    job card with **Cancel**; clicking Cancel cancels server-side (Alpine `@click`
    inside `x-html` works); history reload renders persisted messages.
  - `/api/models` and `/api/system/status` expose `image_engine`
    (`ready:true, engine:"sd_cpp", capabilities:{txt2img,img2img,inpaint}`).

## 4. Changed files (this phase)

```
M app/api/chat.py              +132   routing + image_job SSE
M app/api/conversations.py      +37   GET /conversations/{id}/images
M app/api/images.py            +142   generate/edit/jobs/cancel + file serving
M app/api/system.py             +10   engine status exposure (version → phase2)
M app/config.py                 +20   engine settings + DB-path isolation fix
M app/core/resource_manager.py  +17   heavy-task gate + engine ready flags
M app/main.py                    +5   dependency override (stable DB)
M app/services/image_service.py +572   job lifecycle (rewritten)
M app/static/css/app.css       +128   image-job card/figure/button styles
M app/static/js/app.js         +291   SSE branch, polling, history images
?? app/inference/image_engine.py      new: ImageEngine ABC + SDCppEngine (521 lines)
?? tests/test_phase2_images.py        new: 20 Phase 2 tests (498 lines)
?? scripts/mock_llama_server.py       new: dev mock llama-server (SSE, stdlib)
?? app/models/                        now tracked after the .gitignore fix
```

`git diff --stat`: 11 files, +1316 / −43 (plus new files above).

## 5. Known gaps and decisions (for Phase 3)

1. **No real completed PNG yet.** ~555 s/step at 512² on the target CPU, so
   completion rendering is proven only with `FakeEngine`; the full engine path was
   proven with queue→cancel. A one-off long-running generation test remains open.
2. **Polluted rows in the real `data/localgpt.db`** (~163 conversations from
   pre-fix test runs; some reference PNGs that live in deleted pytest temp dirs,
   which is why history can render a broken `<img>` → 404 for *those* rows).
   Cleanup SQL can be provided on request — nothing was deleted automatically.
3. **`chat.html` status chip dropped**: the job card lives inside the message HTML,
   so a separate streaming-bubble chip was unnecessary (verified in browser).
4. **Docs/CHANGELOG still pending** (per the agreed sequencing, Phase 3):
   stable-diffusion.cpp deviation from AGENTS.md's FastSD wording, image-generation /
   image-editing / api / architecture docs, CHANGELOG, README refresh.
5. **Single active generation** by design (§14/§15): a second job queues behind
   `_run_lock`; UI polls all watched jobs via the `imageJobs` list.
6. **Dev tooling:** mock llama-server lives at `scripts/mock_llama_server.py`
   (root-level copy moved); LocalGPT `:8000` + mock `:8080` are running now.

## 6. Recommended commit (creator handles Git per AGENTS.md §36)

```
PHASE 2 COMPLETED — image generation/editing, unified routing, job system, inline image UI
```

Suggested staging: all modified files above, `app/inference/image_engine.py`,
`tests/test_phase2_images.py`, `scripts/mock_llama_server.py`, and `app/models/`
(the .gitignore fix un-ignores it — without `git add app/models/`, the database and
schemas modules stay untracked).
