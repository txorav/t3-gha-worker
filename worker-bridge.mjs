import http from "node:http";
import { spawn, exec } from "node:child_process";
import { randomUUID, timingSafeEqual } from "node:crypto";
import os from "node:os";

const PORT = Number(process.env.T3_WORKER_PORT || 4096);
const OPENCODE_BIN = process.env.T3_OPENCODE_BIN || "opencode";
const WORKDIR = process.env.T3_WORKDIR || process.env.HOME || "/tmp/workspace";
const MODELS = (process.env.T3_MODELS || "colab/opencode,colab/opencode-fast,colab/opencode-deep").split(",");
const AUTH_TOKEN = process.env.T3_WORKER_TOKEN || "";
const TURN_TIMEOUT_MS = Number(process.env.T3_TURN_TIMEOUT_MS || 45_000);

let sessionId = null;
let turnLock = Promise.resolve();
let pendingTurns = 0;
let lastLogMessages = [];
let currentTurnInfo = null;

function log(...args) {
  const msg = `[worker ${new Date().toISOString()}] ${args.map(a => typeof a === "object" ? JSON.stringify(a) : String(a)).join(" ")}`;
  console.log(msg);
  lastLogMessages.push(msg);
  if (lastLogMessages.length > 200) lastLogMessages.shift();
}

function json(res, code, body) {
  const payload = JSON.stringify(body);
  res.writeHead(code, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(payload),
    "access-control-allow-origin": "*",
  });
  res.end(payload);
}

function authorized(req) {
  if (!AUTH_TOKEN) return true;
  const header = String(req.headers.authorization || "");
  const prefix = "Bearer ";
  if (!header.startsWith(prefix)) return false;
  const provided = Buffer.from(header.slice(prefix.length));
  const expected = Buffer.from(AUTH_TOKEN);
  return provided.length === expected.length && timingSafeEqual(provided, expected);
}

function readBody(req, limit = 32 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error("payload too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function flatten(messages) {
  const parts = [];
  for (const m of messages) {
    const text =
      typeof m.content === "string"
        ? m.content
        : Array.isArray(m.content)
          ? m.content.map((p) => (typeof p === "string" ? p : p.text || "")).join("\n")
          : "";
    if (!text) continue;
    if (m.role === "system") parts.push(`[system]\n${text}`);
    else if (m.role === "assistant") parts.push(`[assistant]\n${text}`);
    else parts.push(text);
  }
  return parts.join("\n\n");
}

const CANDIDATE_MODELS = [
  "opencode/glm-5.3-flash-free",
  "opencode/flint-40b-free",
  "opencode/beam-free",
  "opencode/nemotron-3.5-lightning-free",
  "opencode/deepseek-v4.1-flash-free",
  "opencode/mimo-v2.6-flash-free",
];

function resolveModel(model) {
  if (!model) return CANDIDATE_MODELS[0];
  if (model === "colab/opencode" || model === "colab/opencode-fast") return "opencode/glm-5.3-flash-free";
  if (model === "colab/opencode-deep") return "opencode/flint-40b-free";
  if (model.startsWith("colab/")) return "opencode/glm-5.3-flash-free";
  return model;
}

function runTurn(prompt, model) {
  return new Promise((resolve) => {
    const actualModel = resolveModel(model);
    const args = ["run", "--auto", "--standalone", "--format", "json", "--model", actualModel];
    if (sessionId) args.push("--session", sessionId);
    args.push(prompt);

    currentTurnInfo = { model: actualModel, startedAt: new Date().toISOString(), promptLength: prompt.length };
    log(`Starting turn with model ${actualModel}...`);

    const child = spawn(OPENCODE_BIN, args, {
      cwd: WORKDIR,
      env: { ...process.env, NO_COLOR: "1", CI: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let out = "";
    let err = "";
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      currentTurnInfo = null;
      log(`Turn finished: code=${value.code} textLen=${value.text?.length || 0} err=${value.error?.slice(0, 100)}`);
      resolve(value);
    };

    const timer = setTimeout(() => {
      log(`Turn timeout reached (${TURN_TIMEOUT_MS}ms) for model ${actualModel}, killing child`);
      try { child.kill("SIGKILL"); } catch {}
      finish({ text: "", error: `opencode turn timed out after ${TURN_TIMEOUT_MS}ms`, limited: false, code: -1 });
    }, TURN_TIMEOUT_MS);
    timer.unref?.();

    child.stdout.on("data", (d) => {
      const s = d.toString();
      out += s;
      log(`stdout chunk: ${s.trim().slice(0, 150)}`);
    });
    child.stderr.on("data", (d) => {
      const s = d.toString();
      err += s;
      log(`stderr chunk: ${s.trim().slice(0, 150)}`);
    });
    child.on("error", (e) => {
      log(`child error: ${e}`);
      finish({ text: "", error: String(e), limited: false, code: -1 });
    });
    child.on("close", (code) => {
      const combined = `${out}\n${err}`;
      const limited = /usage limit|quota|rate limit|limit reached|too many requests|\b429\b|FreeUsageLimitError/i.test(combined);

      let text = "";
      let capturedError = "";
      for (const line of out.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("{")) continue;
        try {
          const evt = JSON.parse(t);
          if (typeof evt.sessionID === "string") sessionId = evt.sessionID;
          if (typeof evt.sessionId === "string") sessionId = evt.sessionId;
          const part = evt.part || evt;
          if (part && part.type === "text" && typeof part.text === "string") text += part.text;
          else if (typeof evt.text === "string") text += evt.text;
          if (evt.type === "error" || evt.error) {
            const msg = evt.error?.message || evt.error?.data?.message || (typeof evt.error === "string" ? evt.error : "");
            if (msg) capturedError = capturedError ? `${capturedError}\n${msg}` : msg;
          }
        } catch {}
      }
      if (!text && !capturedError) text = out.trim();
      const finalError = (code !== 0 || limited || !text) ? (capturedError || err.trim() || out.trim()) : "";
      finish({ text, error: finalError, limited, code });
    });
  });
}

function enqueueTurn(fn) {
  pendingTurns++;
  const result = turnLock.then(fn);
  turnLock = result.then(() => undefined, () => undefined);
  result.then(() => pendingTurns--, () => pendingTurns--);
  return result;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "authorization, content-type",
      "access-control-allow-methods": "GET, POST, OPTIONS",
    });
    return res.end();
  }

  if (!authorized(req)) {
    return json(res, 401, { error: { message: "unauthorized" } });
  }

  if (url.pathname === "/health") {
    return json(res, 200, {
      ok: true,
      sessionId,
      models: MODELS,
      host: os.hostname(),
      cwd: WORKDIR,
      uptime: process.uptime(),
      queue: pendingTurns,
    });
  }

  if (url.pathname === "/debug/logs") {
    return json(res, 200, {
      currentTurn: currentTurnInfo,
      queue: pendingTurns,
      logs: lastLogMessages.slice(-60),
    });
  }

  if (url.pathname === "/debug/exec" && req.method === "POST") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch (e) {
      return json(res, 400, { error: String(e) });
    }
    exec(body.command || "echo ok", { timeout: 30000, env: process.env }, (err, stdout, stderr) => {
      json(res, 200, { stdout, stderr, err: err ? String(err) : null });
    });
    return;
  }

  if (url.pathname === "/v1/models") {
    return json(res, 200, {
      object: "list",
      data: MODELS.map((id) => ({ id, object: "model", owned_by: "t3-gha" })),
    });
  }

  if (url.pathname === "/v1/session/reset" && req.method === "POST") {
    sessionId = null;
    return json(res, 200, { ok: true });
  }

  if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch (e) {
      return json(res, 400, { error: { message: String(e) } });
    }

    const requestedModel = resolveModel(body.model || MODELS[0]);
    const candidateModels = [
      requestedModel,
      ...CANDIDATE_MODELS.filter((m) => m !== requestedModel),
    ];
    const prompt = flatten(body.messages || []);

    let result = { text: "", error: "", limited: false, code: null };
    for (const m of candidateModels) {
      result = await enqueueTurn(() => runTurn(prompt, m));
      if (result.text) break;
      if (!result.limited && !result.error) break;
    }

    const { text, error, limited } = result;
    if (limited && !text) {
      return json(res, 429, {
        error: { message: error || "All free models reached rate limit", type: "usage_limit" },
      });
    }

    const responseText = text || (error ? `(Turn ended with error: ${error})` : "Task completed.");
    const completion = {
      id: `chatcmpl-${randomUUID()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: requestedModel,
      choices: [{ index: 0, message: { role: "assistant", content: responseText }, finish_reason: "stop" }],
      usage: {
        prompt_tokens: Math.ceil(prompt.length / 4),
        completion_tokens: Math.ceil(responseText.length / 4),
        total_tokens: Math.ceil((prompt.length + responseText.length) / 4),
      },
    };

    return json(res, 200, completion);
  }

  json(res, 404, { error: { message: `no route ${req.method} ${url.pathname}` } });
});

server.listen(PORT, "0.0.0.0", () => {
  log(`GHA Worker listening on :${PORT}`);
});
