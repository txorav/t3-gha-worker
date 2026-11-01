import http from "node:http";
import { exec } from "node:child_process";
import { timingSafeEqual } from "node:crypto";
import os from "node:os";

const PORT = Number(process.env.T3_WORKER_PORT || 4096);
const OLLAMA_URL = process.env.T3_OLLAMA_URL || "http://127.0.0.1:11434";
const DEFAULT_MODEL = process.env.T3_DEFAULT_MODEL || "qwen2.5-coder:1.5b";
const MODELS = ["colab/opencode", "colab/opencode-fast", "colab/opencode-deep", "qwen2.5-coder:1.5b", "qwen2.5-coder:0.5b"];
const AUTH_TOKEN = process.env.T3_WORKER_TOKEN || "t3-gha-worker-token-free";

let lastLogMessages = [];
let pendingTurns = 0;

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
      models: MODELS,
      defaultModel: DEFAULT_MODEL,
      host: os.hostname(),
      uptime: process.uptime(),
      queue: pendingTurns,
    });
  }

  if (url.pathname === "/v1/models") {
    return json(res, 200, {
      object: "list",
      data: MODELS.map((id) => ({ id, object: "model", owned_by: "t3-gha-ollama" })),
    });
  }

  if (url.pathname === "/debug/exec" && req.method === "POST") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch (e) {
      return json(res, 400, { error: String(e) });
    }
    exec(body.command || "echo ok", { timeout: 60000, env: process.env }, (err, stdout, stderr) => {
      json(res, 200, { stdout, stderr, err: err ? String(err) : null });
    });
    return;
  }

  if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
    let body;
    try {
      body = JSON.parse(await readBody(req));
    } catch (e) {
      return json(res, 400, { error: { message: String(e) } });
    }

    pendingTurns++;
    try {
      const targetModel = (body.model === "colab/opencode-fast" || body.model === "qwen2.5-coder:0.5b")
        ? "qwen2.5-coder:0.5b"
        : DEFAULT_MODEL;

      log(`Routing completion request to Ollama model ${targetModel}...`);
      const payload = JSON.stringify({
        ...body,
        model: targetModel,
        stream: false,
      });

      const ollamaRes = await fetch(`${OLLAMA_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
      });

      const ollamaData = await ollamaRes.json();
      log(`Ollama responded: status=${ollamaRes.status}`);
      json(res, ollamaRes.status, ollamaData);
    } catch (err) {
      log(`Error forwarding to Ollama: ${err}`);
      json(res, 500, { error: { message: String(err) } });
    } finally {
      pendingTurns--;
    }
    return;
  }

  json(res, 404, { error: { message: `no route ${req.method} ${url.pathname}` } });
});

server.listen(PORT, "0.0.0.0", () => {
  log(`GHA Ollama Worker Bridge listening on :${PORT}`);
});
