// Tiny static server for docs/ that gzips responses like GitHub Pages does. Used by the e2e tests.
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createGzip } from "node:zlib";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../docs");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".css": "text/css", ".wasm": "application/wasm", ".bin": "application/octet-stream", ".svg": "image/svg+xml", ".md": "text/markdown" };

export function serve(port = 0) {
  const server = createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    let file = path.join(ROOT, url.endsWith("/") ? url + "index.html" : url);
    if (!file.startsWith(ROOT)) return res.writeHead(403).end();
    let st;
    try {
      st = statSync(file);
    } catch {
      return res.writeHead(404).end("not found");
    }
    const type = TYPES[path.extname(file)] || "application/octet-stream";
    const gzip = /\bgzip\b/.test(req.headers["accept-encoding"] || "");
    res.writeHead(200, { "Content-Type": type, ...(gzip ? { "Content-Encoding": "gzip" } : { "Content-Length": st.size }) });
    const stream = createReadStream(file);
    gzip ? stream.pipe(createGzip({ level: 6 })).pipe(res) : stream.pipe(res);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve({ server, url: `http://127.0.0.1:${server.address().port}/` })));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { url } = await serve(Number(process.argv[2] || 8000));
  console.log(`serving docs/ at ${url}`);
}
