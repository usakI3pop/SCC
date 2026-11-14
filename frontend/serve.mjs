import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('.', import.meta.url)));
const types = { '.html':'text/html; charset=utf-8', '.css':'text/css; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.svg':'image/svg+xml' };
createServer((req, res) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); }
  catch { res.writeHead(400).end('Bad request'); return; }
  const candidate = resolve(root, `.${pathname === '/' ? '/index.html' : pathname}`);
  if (candidate !== root && !candidate.startsWith(root + sep)) { res.writeHead(403).end('Forbidden'); return; }
  try {
    if (!statSync(candidate).isFile()) throw Error('not a file');
    res.writeHead(200, { 'Content-Type': types[extname(candidate)] || 'application/octet-stream', 'X-Content-Type-Options':'nosniff' });
    createReadStream(candidate).pipe(res);
  } catch { res.writeHead(404).end('Not found'); }
}).listen(4173, '127.0.0.1', () => console.log('Sentinel AI interface: http://127.0.0.1:4173'));
