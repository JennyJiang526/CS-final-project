import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const jsonlPath = path.join(__dirname, "medical_qa_1000.jsonl");
const outPath = path.join(__dirname, "dataset_compare.html");

const order = ["MediQA", "BioASQ", "MedQA-US", "PubMedQA"];
const first = {};
const lines = fs.readFileSync(jsonlPath, "utf8").split(/\n/);
for (const line of lines) {
  if (!line.trim()) continue;
  const o = JSON.parse(line);
  const d = o.dataset;
  if (order.includes(d) && first[d] == null) first[d] = o;
}

const payload = order.map((k) => first[k]);
const json = JSON.stringify(payload).replace(/</g, "\\u003c");

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>medical_qa_1000 — 四类数据集并排对比</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --border: #2d3a4d;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #5b9fd4;
      --ok: #6bcb77;
      --empty: #c4a35a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
      min-height: 100vh;
    }
    header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #152030 0%, var(--bg) 100%);
    }
    header h1 {
      margin: 0 0 0.35rem;
      font-size: 1.25rem;
      font-weight: 600;
    }
    header p {
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      padding: 1rem 1.25rem 2rem;
      max-width: 1800px;
      margin: 0 auto;
    }
    @media (max-width: 1200px) {
      .grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
      .grid { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 420px;
    }
    .panel-head {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .badge {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent);
      letter-spacing: 0.02em;
    }
    .tag {
      font-size: 0.72rem;
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      background: #243044;
      color: var(--muted);
    }
    .tag.has { background: #1e3d2a; color: var(--ok); }
    .tag.none { background: #3d3518; color: var(--empty); }
    section {
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      flex: 0 0 auto;
    }
    section:last-child { border-bottom: none; flex: 1; display: flex; flex-direction: column; min-height: 0; }
    section h2 {
      margin: 0 0 0.4rem;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 600;
    }
    .scroll {
      flex: 1;
      overflow: auto;
      max-height: 320px;
      font-size: 0.82rem;
      white-space: pre-wrap;
      word-break: break-word;
      background: #121a26;
      padding: 0.65rem 0.75rem;
      border-radius: 6px;
      border: 1px solid #243044;
    }
    .answer {
      font-size: 0.9rem;
      font-weight: 500;
      color: #b8d4f0;
    }
    .ctx-empty {
      color: var(--empty);
      font-style: italic;
      font-size: 0.88rem;
    }
    .ctx-box { max-height: 220px; }
  </style>
</head>
<body>
  <header>
    <h1>medical_qa_1000.jsonl — 四类并排对比</h1>
    <p>每列展示该 dataset 在文件中的<strong>首条</strong>记录；MedQA-US 与 MediQA 的 context 在本文件中均为空。</p>
  </header>
  <div class="grid" id="root"></div>
  <script type="application/json" id="emb">${json}</script>
  <script>
    const samples = JSON.parse(document.getElementById("emb").textContent);
    const root = document.getElementById("root");
    function esc(s) {
      const d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }
    for (const row of samples) {
      const has = row.context && row.context.length > 0;
      const panel = document.createElement("article");
      panel.className = "panel";
      panel.innerHTML =
        '<div class="panel-head">' +
          '<span class="badge">' + esc(row.dataset) + '</span>' +
          '<span class="tag ' + (has ? "has" : "none") + '">' +
            (has ? "context 有正文 (" + row.context.length + " 字)" : "context 为空") +
          "</span>" +
        "</div>" +
        "<section><h2>question</h2><div class=\"scroll\">" + esc(row.question) + "</div></section>" +
        "<section><h2>answer</h2><div class=\"answer\">" + esc(row.answer) + "</div></section>" +
        "<section><h2>context</h2>" +
          (has
            ? '<div class="scroll ctx-box">' + esc(row.context) + "</div>"
            : '<p class="ctx-empty\">无检索段落（空字符串）</p>') +
        "</section>";
      root.appendChild(panel);
    }
  </script>
</body>
</html>
`;

fs.writeFileSync(outPath, html, "utf8");
console.log("Wrote", outPath);
