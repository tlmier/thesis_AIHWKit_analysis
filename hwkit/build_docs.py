"""Generates docs.html: extracts in-code docstrings (module/class/function,
verbatim, unmodified) into a browsable HTML reference, with the user manual
(manual.html) embedded as the second tab. Opening docs.html#manual starts on
the Manual tab (the GUI's Help menu uses that); the tabs switch freely.

Run: python3 build_docs.py
"""
import ast
import html
import json
import os
import re

SOURCE_FILES = [
    "main.py",
    "config_loading.py",
    "network.py",
    "device_fitting.py",
    "framework.py",
    "simulation.py",
    "plot.py",
    "gui_helpers.py",
    "gui_frontend.py",
    "batch_runner.py",
]

SECTION_RE = re.compile(r"^\s*(Args|Arguments|Attr|Attributes|Returns|Raises|Note|Notes)\s*:\s*(.*)$")
ARG_RE = re.compile(r"^\s{0,12}(?P<name>\*{0,2}[A-Za-z_]\w*)\s*(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<desc>.*)$")
SECTION_KEYS = {
    "args": "args", "arguments": "args", "attr": "args", "attributes": "args",
    "returns": "returns", "raises": "raises", "note": "note", "notes": "note",
}


def parse_docstring(doc):
    """Split a (loosely) Google-style docstring into description / args /
    returns / raises / note, so the reference can lay the parts out
    separately. Anything that doesn't parse stays in the description --
    the raw docstring is never lost, only restructured.

    Args:
        doc (str | None): the docstring as written in the source

    Returns:
        dict | None: {"description": str, "args": [{"name","type","desc"}],
            "returns": str, "raises": str, "note": str}, or None for empty docs
    """
    if not doc:
        return None
    parts = {"description": [], "args": [], "returns": [], "raises": [], "note": []}
    current = "description"
    current_arg = None
    for line in doc.splitlines():
        match = SECTION_RE.match(line)
        if match and match.group(1).lower() in SECTION_KEYS:
            current = SECTION_KEYS[match.group(1).lower()]
            current_arg = None
            rest = match.group(2).strip()
            if rest:
                arg = ARG_RE.match(rest) if current == "args" else None
                if arg:
                    current_arg = {"name": arg["name"], "type": (arg["type"] or "").strip(),
                                   "desc": arg["desc"].strip()}
                    parts["args"].append(current_arg)
                elif current != "args":
                    parts[current].append(rest)
            continue
        if current == "args":
            arg = ARG_RE.match(line)
            if arg:
                current_arg = {"name": arg["name"], "type": (arg["type"] or "").strip(),
                               "desc": arg["desc"].strip()}
                parts["args"].append(current_arg)
            elif current_arg is not None and line.strip():
                current_arg["desc"] += " " + line.strip()
        else:
            parts[current].append(line.rstrip())
    result = {
        "description": "\n".join(parts["description"]).strip(),
        "args": parts["args"],
        "returns": "\n".join(parts["returns"]).strip(),
        "raises": "\n".join(parts["raises"]).strip(),
        "note": "\n".join(parts["note"]).strip(),
    }
    if not (result["description"] or result["args"] or result["returns"]):
        return None
    return result


def extract_module(path):
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source, filename=path)

    module = {
        "file": path,
        "doc": ast.get_docstring(tree),
        "classes": [],
        "functions": [],
    }

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = {
                "name": node.name,
                "doc": ast.get_docstring(node),
                "methods": [],
            }
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls["methods"].append({
                        "name": sub.name,
                        "args": [a.arg for a in sub.args.args if a.arg != "self"],
                        "doc": ast.get_docstring(sub),
                        "parsed": parse_docstring(ast.get_docstring(sub)),
                    })
            cls["parsed"] = parse_docstring(cls["doc"])
            module["classes"].append(cls)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module["functions"].append({
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "doc": ast.get_docstring(node),
                "parsed": parse_docstring(ast.get_docstring(node)),
            })

    return module


def build_data():
    return [extract_module(f) for f in SOURCE_FILES if os.path.exists(f)]


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HWKit GUI Documentation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg: #ffffff; --fg: #1c1e21; --muted: #6b7280; --border: #e5e7eb;
  --accent: #2563eb; --code-bg: #f3f4f6; --sidebar-bg: #fafafa;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e5e7eb; --muted:#9ca3af; --border:#262a33;
    --accent:#60a5fa; --code-bg:#1a1d24; --sidebar-bg:#14161b; }
}
:root[data-theme="dark"] { --bg:#0f1115; --fg:#e5e7eb; --muted:#9ca3af; --border:#262a33;
  --accent:#60a5fa; --code-bg:#1a1d24; --sidebar-bg:#14161b; }
:root[data-theme="light"] { --bg:#ffffff; --fg:#1c1e21; --muted:#6b7280; --border:#e5e7eb;
  --accent:#2563eb; --code-bg:#f3f4f6; --sidebar-bg:#fafafa; }

* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--fg); background: var(--bg); display: flex; min-height: 100vh;
}
.sidebar {
  width: 260px; flex-shrink: 0; background: var(--sidebar-bg); border-right: 1px solid var(--border);
  padding: 1rem; overflow-y: auto; height: 100vh; position: sticky; top: 0;
}
.sidebar h1 { font-size: 1rem; margin: 0 0 1rem; }
.tabs { display: flex; gap: 0.25rem; margin-bottom: 1rem; }
.tab-btn {
  flex: 1; padding: 0.4rem; font-size: 0.85rem; border: 1px solid var(--border); background: var(--bg);
  color: var(--fg); border-radius: 6px; cursor: pointer;
}
.tab-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.nav-section { margin-bottom: 0.75rem; }
.nav-file { font-weight: 600; font-size: 0.85rem; margin: 0.6rem 0 0.2rem; color: var(--muted); }
.nav-link {
  display: block; font-size: 0.85rem; padding: 0.15rem 0 0.15rem 0.6rem; color: var(--fg);
  text-decoration: none; border-left: 2px solid transparent; word-break: break-word;
}
.nav-link:hover { border-left-color: var(--accent); color: var(--accent); }
.nav-link.indent { padding-left: 1.2rem; color: var(--muted); }

main { flex: 1; padding: 2rem 3rem; max-width: 900px; overflow-x: auto; }
.view { display: none; }
.view.active { display: block; }
#view-manual.active { display: flex; margin: -2rem -3rem; }
.manual-frame { flex: 1; width: 100%; min-height: 100vh; border: none; }

.module { margin-bottom: 3rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border); }
.module h2 { font-family: monospace; font-size: 1.3rem; margin-bottom: 0.3rem; }
.module .module-doc { color: var(--muted); white-space: pre-wrap; margin-bottom: 1rem; }

.entity { margin: 1.2rem 0 1.2rem 0.5rem; padding-left: 0.8rem; border-left: 2px solid var(--border); }
.entity h3 { font-family: monospace; font-size: 1.05rem; margin: 0 0 0.3rem; }
.entity h4 { font-family: monospace; font-size: 0.95rem; margin: 0.8rem 0 0.2rem 1rem; color: var(--fg); }
.sig { color: var(--muted); font-weight: normal; }
.doc { white-space: pre-wrap; background: var(--code-bg); border-radius: 6px; padding: 0.6rem 0.8rem;
  font-size: 0.9rem; margin: 0.3rem 0 0.3rem 1rem; }
.no-doc { color: var(--muted); font-style: italic; font-size: 0.85rem; margin-left: 1rem; }
.desc { white-space: pre-wrap; font-size: 0.9rem; margin: 0.3rem 0 0.5rem 1rem; }
.argtable { border-collapse: collapse; margin: 0.4rem 0 0.6rem 1rem; font-size: 0.85rem; width: calc(100% - 1rem); }
.argtable th { text-align: left; font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; padding: 0.25rem 0.9rem 0.25rem 0; border-bottom: 1px solid var(--border); }
.argtable td { padding: 0.3rem 0.9rem 0.3rem 0; border-bottom: 1px solid var(--border); vertical-align: top; }
.argtable tr:last-child td { border-bottom: none; }
.argname { font-family: monospace; white-space: nowrap; }
.argtype { color: var(--muted); font-family: monospace; font-size: 0.8rem; }
.secblock { font-size: 0.85rem; margin: 0.35rem 0 0.35rem 1rem; white-space: pre-wrap; }
.secblock .lbl { display: inline-block; font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-right: 0.5rem; }

.manual-section { margin-bottom: 1.5rem; }
.manual-section h2 { font-size: 1.15rem; margin-bottom: 0.4rem; }
.manual-section ul { margin: 0; padding-left: 1.4rem; color: var(--muted); }
.manual-section li { padding: 0.15rem 0; }
.manual-note { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
</style>
</head>
<body>

<nav class="sidebar">
  <h1>HWKit GUI Docs</h1>
  <div class="tabs">
    <button class="tab-btn active" data-view="reference">Reference</button>
    <button class="tab-btn" data-view="manual">Manual</button>
  </div>
  <div id="nav-reference"></div>
  <div id="nav-manual" style="display:none"></div>
</nav>

<main>
  <section id="view-reference" class="view active"></section>
  <section id="view-manual" class="view"></section>
</main>

<script>
const MODULES = __MODULES_JSON__;

function esc(s) {
  const d = document.createElement('div');
  d.innerText = s;
  return d.innerHTML;
}
function slug(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

// Structured rendering: description first, then the arguments as their own
// table, then Returns/Raises/Note blocks. Entities whose docstring could not
// be sectioned fall back to the verbatim block.
function renderDoc(entity) {
  const p = entity.parsed;
  if (!p) {
    return entity.doc ? `<div class="doc">${esc(entity.doc)}</div>`
                      : `<div class="no-doc">(no docstring)</div>`;
  }
  let h = '';
  if (p.description) h += `<div class="desc">${esc(p.description)}</div>`;
  if (p.args.length) {
    h += `<table class="argtable"><tr><th>Parameter</th><th>Type</th><th>Description</th></tr>`;
    p.args.forEach(a => {
      h += `<tr><td class="argname">${esc(a.name)}</td>` +
           `<td class="argtype">${esc(a.type)}</td><td>${esc(a.desc)}</td></tr>`;
    });
    h += `</table>`;
  }
  if (p.returns) h += `<div class="secblock"><span class="lbl">Returns</span>${esc(p.returns)}</div>`;
  if (p.raises) h += `<div class="secblock"><span class="lbl">Raises</span>${esc(p.raises)}</div>`;
  if (p.note) h += `<div class="secblock"><span class="lbl">Note</span>${esc(p.note)}</div>`;
  return h;
}

function renderReference() {
  const navEl = document.getElementById('nav-reference');
  const viewEl = document.getElementById('view-reference');
  let nav = '';
  let body = '';

  MODULES.forEach(mod => {
    const modId = slug(mod.file);
    nav += `<div class="nav-file"><a class="nav-link" href="#${modId}">${esc(mod.file)}</a></div>`;

    body += `<div class="module" id="${modId}"><h2>${esc(mod.file)}</h2>`;
    if (mod.doc) body += `<div class="module-doc">${esc(mod.doc)}</div>`;

    mod.classes.forEach(cls => {
      const cid = `${modId}-${slug(cls.name)}`;
      nav += `<a class="nav-link indent" href="#${cid}">class ${esc(cls.name)}</a>`;
      body += `<div class="entity" id="${cid}"><h3>class ${esc(cls.name)}</h3>`;
      body += renderDoc(cls);
      cls.methods.forEach(m => {
        const mid = `${cid}-${slug(m.name)}`;
        body += `<h4 id="${mid}">${esc(m.name)}(<span class="sig">${esc(m.args.join(', '))}</span>)</h4>`;
        body += renderDoc(m);
      });
      body += `</div>`;
    });

    mod.functions.forEach(fn => {
      const fid = `${modId}-${slug(fn.name)}`;
      nav += `<a class="nav-link indent" href="#${fid}">${esc(fn.name)}()</a>`;
      body += `<div class="entity" id="${fid}"><h3>${esc(fn.name)}(<span class="sig">${esc(fn.args.join(', '))}</span>)</h3>`;
      body += renderDoc(fn);
      body += `</div>`;
    });

    body += `</div>`;
  });

  navEl.innerHTML = nav;
  viewEl.innerHTML = body;
}

function renderManual() {
  document.getElementById('nav-manual').innerHTML =
    `<p class="manual-note">The full manual, embedded. ` +
    `Standalone page: <a class="nav-link" href="manual.html">manual.html</a></p>`;
  document.getElementById('view-manual').innerHTML =
    `<iframe class="manual-frame" src="manual.html" title="HWKit GUI Manual"></iframe>`;
}

function activateTab(view) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.view === view));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(`view-${view}`).classList.add('active');
  document.getElementById('nav-reference').style.display = view === 'reference' ? '' : 'none';
  document.getElementById('nav-manual').style.display = view === 'manual' ? '' : 'none';
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    activateTab(btn.dataset.view);
    history.replaceState(null, '', `#${btn.dataset.view}`);
  });
});

renderReference();
renderManual();
// The GUI's Help menu opens docs.html#manual or docs.html#reference to land on a
// predefined tab; every other hash is a reference anchor and keeps the default tab.
if (location.hash === '#manual') activateTab('manual');
</script>
</body>
</html>
"""


def main():
    data = build_data()
    out = HTML_TEMPLATE.replace("__MODULES_JSON__", json.dumps(data))
    with open("docs.html", "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"Wrote docs.html ({len(data)} modules)")


if __name__ == "__main__":
    main()
