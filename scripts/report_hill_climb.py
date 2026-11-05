"""Hill-climb report: every adopted step on the 90 development cases x 3 runs, plus a per-task table.

Each step is one recorded 270-episode run in ``runs/<label>/development/episodes.jsonl[.gz]``, graded with the
deterministic six-check grader (``agent/grader.py``). Numbers are computed from those episodes, never typed in.

    python scripts/report_hill_climb.py [--png]

Writes ``reports/hill-climb/climb.html`` and ``by-task.html`` (and ``.png`` screenshots with ``--png``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "hill-climb"
# (segment, chart label, table label, run label)
STEPS = [
    ("single", "Baseline\n(one prompt)", "Baseline", "baseline"),
    ("single", "+ Agent\nOptimizer\nprompt", "+ Agent Optimizer prompt", "opt-r2-cand1"),
    ("routed", "Decision-1\nrouter + 6\nspecialists", "Router + 6 specialists", "routed-mai-baseline"),
    ("routed", "+ incident\ndate checks", "+ incident date checks", "routed-mai-optimized"),
    ("routed", "+ prompt fixes,\nbatched\nportfolio", "+ prompt fixes, batched portfolio", "routed-mai-r4"),
    ("routed", "+ threshold\nrules", "+ threshold rules (v8)", "routed-mai-r5"),
    ("routed", "+ RFT, efficiency\nreward (v9)", "+ RFT, efficiency reward (v9)", "rft-eff"),
    ("routed", "RFT except\nportfolio\n(deployed v10)", "RFT except portfolio (deployed v10)", "rft-eff-mixed"),
]
SEGMENTS = {"single": "One agent, one shared prompt", "routed": "Decision-1 router + six task specialists"}
SOL = ("GPT-6 Sol, same router and prompts", "routed-gpt6-sol-r5")
SOL_BASE = "routed-gpt6-sol"
TASKS = [("validity", "Validity"), ("ship_decision", "Ship decision"), ("early_look", "Early look"),
         ("attribution", "Headline vs image"), ("portfolio", "Portfolio scorecard"), ("planning", "Test planning")]
BLUE, RED, INK, MUTED, GRID = "#2f7be5", "#e4572e", "#1d1d1f", "#6b6b6b", "#ececec"


def episodes(label: str) -> list[dict]:
    base = ROOT / "runs" / label / "development" / "episodes.jsonl"
    if base.exists():
        text = base.read_text(encoding="utf-8")
    else:
        with gzip.open(base.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as handle:
            text = handle.read()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) != 270 or any(r["error"] for r in rows):
        raise SystemExit(f"{label}: expected 270 error-free episodes, found {len(rows)} "
                         f"({sum(bool(r['error']) for r in rows)} errors)")
    return rows


def stats(rows: list[dict]) -> dict:
    return {"pass": 100 * sum(r["strict_pass"] for r in rows) / len(rows),
            "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows) / len(rows) / 1000}


def measure() -> dict:
    out = {}
    for label in [s[3] for s in STEPS] + [SOL[1], SOL_BASE]:
        rows = episodes(label)
        out[label] = {"overall": stats(rows), **{t: stats([r for r in rows if r["mode"] == t]) for t, _ in TASKS}}
    return out


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def positions(L: float, R: float) -> list[float]:
    gap = 60
    step = (R - L - 80 - gap) / (len(STEPS) - 1)
    xs, x = [], L + 40
    for i, s in enumerate(STEPS):
        if i and s[0] != STEPS[i - 1][0]:
            x += gap
        xs.append(x)
        x += step
    return xs


def chart(m: dict, w: int = 1180, h: int = 540) -> str:
    lo, hi = 75, 100
    L, R, T, B = 64, w - 24, 70, h - 110
    y = lambda v: B - (v - lo) / (hi - lo) * (B - T)  # noqa: E731
    xs = positions(L, R)
    split = next(i for i in range(1, len(STEPS)) if STEPS[i][0] != STEPS[i - 1][0])
    mid = (xs[split - 1] + xs[split]) / 2
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">']
    for v in range(lo, hi + 1, 5):
        o.append(f'<line x1="{L}" x2="{R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{y(v) + 5:.1f}" font-size="13" text-anchor="end" fill="{MUTED}">{v}%</text>')
    sol = m[SOL[1]]["overall"]["pass"]
    for name, color, dash, x0 in (("MAI-Code-1.1-Flash strict pass", BLUE, "", L), (SOL[0], RED, "7 6", L + 300)):
        o.append(f'<line x1="{x0}" x2="{x0 + 24}" y1="16" y2="16" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>'
                 f'<text x="{x0 + 30}" y="21" font-size="13" fill="{INK}">{esc(name)}</text>')
    for seg, x0, x1 in (("single", L, mid), ("routed", mid, R)):
        o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{T - 22}" font-size="13.5" font-weight="600" text-anchor="middle" '
                 f'fill="{MUTED}">{esc(SEGMENTS[seg])}</text>')
    o.append(f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="{T - 36}" y2="{B + 4}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 4"/>')
    o.append(f'<line x1="{mid + 8:.1f}" x2="{R}" y1="{y(sol):.1f}" y2="{y(sol):.1f}" stroke="{RED}" stroke-width="2" '
             f'stroke-dasharray="7 6" opacity="0.85"/><text x="{mid + 12:.1f}" y="{y(sol) - 7:.1f}" font-size="13" '
             f'fill="{RED}" font-weight="600">GPT-6 Sol {sol:.1f}%</text>')
    values = [m[s[3]]["overall"]["pass"] for s in STEPS]
    for seg in ((0, split), (split, len(STEPS))):
        pts = " ".join(f"{xs[i]:.1f},{y(values[i]):.1f}" for i in range(*seg))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="3"/>')
    for i, v in enumerate(values):
        last = i == len(values) - 1
        o.append(f'<circle cx="{xs[i]:.1f}" cy="{y(v):.1f}" r="{9 if last else 6.5}" fill="{BLUE}" stroke="#fff" stroke-width="2.5"/>'
                 f'<text x="{xs[i]:.1f}" y="{y(v) + 26:.1f}" font-size="14" text-anchor="middle" fill="{BLUE}" '
                 f'font-weight="{700 if last else 500}" stroke="#fff" stroke-width="5" paint-order="stroke">{v:.1f}%</text>')
    for i, s in enumerate(STEPS):
        for k, part in enumerate(s[1].split("\n")):
            o.append(f'<text x="{xs[i]:.1f}" y="{B + 24 + 16 * k:.1f}" font-size="13" text-anchor="middle" fill="{INK}">{esc(part)}</text>')
    o.append("</svg>")
    return "".join(o)


def token_chart(m: dict, w: int = 1180, h: int = 190) -> str:
    L, R, T, B = 64, w - 24, 26, h - 26
    xs = positions(L, R)
    values = [m[s[3]]["overall"]["tokens"] for s in STEPS]
    sol = m[SOL[1]]["overall"]["tokens"]
    top = max(values) * 1.15
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">',
         f'<text x="{L}" y="14" font-size="13" font-weight="600" fill="{INK}">Tokens per episode (k), input + output</text>']
    for i, v in enumerate(values):
        hgt = v / top * (B - T)
        color = "#94a3b8" if i == 0 else BLUE
        o.append(f'<rect x="{xs[i] - 16:.1f}" y="{B - hgt:.1f}" width="32" height="{hgt:.1f}" rx="3" fill="{color}"/>'
                 f'<text x="{xs[i]:.1f}" y="{B - hgt - 5:.1f}" font-size="12" text-anchor="middle" fill="{color}" font-weight="600">{v:.1f}k</text>')
    ys = B - sol / top * (B - T)
    split = next(i for i in range(1, len(STEPS)) if STEPS[i][0] != STEPS[i - 1][0])
    o.append(f'<line x1="{xs[split] - 30:.1f}" x2="{R}" y1="{ys:.1f}" y2="{ys:.1f}" stroke="{RED}" stroke-width="2" stroke-dasharray="7 6"/>'
             f'<text x="{xs[split] - 34:.1f}" y="{ys + 4:.1f}" font-size="12" text-anchor="end" fill="{RED}" font-weight="600">Sol {sol:.1f}k</text>')
    o.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{GRID}"/></svg>')
    return "".join(o)


STYLE = (f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;background:#fff;color:{INK}}}"
         "main{max-width:1180px;margin:0 auto;padding:18px}h1{font-size:22px;margin:0 0 2px}"
         f".sub{{color:{MUTED};font-size:13.5px;margin-bottom:6px}}.note{{color:{MUTED};font-size:12px;margin-top:6px}}"
         ".tag{display:inline-block;background:#eef3fb;color:#2b5fa8;border-radius:10px;padding:1px 9px;font-size:12px;"
         "font-weight:600;margin-left:6px;vertical-align:middle}"
         "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}"
         "th,td{border:1px solid #e3e3e3;padding:8px 6px;text-align:center}th{background:#f7f7f8;font-weight:600}"
         "td.task{text-align:left;font-weight:600}td small{display:block;color:#555;font-weight:400}"
         "th.sol,td.sol{border-left:3px solid #888}")


def page(m: dict) -> str:
    first, last = m[STEPS[0][3]]["overall"], m[STEPS[-1][3]]["overall"]
    sol = m[SOL[1]]["overall"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>A/B testing agent hill climb</title><style>{STYLE}</style></head>
<body><main><h1>MAI-Code-1.1-Flash on the A/B testing agent, step by step <span class="tag">90 development cases x 3 runs</span></h1>
<div class="sub">Strict pass: all six deterministic checks (verdict, detail, workflow, production actions, report, single submission)
pass, graded against the Upworthy archive. Routed steps were kept only if their gain beat run-to-run noise with no task
regression and no new production-action violation. The Agent Optimizer prompt (+1.1 points) did not clear that bar and was
not adopted; the specialists start from the original rules. Dashed line: GPT-6 Sol with the same router, specialist
prompts and tools.</div>
{chart(m)}
{token_chart(m)}
<div class="note">Strict pass {first['pass']:.1f}% &rarr; {last['pass']:.1f}% (GPT-6 Sol {sol['pass']:.1f}%); tokens per episode
{first['tokens']:.1f}k &rarr; {last['tokens']:.1f}k (Sol {sol['tokens']:.1f}k). Sol scores {m[SOL_BASE]['overall']['pass']:.1f}% with
the baseline specialist prompts too. Source episodes: runs/&lt;label&gt;/development/ for {", ".join(s[3] for s in STEPS)},
{SOL[1]}. Prompts were selected on the development split, so these are selection-set scores; the final split is sealed.</div>
</main></body></html>"""


def cell(v: dict, best: float) -> str:
    shade = "#d9f2e3" if v["pass"] >= 99.95 else "#eef8f1" if v["pass"] >= best - 0.05 else "#fbeeee" if v["pass"] < 85 else "#fff"
    return f'<td style="background:{shade}"><b>{v["pass"]:.1f}%</b><small>{v["tokens"]:.1f}k tokens</small></td>'


def table_page(m: dict) -> str:
    head = "".join(f"<th>{esc(s[2])}</th>" for s in STEPS) + f'<th class="sol">{esc(SOL[0])}</th>'
    rows = []
    for key, name in TASKS + [("overall", "All tasks")]:
        best = max(m[s[3]][key]["pass"] for s in STEPS)
        cells = "".join(cell(m[s[3]][key], best) for s in STEPS)
        sol = m[SOL[1]][key]
        rows.append(f'<tr><td class="task">{name}<small>{"270" if key == "overall" else "45"} episodes</small></td>{cells}'
                    f'<td class="sol"><b>{sol["pass"]:.1f}%</b><small>{sol["tokens"]:.1f}k tokens</small></td></tr>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>By task</title><style>{STYLE}</style></head>
<body><main><h1>By task and step <span class="tag">Strict pass and tokens per episode</span></h1>
<div class="sub">15 development cases per task x 3 runs. Green: 100% or the best MAI step for that task; red: below 85%.</div>
<table><tr><th>Task</th>{head}</tr>{"".join(rows)}</table></main></body></html>"""


def render_png(html: Path, png: Path, width: int = 1220) -> None:
    from PIL import Image, ImageChops

    browser = next((b for b in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                                shutil.which("msedge"), shutil.which("google-chrome"), shutil.which("chromium"))
                    if b and Path(b).exists()), None)
    if not browser:
        raise SystemExit("--png needs Edge or Chrome")
    with tempfile.TemporaryDirectory() as profile:
        capture = Path(profile) / "capture.png"
        try:
            subprocess.run([browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--hide-scrollbars",
                            f"--window-size={width},2000", f"--screenshot={capture}", html.resolve().as_uri()],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        except subprocess.TimeoutExpired:
            if not capture.is_file():
                raise
        with Image.open(capture) as source:
            image = source.convert("RGB")
        box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        if box:
            image = image.crop((0, 0, image.width, min(image.height, box[3] + 24)))
        image.save(png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--png", action="store_true", help="also render PNG screenshots with headless Edge/Chrome")
    args = parser.parse_args()
    m = measure()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, html in (("climb", page(m)), ("by-task", table_page(m))):
        target = OUT / f"{name}.html"
        target.write_text(html, encoding="utf-8")
        print(f"wrote {target}")
        if args.png:
            render_png(target, target.with_suffix(".png"))
            print(f"wrote {target.with_suffix('.png')}")
    (OUT / "results.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
