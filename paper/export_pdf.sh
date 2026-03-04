#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_INPUT="${ROOT_DIR}/AAL-SD_ Agent-Augmented Active Learning for Landslide Detection in Remote Sensing Imagery_中文.md"

INPUT_MD="${1:-$DEFAULT_INPUT}"
OUTPUT_PDF="${2:-${ROOT_DIR}/AAL-SD_ Agent-Augmented Active Learning for Landslide Detection in Remote Sensing Imagery_中文.pdf}"

if [[ ! -f "${INPUT_MD}" ]]; then
  echo "Input markdown not found: ${INPUT_MD}" >&2
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx not found. Please install Node.js (npm/npx) first." >&2
  exit 1
fi

CHROME_BIN=""
if command -v google-chrome >/dev/null 2>&1; then
  CHROME_BIN="$(command -v google-chrome)"
elif command -v chromium >/dev/null 2>&1; then
  CHROME_BIN="$(command -v chromium)"
elif [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fi

if [[ -z "${CHROME_BIN}" ]]; then
  echo "Chrome/Chromium not found. Please install Google Chrome or Chromium." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

BODY_HTML="${TMP_DIR}/body.html"
WRAPPED_HTML="${TMP_DIR}/paper.html"

npx --yes marked --gfm -i "${INPUT_MD}" -o "${BODY_HTML}"

BASE_URL="file://${ROOT_DIR}/"

cat > "${WRAPPED_HTML}" <<'HTML'
<!doctype html>
<html lang="zh">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <base href="__BASE_URL__" />
    <style>
      @page { size: A4; margin: 20mm; }
      body {
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif;
        line-height: 1.55;
        font-size: 12pt;
        color: #111;
      }
      h1, h2, h3, h4 { page-break-after: avoid; }
      pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      pre { white-space: pre-wrap; word-break: break-word; }
      img { max-width: 100%; height: auto; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #bbb; padding: 6px 8px; vertical-align: top; }
      blockquote { border-left: 4px solid #ddd; padding-left: 12px; color: #333; }
    </style>
    <script>
      window.MathJax = {
        tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] },
        options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
      };
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <title>AAL-SD 中文版</title>
  </head>
  <body>
__BODY_HTML__
  </body>
</html>
HTML

python - <<PY
from pathlib import Path

wrapped = Path("${WRAPPED_HTML}")
body = Path("${BODY_HTML}").read_text(encoding="utf-8")
html = wrapped.read_text(encoding="utf-8")
html = html.replace("__BASE_URL__", "${BASE_URL}")
html = html.replace("__BODY_HTML__", body)
wrapped.write_text(html, encoding="utf-8")
PY

"${CHROME_BIN}" \
  --headless=new \
  --disable-gpu \
  --no-sandbox \
  --allow-file-access-from-files \
  --virtual-time-budget=20000 \
  --print-to-pdf-no-header \
  --print-to-pdf="${OUTPUT_PDF}" \
  "${WRAPPED_HTML}"

echo "Exported PDF: ${OUTPUT_PDF}"
