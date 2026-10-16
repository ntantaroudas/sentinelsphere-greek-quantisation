#!/usr/bin/env bash
# Paper C, evaluate each model on the English and Greek sets, one at a time.
# Each model file is downloaded, evaluated, then deleted to stay within the disk budget.
# Results accumulate in ../results/mcq_<label>.jsonl. Resumable, already downloaded or
# already scored items are skipped.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="${MODELS_DIR:-/home/claude/models/paperC}"
THREADS="${THREADS:-2}"
mkdir -p "$MODELS_DIR" "$HERE/../results"

# label  repo  filename
MATRIX=$(cat <<'EOF'
phi4mini_Q2_K bartowski/microsoft_Phi-4-mini-instruct-GGUF microsoft_Phi-4-mini-instruct-Q2_K.gguf
phi4mini_Q4_K_M bartowski/microsoft_Phi-4-mini-instruct-GGUF microsoft_Phi-4-mini-instruct-Q4_K_M.gguf
phi4mini_Q8_0 bartowski/microsoft_Phi-4-mini-instruct-GGUF microsoft_Phi-4-mini-instruct-Q8_0.gguf
qwen2.5-3b_Q4_K_M Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf
krikri-8b_Q4_K_M ilsp/Llama-Krikri-8B-Instruct-GGUF llama-krikri-8b-instruct-q4_k_m.gguf
EOF
)

echo "$MATRIX" | while read -r label repo fname; do
  [ -z "$label" ] && continue
  if grep -q "\"label\": \"$label\", \"summary\"" "$HERE/../results/mcq_${label}.jsonl" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] $label already complete, skipping"; continue
  fi
  f="$MODELS_DIR/$label.gguf"
  echo "[$(date +%H:%M:%S)] downloading $label"
  curl -sS -L --retry 3 -o "$f" "https://huggingface.co/$repo/resolve/main/$fname" || { echo "download failed $label"; rm -f "$f"; continue; }
  echo "[$(date +%H:%M:%S)] evaluating $label ($(du -h "$f" | cut -f1))"
  python3 "$HERE/eval_mcq.py" "$f" "$label" --threads "$THREADS"
  rm -f "$f"
  echo "[$(date +%H:%M:%S)] done $label, file removed"
done
echo "[$(date +%H:%M:%S)] all models complete"
