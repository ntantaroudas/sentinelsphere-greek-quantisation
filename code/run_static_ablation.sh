#!/usr/bin/env bash
# Paper C ablation on the four bit setting of Phi-4-mini.
#   phi4mini_Q4_K_M_deployed  the Q4_K_M file that the SentinelSphere chatbot publishes on Hugging Face
#                             (daskalos-apps/phi4-cybersec-Q4_K_M), same tensor layout as the community
#                             file but quantised separately and with no importance matrix recorded
#   phi4mini_Q4_K_M_static    static Q4_K_M requantised by us from the Q8_0 file, no importance matrix
#   phi4mini_Q2_K_static      static Q2_K requantised by us from the Q8_0 file, no importance matrix
# Waits for the main pipeline to finish so that the CPU is idle, evaluates each file on the same
# English and Greek items, deletes the model files and reruns the analysis.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MD="${MODELS_DIR:-/home/claude/models/paperC}"
mkdir -p "$MD"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
done_label(){ grep -q "\"label\": \"$1\", \"summary\"" "$HERE/../results/mcq_$1.jsonl" 2>/dev/null; }
if [ -f /home/claude/ss/run_rest.log ]; then
  log "waiting for the main pipeline to finish"
  while ! grep -q "ALL DONE" /home/claude/ss/run_rest.log; do sleep 30; done
fi

lab=phi4mini_Q4_K_M_deployed
if ! done_label $lab; then
  log "downloading the deployed SentinelSphere file"
  curl -sS -L --retry 3 -o "$MD/$lab.gguf" \
    "https://huggingface.co/daskalos-apps/phi4-cybersec-Q4_K_M/resolve/main/phi4-mini-instruct-Q4_K_M.gguf"
  log "evaluating $lab"
  python3 "$HERE/eval_mcq.py" "$MD/$lab.gguf" "$lab" --threads 2
  rm -f "$MD/$lab.gguf"
fi

if ! done_label phi4mini_Q4_K_M_static || ! done_label phi4mini_Q2_K_static; then
  SRC="$MD/phi4mini_Q8_0_src.gguf"
  log "downloading Q8_0 source"
  [ -f "$SRC" ] || curl -sS -L --retry 3 -o "$SRC" \
    "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q8_0.gguf"
  log "requantising"
  done_label phi4mini_Q4_K_M_static || python3 "$HERE/requantise_static.py" "$SRC" "$MD/phi4mini_Q4_K_M_static.gguf" Q4_K_M
  done_label phi4mini_Q2_K_static || python3 "$HERE/requantise_static.py" "$SRC" "$MD/phi4mini_Q2_K_static.gguf" Q2_K
  rm -f "$SRC"
  for lab in phi4mini_Q4_K_M_static phi4mini_Q2_K_static; do
    done_label $lab && continue
    log "evaluating $lab"
    python3 "$HERE/eval_mcq.py" "$MD/$lab.gguf" "$lab" --threads 2
    rm -f "$MD/$lab.gguf"
  done
fi
cd "$HERE" && python3 analyse.py > ../results/analyse_out.txt 2>&1
log "STATIC ABLATION DONE"
