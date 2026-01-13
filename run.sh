#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -------------------------
# Define case here
# -------------------------
GPUS="0"             # CUDA_VISIBLE_DEVICES
MAIN="InternLM7b"          # main model name
ASSIST=""            # assist model(s), e.g., "OpenChat" or "OpenChat,Llama3"; empty for single model
ALIGN="unite"        # align_method: "mined", "unite", "gac", "eva"
VARIANT="vanilla"    # "vanilla", "consist-rbf", "consist-linear"
TASK="TriviaQA"           # "GSM8K", "SAMSum", "NQ", "PIQA", "TriviaQA"
RUN_MODE="test"      # "test" / "dev"

OUT_DIR="out"


# -------------------------
# Select evaluator by TASK
# -------------------------
RUN_SCRIPT=""
case "${TASK}" in
  NQ|TriviaQA|PIQA|GSM8K|SAMSum)
    RUN_SCRIPT="src/run.py"
    ;;
  MMLU)
    RUN_SCRIPT="src/run_mmlu.py"
    ;;
  *)
    echo "[ERROR] Unsupported TASK='${TASK}'. Use: NQ, TriviaQA, MMLU, PIQA, SAMSum, GSM8K." >&2
    exit 1
    ;;
esac


# -------------------------
# Helpers
# -------------------------
normalize_assist() {
  local s="$1"
  # remove spaces
  s="$(echo "${s}" | tr -d '[:space:]')"
  # remove smart quotes if user accidentally wrote: ASSIST=“”
  s="${s//“/}"
  s="${s//”/}"
  # also remove plain quotes if any got into the value
  s="${s//\"/}"
  echo "${s}"
}


# -------------------------
# Helper: build output file path (your requested convention)
# 1) single: out/TASK/MAIN/single.out
# 2) multi : out/TASK/MAIN+ASSIST/ALIGN-VARIANT.out
# -------------------------
build_outfile() {
  local main="$1" assist="$2" align="$3" variant="$4" task="$5" run_mode="$6"
  local assist_tag models dir file

  if [[ -z "${assist}" ]]; then
    dir="${OUT_DIR}/${task}/${run_mode}/${main}"
    file="single.out"
  else
    if [[ -z "${align}" ]]; then
      echo "[ERROR] ASSIST is set but ALIGN is empty. Please set ALIGN for multi-model runs." >&2
      exit 1
    fi
    assist_tag="${assist//,/+}"          # "OpenChat,Llama3" -> "OpenChat+Llama3"
    models="${main}+${assist_tag}"       # "InternLM7b+OpenChat(+Llama3)"
    dir="${OUT_DIR}/${task}/${run_mode}/${models}"
    file="${align}-${variant}.out"       # "unite-vanilla.out"
  fi

  mkdir -p "${dir}"
  echo "${dir}/${file}"
}

# -------------------------
# Helper: run one job
# -------------------------
run_job() {
  local gpus="$1"
  local main="$2"
  local assist_raw="$3"
  local align="$4"

  local assist
  assist="$(normalize_assist "${assist_raw}")"

  local outfile
  outfile="$(build_outfile "${main}" "${assist}" "${align}" "${VARIANT}" "${TASK}" "${RUN_MODE}")"

  local -a args
  args=(python "${RUN_SCRIPT}"
        --main_model "${main}"
        --variant "${VARIANT}"
        --task "${TASK}"
        --run_mode "${RUN_MODE}")

  # Only multi-model runs need assist_model + align_method
  if [[ -n "${assist}" ]]; then
    args+=(--assist_model "${assist}")
    if [[ -z "${align}" ]]; then
      echo "[ERROR] ASSIST is set but ALIGN is empty. Please set ALIGN for multi-model runs." >&2
      exit 1
    fi
    args+=(--align_method "${align}")
  fi

  echo "[INFO] CUDA_VISIBLE_DEVICES=${gpus}"
  echo "[INFO] ${args[*]}"
  echo "[INFO] -> ${outfile}"

  nohup env CUDA_VISIBLE_DEVICES="${gpus}" "${args[@]}" > "${outfile}" 2>&1 &
  echo "[INFO] Started PID=$!  log=${outfile}"
}


run_job "${GPUS}" "${MAIN}" "${ASSIST}" "${ALIGN}"
echo "[INFO] All jobs submitted."
