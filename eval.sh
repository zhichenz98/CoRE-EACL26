#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -------------------------
# Define case here
# -------------------------
TASK="TriviaQA"                 # NQ / TriviaQA / MMLU / PIQA / SAMSum / GSM8K
RUN_MODE="test"
ALIGN="unite"
VARIANT="vanilla"
RES_DIR="./res"

MAIN="InternLM7b"
ASSIST=""                    # empty => single model; multi => "A,B,C" or "A+B+C"


# -------------------------
# Select evaluator by TASK
# -------------------------
EVAL_SCRIPT=""
case "${TASK}" in
  NQ|TriviaQA|MMLU|PIQA)
    EVAL_SCRIPT="src/evaluate/EM_dir_test_arg.py"
    ;;
  SAMSum)
    EVAL_SCRIPT="src/evaluate/rouge_dir_test_arg.py"
    ;;
  GSM8K)
    EVAL_SCRIPT="src/evaluate/GSM_dir_test_arg.py"
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
  s="$(echo "${s}" | tr -d '[:space:]')"
  s="${s//“/}"
  s="${s//”/}"
  s="${s//\"/}"
  # allow "A,B,C" -> "A+B+C"
  s="${s//,/+}"
  echo "${s}"
}

# Build the prediction file path that evaluator should read.
build_pred_file() {
  local main="$1" assist="$2" align="$3"

  if [[ -z "${assist}" ]]; then
    echo "${RES_DIR}/${TASK}/${RUN_MODE}/${main}/single.jsonl"
  else
    local models="${main}+${assist}"
    echo "${RES_DIR}/${TASK}/${RUN_MODE}/${models}/${align}-${VARIANT}.jsonl"
  fi
}

# -------------------------
# Run evaluation
# -------------------------
run_eval() {
  local main="$1"
  local assist_raw="$2"
  local align="$3"

  local assist
  assist="$(normalize_assist "${assist_raw}")"

  # Determine which prediction file to evaluate
  local pred_file
  pred_file="$(build_pred_file "${main}" "${assist}" "${align}")"

  if [[ ! -f "${pred_file}" ]]; then
    echo "[ERROR] Prediction file not found: ${pred_file}" >&2
    echo "[HINT] Check RES_DIR/TASK/RUN_MODE/MAIN/ASSIST/ALIGN/VARIANT settings." >&2
    exit 1
  fi

  # Base args
  local -a args
  args=(python "${EVAL_SCRIPT}"
        --res_dir "${RES_DIR}"
        --task "${TASK}"
        --run_mode "${RUN_MODE}"
        --main_model "${main}"
        --variant "${VARIANT}")

  # Only multi-model runs should include assist_model + align_method
  if [[ -n "${assist}" ]]; then
    if [[ -z "${align}" ]]; then
      echo "[ERROR] ASSIST is set but ALIGN is empty. Please set ALIGN for multi-model evaluation." >&2
      exit 1
    fi
    args+=(--assist_model "${assist}" --align_method "${align}")
  fi

  echo "[INFO] Evaluating: ${pred_file}"
  echo "[INFO] ${args[*]}"

  # Also expose as env var for easy access inside python if you want:
  # In python, you can read os.environ["PRED_FILE"].
  PRED_FILE="${pred_file}" "${args[@]}"
}

run_eval "${MAIN}" "${ASSIST}" "${ALIGN}"
