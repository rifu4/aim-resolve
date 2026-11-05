#!/bin/bash
# Run the aim-resolve pipeline with `base_detect.method: manual` and let headless Claude Code
# answer each detection task: snakemake runs until a detect step stops for its submission,
# then one short `claude -p` call reviews agent/<it>_task/ and writes agent/<it>_submission.json
# (the Claude process ends afterwards), and snakemake is started again. Repeats until done.
#
# usage (from anywhere; best inside tmux/screen so it survives logging out):
#   steering/run_with_claude.sh <config.yml> <cuda_device>
#   e.g. tmux new -s agent 'steering/run_with_claude.sh config/fast_sf.yml 0'
#
# environment (optional):
#   CLAUDE_BIN       path of the claude executable (default: `claude` on PATH, else the one
#                    shipped with the VS Code extension)
#   CLAUDE_TIMEOUT   seconds per detection task (default 3600)
#   CLAUDE_MODEL     model for the detection tasks (default: your Claude Code default)
#   THREADS          CPU threads of the pre-fit/reconstruction steps (default 16, snakemake --cores)
#
# Claude runs with --permission-mode dontAsk: it may only read files, run the detection tools
# (python -m aim_resolve.agent_detect ...) and write the one submission file. Usage counts
# against your Claude plan; if a call fails (limit, timeout, invalid submission) the wrapper
# stops and the run waits at the detect step.
set -uo pipefail

CFG=${1:?usage: $0 <config.yml> <cuda_device>}
GPU=${2:?usage: $0 <config.yml> <cuda_device>}
STEERING=$(cd "$(dirname "$0")" && pwd)
cd "$STEERING"

PIXI=~/.pixi/bin/pixi
PY=$(cd .. && pwd)/.pixi/envs/default/bin/python
CLAUDE_BIN=${CLAUDE_BIN:-$(command -v claude || ls -d ~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1)}
CLAUDE_TIMEOUT=${CLAUDE_TIMEOUT:-3600}
THREADS=${THREADS:-16}
MODEL_OPT=()
[ -n "${CLAUDE_MODEL:-}" ] && MODEL_OPT=(--model "$CLAUDE_MODEL")

[ -x "$CLAUDE_BIN" ] || { echo "claude executable not found, set CLAUDE_BIN" >&2; exit 1; }
ODIR=$(cd "$STEERING" && "$PY" -c "from aim_resolve import yaml_load; print(yaml_load('$CFG')['odir'])" 2>/dev/null)
[ -n "$ODIR" ] || { echo "could not read odir from $CFG" >&2; exit 1; }
ODIR_ABS=$STEERING/$ODIR
LOG=${ODIR_ABS}_snakemake.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for round in $(seq 1 50); do
    say "starting snakemake (round $round): $CFG on GPU $GPU"
    # --nolock: several runs (each with its own odir) share this working directory, and the
    # directory lock of one Snakemake process would otherwise stop the others
    if "$PIXI" run snakemake --nolock --cores "$THREADS" --config file="$CFG" cuda_device="$GPU" threads="$THREADS" >> "$LOG" 2>&1; then
        say "pipeline finished"
        exit 0
    fi

    # find the detection task waiting for its submission
    pending=""
    for t in "$ODIR_ABS"/agent/*_task; do
        [ -f "$t/task.md" ] || continue
        it=$(basename "$t" _task)
        [ -f "$ODIR_ABS/agent/${it}_submission.json" ] || { pending=$it; task_dir=$t; }
    done
    if [ -z "$pending" ]; then
        say "snakemake failed and no detection task is waiting: see $LOG (stopping)"
        exit 1
    fi

    SUB=$ODIR_ABS/agent/${pending}_submission.json
    REC=$ODIR_ABS/files/$((pending - 1))_rec.pkl
    CLOG=$task_dir/claude.log
    say "detection task $pending waiting: running claude -p (log: $CLOG)"

    PROMPT="You are the source detection agent of the aim-resolve pipeline. Work fully autonomously,
nobody can answer questions.

Read the task description $task_dir/task.md and the overview image $task_dir/overview.png
(the reconstruction is $REC). Follow the instructions in task.md exactly: decide the extended
objects (bounding boxes), review ALL compact candidates (accept/reject), decide the lobe groups,
and add missing compact sources if you see any.

Tools: run them exactly in this form (the full python path is required), and write all images
into $task_dir/:
  $PY -m aim_resolve.agent_detect sheet $REC --ids ID ID ... --out $task_dir/sheet_N.png
  $PY -m aim_resolve.agent_detect view $REC --box I0 I1 J0 J1 --out $task_dir/zoom_N.png
  $PY -m aim_resolve.agent_detect peaks $REC --box I0 I1 J0 J1 --min-rel-flux 0.001
  $PY -m aim_resolve.agent_detect stats $REC --box I0 I1 J0 J1
Keep the --n-orders/--min-contrast/--merge-dist options exactly as given in task.md. Review the
candidates with contact sheets of up to 40 ids each, then look at the images with the Read tool.

Finally write the submission as JSON (schema in task.md, fields objects, groups, compact_ids,
extra_compact, with short notes) with the Write tool to:
  $SUB
That file is the only file you may write. End with a short summary of your decisions
(objects, number of accepted/rejected candidates, groups, borderline cases)."

    # run claude without the environment of an outer Claude Code session (if any)
    unset_vars=$(env | grep -oE '^CLAUDE[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')
    # shellcheck disable=SC2086
    env $unset_vars timeout "$CLAUDE_TIMEOUT" "$CLAUDE_BIN" -p "$PROMPT" "${MODEL_OPT[@]}" \
        --permission-mode dontAsk \
        --add-dir "$ODIR_ABS" \
        --allowedTools "Read" "Glob" "Grep" \
            "Bash($PY -m aim_resolve.agent_detect *)" \
            "Edit(/$SUB)" \
        > "$CLOG" 2>&1 < /dev/null
    rc=$?
    say "claude finished with exit code $rc"
    if [ $rc -ne 0 ] || [ ! -f "$SUB" ]; then
        say "no submission written for task $pending (see $CLOG); stopping, the run waits at detect step $pending"
        exit 1
    fi
done
say "too many rounds, stopping"
exit 1
