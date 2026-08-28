set -u

# -----------------------------------------------------------------------------
# Paths (Relative to repository root)
# -----------------------------------------------------------------------------
VARIANTS_DIR="lavis/projects/blip2/eval_variants_msrvtt"
OUTROOT="hyper_output/grid_msrvtt_variants"
CSV="${OUTROOT}/results.csv"
LOCK="${OUTROOT}/.csv.lock"

mkdir -p "$OUTROOT"

shopt -s nullglob
BASE_SCRIPTS=("$VARIANTS_DIR"/*.yaml)
shopt -u nullglob

if [ ${#BASE_SCRIPTS[@]} -eq 0 ]; then
    echo "[Error] No YAML files found in ${VARIANTS_DIR}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
GPUS=(0 1)
BASE_PORT=29501

# -----------------------------------------------------------------------------
# Hyperparameter Search Space
# -----------------------------------------------------------------------------
BATCHES=(16)
BEAMS=(5)
LENPENS=(0.7 0.85 1.0 1.15 1.3)
REPPENS=(0.7 0.85 1.0 1.15 1.3)
MINLENS=(5)
MAXLENS=(50)

# -----------------------------------------------------------------------------
# CSV Initialization
# -----------------------------------------------------------------------------
if [ ! -f "$CSV" ]; then
    echo "variant,batch,beam,lp,rp,min,max,bleu4,meteor,rouge,cider,spice" > "$CSV"
fi

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
get_metric() {
    # Extract metric value from log file
    grep -E "$2:" "$1" 2>/dev/null \
        | tail -1 \
        | awk -F: '{print $NF}' \
        | tr -d ' '
}

already_done() {
    local variant=$1 batch=$2 beam=$3 lp=$4 rp=$5
    [ -f "$CSV" ] || return 1

    # Check if this exact configuration has already succeeded in the CSV
    awk -F, -v v="$variant" -v b="$batch" -v be="$beam" -v lpv="$lp" -v rpv="$rp" '
        NR == 1 { next }
        {
            gsub(/\r/, "")
            # Columns: 1=variant, 2=batch, 3=beam, 4=lp, 5=rp, 12=cider (index 12 or 11 depending on split)
            if ($1 == v && ($2+0) == (b+0) && ($3+0) == (be+0) &&
                ($4+0) == (lpv+0) && ($5+0) == (rpv+0)) {
                if ($12 != "" && $12 != "NA") { found = 1; exit }
            }
        }
        END { exit (found ? 0 : 1) }
    ' "$CSV"
}

write_variant_cfg() {
    local base_yaml=$1
    local out_yaml=$2
    local batch=$3
    local beam=$4
    local lenp=$5
    local repp=$6
    local minl=$7
    local maxl=$8
    local outdir=$9

    cp "$base_yaml" "$out_yaml"

    # Safely update generation hyperparameters inside the YAML file via Python
    python3 -c "
import yaml

with open('$out_yaml', 'r') as f:
    config = yaml.safe_load(f)

config['run']['batch_size_eval'] = int($batch)
config['run']['num_beams'] = int($beam)
config['run']['length_penalty'] = float($lenp)
config['run']['repetition_penalty'] = float($repp)
config['run']['min_len'] = int($minl)
config['run']['max_len'] = int($maxl)
config['run']['output_dir'] = '$outdir'

with open('$out_yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
"
}

# -----------------------------------------------------------------------------
# Evaluation Job Execution
# -----------------------------------------------------------------------------
run_job() {
    local gpu=$1
    local port=$2
    local base_script=$3
    local variant_name=$4
    local batch=$5
    local beam=$6
    local lenp=$7
    local repp=$8
    local minl=$9
    local maxl=${10}

    local tag="${variant_name}_b${batch}_be${beam}_lp${lenp}_rp${repp}"
    local outdir="${OUTROOT}/${tag}"
    local cfg="${outdir}/eval.yaml"

    mkdir -p "$outdir"

    write_variant_cfg "$base_script" "$cfg" "$batch" "$beam" "$lenp" "$repp" "$minl" "$maxl" "$outdir"

    CUDA_VISIBLE_DEVICES=$gpu \
    torchrun \
        --nnodes=1 \
        --nproc_per_node=1 \
        --master_addr=127.0.0.1 \
        --master_port=${port} \
        evaluate.py \
        --cfg-path "$cfg" \
        > "${outdir}/eval.log" 2>&1

    local LOG="${outdir}/eval.log"
    local BLEU4 METEOR ROUGE CIDER SPICE
    BLEU4=$(get_metric "$LOG" "BLEU-4")
    METEOR=$(get_metric "$LOG" "METEOR")
    ROUGE=$(get_metric "$LOG" "ROUGE")
    CIDER=$(get_metric "$LOG" "CIDER")
    SPICE=$(get_metric "$LOG" "SPICE")

    # Thread-safe write to CSV
    (
        flock 200
        echo "\"${variant_name}\",${batch},${beam},${lenp},${repp},${minl},${maxl},${BLEU4:-NA},${METEOR:-NA},${ROUGE:-NA},${CIDER:-NA},${SPICE:-NA}" \
            >> "$CSV"
    ) 200>"$LOCK"

    echo "[done] gpu${gpu} | ${tag} | CIDEr=${CIDER:-NA}"
}

# -----------------------------------------------------------------------------
# Scheduler Loop
# -----------------------------------------------------------------------------
declare -A PID2GPU
FREE=("${GPUS[@]}")

pop_free() {
    POPPED=${FREE[0]}
    FREE=("${FREE[@]:1}")
}

for base_script in "${BASE_SCRIPTS[@]}"; do
    # Extract variant name without path and extension (e.g., Aligned_6)
    variant_name=$(basename "$base_script" .yaml)

    echo "=== Processing variant: ${variant_name} ==="

    for batch in "${BATCHES[@]}"; do
        for beam in "${BEAMS[@]}"; do
            for lp in "${LENPENS[@]}"; do
                for rp in "${REPPENS[@]}"; do

                    if already_done "$variant_name" "$batch" "$beam" "$lp" "$rp"; then
                        echo "[skip] ${variant_name} b=${batch} beam=${beam} lp=${lp} rp=${rp}"
                        continue
                    fi

                    # Wait for an available GPU
                    while [[ ${#FREE[@]} -eq 0 ]]; do
                        done_pid=
                        wait -n -p done_pid
                        if [[ -n "${PID2GPU[$done_pid]:-}" ]]; then
                            FREE+=("${PID2GPU[$done_pid]}")
                            unset 'PID2GPU[$done_pid]'
                        fi
                    done

                    pop_free
                    gpu=$POPPED
                    port=$((BASE_PORT + gpu))

                    run_job \
                        "$gpu" \
                        "$port" \
                        "$base_script" \
                        "$variant_name" \
                        "$batch" \
                        "$beam" \
                        "$lp" \
                        "$rp" \
                        5 \
                        50 &

                    PID2GPU[$!]=$gpu

                done
            done
        done
    done
done

# Wait for all remaining background jobs to finish
wait

echo ""
echo "Grid search completed successfully."
echo "Results saved to: $CSV"