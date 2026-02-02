#!/bin/bash

set -euo pipefail

readonly OUTPUT_DIR="./output"
readonly CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint"
readonly EVAL_OUTPUT_DIR="$OUTPUT_DIR/eval"
readonly CONFIG_FILE="conf/settings.yaml"
readonly TARGETS_DIR="./targets"
readonly CONF_DIR="./conf"
readonly DATA_DIR="./data"
readonly LOG_DIR="./wandb"
readonly SRC_DIR="./src"

if [[ -t 1 ]]; then
    readonly RED=$(tput setaf 1)
    readonly GREEN=$(tput setaf 2)
    readonly YELLOW=$(tput setaf 3)
    readonly BLUE=$(tput setaf 4)
    readonly BOLD=$(tput bold)
    readonly NC=$(tput sgr0)
else
    readonly RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

log() {
    local level="$1" color="$2" msg="$3" timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${color}${BOLD}[${level}]${NC} ${timestamp} $msg" >&2
}

log_info()    { log "INFO"    "$BLUE"   "$1"; }
log_success() { log "SUCCESS" "$GREEN"  "$1"; }
log_warning() { log "WARNING" "$YELLOW" "$1"; }
log_error()   { log "ERROR"   "$RED"    "$1"; exit 1; }

show_help() {
cat <<EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
  --generate              Run dataset generation
  --train                 Run model training
  --eval                  Run evaluation
  --clean                 Remove generated outputs and caches
  -h, --help              Show this help message

Supported Datasets:
  gsm8k            GSM8K
  aime24           AIME24
  aime25           AIME25
  aqua             AQUA-RAT
  mmlupro          MMLU-Pro
  truthfulqa       TruthfulQA
  strategyqa       StrategyQA
  metamathqa       MetaMathQA
  commonsenseqa    CommonsenseQA
  
Generator Options:
  --dataset DATASET       Dataset to generate (see Supported Datasets)
  --mode MODE             Generation mode: train | test
  --api-key KEY           LLM API key
  --api-base URL          LLM API endpoint
  --model MODEL           Model name
  --num-examples NUM      Number of examples to generate
  --dry-run               Validate configuration without API calls

Training Options:
  --dataset DATASET       Dataset for training
  --model-path MODEL      Base model or checkpoint path
  --learning-rate LR      Learning rate
  --num-train NUM         Number of training examples
  --checkpoint-dir DIR    Output checkpoint directory

Evaluation Options:
  --dataset DATASET       Dataset for evaluation
  --model-path MODEL      Trained model checkpoint
  --num-test NUM          Number of test examples

Examples:
  $0 --generate --dataset truthfulqa --mode train
  $0 --train --dataset truthfulqa --model-path Qwen/Qwen3-4B
  $0 --eval --dataset truthfulqa --model-path $CHECKPOINT_DIR

EOF
}

parse_args() {
    local args=()
    local cuda_device=""
    
    while (( $# > 0 )); do
        case "$1" in
            --config)
                CONFIG_FILE="$2"
                args+=("$1" "$2")
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                args+=("$1")
                if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                    args+=("$2")
                    shift
                fi
                shift
                ;;
        esac
    done
    
    printf '%s\n' "${args[@]}"
}

check_config_file() {
    if [[ ! -f "${CONFIG_FILE:-}" ]]; then
        log_warning "Config file not found: $CONFIG_FILE"
        log_warning "Using command-line arguments only"
        return 1
    fi
    return 0
}

setup_directories() {
    local dirs=(
        "$CHECKPOINT_DIR"
        "$DATA_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        if mkdir -p "$dir" 2>/dev/null; then
            log_info "Created directory: $dir"
        fi
    done
}

check_dependencies() {
    local required_packages=("openai" "transformers" "yaml" "tqdm")
    local missing=()
    
    for pkg in "${required_packages[@]}"; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done
    
    if (( ${#missing[@]} > 0 )); then
        log_warning "Missing Python packages: ${missing[*]}"
        log_info "Install with: pip install ${missing[*]}"
        return 1
    fi
    
    return 0
}

validate_required_args() {
    local -n args_ref=$1
    local command=$2
    local missing=()
    
    case "$command" in
        --generate)
            [[ ! " ${args_ref[*]} " =~ "--dataset" ]] && missing+=("--dataset")
            [[ ! " ${args_ref[*]} " =~ "--mode" ]] && missing+=("--mode")
            ;;
        --eval)
            [[ ! " ${args_ref[*]} " =~ "--model" ]] && missing+=("--model")
            ;;
    esac
    
    if (( ${#missing[@]} > 0 )); then
        log_error "$command requires: ${missing[*]}"
        log_info "Using defaults from config if available"
        return 1
    fi
    return 0
}

run_command() {
    local command=$1
    shift
    local args=("$@")
    local python_script
    local operation
    
    case "$command" in
        --generate)
            python_script="$SRC_DIR/generator.py"
            operation="Dataset Generation"
            ;;
        --train)
            python_script="$SRC_DIR/train.py"
            operation="Training"
            ;;
        --eval)
            python_script="$SRC_DIR/eval.py"
            operation="Evaluation"
            ;;
        *)
            log_error "Unknown command: $command"
            return 1
            ;;
    esac
    
    if [[ ! -f "$python_script" ]]; then
        log_error "Python script not found: $python_script"
        return 1
    fi
    
    log_info "Starting $operation"
    log_info "Running: python $python_script ${args[*]}"
    
    if python "$python_script" "${args[@]}"; then
        log_success "$operation completed successfully!"
        return 0
    else
        log_error "$operation failed with exit code: $?"
        return 1
    fi
}

show_summary() {
    local command=$1
    shift
    local args=("$@")
    
    case "$command" in
        --generate)
            log_info "Generated datasets saved to: $DATA_DIR/"
            ;;
        --train)
            log_info "Model checkpoint saved to: $CHECKPOINT_DIR/"
            local latest_checkpoint
            latest_checkpoint=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
            [[ -n "$latest_checkpoint" ]] && log_info "Latest checkpoint: $(basename "$latest_checkpoint")"
            ;;
        --eval)
            local dataset=""
            for ((i=0; i<${#args[@]}; i++)); do
                if [[ "${args[i]}" == "--dataset" && $((i+1)) -lt ${#args[@]} ]]; then
                    dataset="${args[i+1]}"
                    break
                fi
            done
            
            if [[ -n "$dataset" ]]; then
                local result_dir="$EVAL_OUTPUT_DIR/$dataset"
                if [[ -d "$result_dir" ]]; then
                    local latest_result
                    latest_result=$(find "$result_dir" -name "*.json" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
                    
                    if [[ -f "$latest_result" ]]; then
                        local accuracy
                        accuracy=$(grep -o '"accuracy":[[:space:]]*[0-9.]*' "$latest_result" | head -1 | cut -d: -f2 | tr -d ' ')
                        if [[ -n "$accuracy" ]]; then
                            local percentage
                            percentage=$(awk -v acc="$accuracy" 'BEGIN {printf "%.2f%%", acc * 100}')
                            log_info "Latest accuracy: $percentage"
                        fi
                    fi
                fi
            fi
            ;;
    esac
}

clean() {
    log_info "Cleaning output and data directories..."
    
    local dirs_to_clean=("$OUTPUT_DIR" "$DATA_DIR" "$LOG_DIR" "$TARGETS_DIR")
    for dir in "${dirs_to_clean[@]}"; do
        [[ -e "$dir" ]] && rm -rf "$dir"
    done
    
    local py_dirs=("$SRC_DIR" "$CONF_DIR")
    for dir in "${py_dirs[@]}"; do
        [[ -d "$dir" ]] && find "$dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    done
    
    find . -type f -name "*.py[co]" -delete 2>/dev/null || true
    find . -type d -name "__pycache__" -delete 2>/dev/null || true
    
    log_success "Clean completed"
}

main() {
    if (( $# == 0 )); then
        show_help
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        -h|--help)
            show_help
            exit 0
            ;;
        --clean)
            clean
            exit 0
            ;;
        --generate|--train|--eval)
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
    
    setup_directories
    check_config_file
    if ! check_dependencies; then
        log_warning "Some dependencies are missing. Execution may fail."
    fi
    
    local args_array
    mapfile -t args_array < <(parse_args "$@")
    
    if ! validate_required_args args_array "$command"; then
        log_warning "Continuing with missing required arguments..."
    fi
    
    if run_command "$command" "${args_array[@]}"; then
        show_summary "$command" "${args_array[@]}"
    else
        log_error "Command failed"
        exit 1
    fi
}

main "$@"
