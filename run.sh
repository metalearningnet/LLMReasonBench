#!/bin/bash
set -euo pipefail

OS=$(uname -s)

readonly OUTPUT_DIR="./output"
readonly CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint"
readonly EVAL_OUTPUT_DIR="$OUTPUT_DIR/eval"
readonly CONFIG_FILE="conf/settings.yaml"
readonly TARGETS_DIR="./targets"
readonly VENV_DIR="./.venv"
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

check_os_support() {
    case "$OS" in
        Linux|Darwin)
            return 0
            ;;
        *)
            log_error "Unsupported operating system: $OS. This script only supports Linux and macOS."
            ;;
    esac
}

add_common_uv_paths() {
    case "$OS" in
        Linux)
            if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                export PATH="$HOME/.local/bin:$PATH"
            fi
            ;;
        Darwin)
            if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                export PATH="$HOME/.local/bin:$PATH"
            fi
            for p in "$HOME/Library/Python/3."*"/bin"; do
                if [[ -d "$p" && ":$PATH:" != *":$p:"* ]]; then
                    export PATH="$p:$PATH"
                fi
            done
            ;;
    esac
}

ensure_uv_installed() {
    add_common_uv_paths

    if command -v uv &>/dev/null; then
        return 0
    fi

    log_info "uv not found. Installing uv automatically..."

    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        log_error "Neither curl nor wget found. Please install curl or wget, or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi

    add_common_uv_paths

    if ! command -v uv &>/dev/null; then
        log_error "Failed to install uv. Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi

    log_info "uv installed successfully: $(uv --version)"
}

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
  alfworld         ALFWorld (multi‑turn interactive)

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
  --model MODEL           Base model or checkpoint path
  --num-train NUM         Number of training examples
  --mode MODE             Training mode: supervised | fixed_length | multiturn
  --rl MODE               RL training mode: dpo | cpo | kto | orpo (requires --train)

Evaluation Options:
  --dataset DATASET       Dataset for evaluation
  --model MODEL           Trained model checkpoint
  --num-test NUM          Number of test examples
  --interactive           (Multi‑turn only) Use interactive environment evaluation

Examples:
  # Single‑turn CoT generation and training
  $0 --generate --dataset truthfulqa --mode train
  $0 --train --dataset truthfulqa --model Qwen/Qwen3.5-9B
  $0 --train --rl dpo --dataset truthfulqa --model Qwen/Qwen3.5-9B
  $0 --eval --dataset truthfulqa --model $CHECKPOINT_DIR

  # Multi‑turn interactive dataset (ALFWorld)
  $0 --generate --dataset alfworld
  $0 --train --dataset alfworld --model Qwen/Qwen3.5-9B --mode multiturn
  $0 --eval --dataset alfworld --model $CHECKPOINT_DIR
  $0 --eval --dataset alfworld --model $CHECKPOINT_DIR --interactive

EOF
}

parse_args() {
    local args=()
    
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
    log_info "Dependencies will be resolved automatically by uv at runtime"
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
        --train)
            [[ ! " ${args_ref[*]} " =~ "--model" ]] && missing+=("--model")
            ;;
        --eval)
            [[ ! " ${args_ref[*]} " =~ "--model" ]] && missing+=("--model")
            ;;
    esac
    
    if (( ${#missing[@]} > 0 )); then
        log_warning "$command missing recommended arguments: ${missing[*]}"
        log_info "Will attempt to use defaults from config if available"
        return 1
    fi
    
    if [[ " ${args_ref[*]} " =~ "--rl" ]]; then
        local rl_mode=""
        for ((i=0; i<${#args_ref[@]}; i++)); do
            if [[ "${args_ref[i]}" == "--rl" && $((i+1)) -lt ${#args_ref[@]} ]]; then
                rl_mode="${args_ref[i+1]}"
                break
            fi
        done
        if [[ -n "$rl_mode" && "$rl_mode" != "dpo" && "$rl_mode" != "cpo" && "$rl_mode" != "kto" && "$rl_mode" != "orpo" ]]; then
            log_error "--rl mode must be 'dpo', 'cpo', 'kto', or 'orpo', got: $rl_mode"
            return 1
        fi
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
    
    ensure_uv_installed
    
    log_info "Starting $operation"
    log_info "Running: uv run python $python_script ${args[*]}"
    
    if uv run python "$python_script" "${args[@]}"; then
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
            
            if [[ " ${args[*]} " =~ "--rl" ]]; then
                local rl_mode=""
                for ((i=0; i<${#args[@]}; i++)); do
                    if [[ "${args[i]}" == "--rl" && $((i+1)) -lt ${#args[@]} ]]; then
                        rl_mode="${args[i+1]}"
                        break
                    fi
                done
                [[ -n "$rl_mode" ]] && log_info "RL training mode: $rl_mode"
            fi
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
    log_info "Cleaning outputs..."
    
    local dirs_to_clean=("$OUTPUT_DIR" "$DATA_DIR" "$LOG_DIR" "$TARGETS_DIR" "$VENV_DIR")
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
    check_os_support

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
            ensure_uv_installed
            ;;
        --rl)
            log_error "--rl is not a standalone command. Use: $0 --train --rl <dpo|cpo|kto|orpo> [other options]"
            show_help
            exit 1
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
    
    setup_directories
    check_config_file
    check_dependencies
    
    local args_array
    mapfile -t args_array < <(parse_args "$@")
    
    if ! validate_required_args args_array "$command"; then
        log_warning "Continuing with missing required arguments (defaults will be used if available)..."
    fi
    
    if run_command "$command" "${args_array[@]}"; then
        show_summary "$command" "${args_array[@]}"
    else
        log_error "Command failed"
        exit 1
    fi
}

main "$@"
