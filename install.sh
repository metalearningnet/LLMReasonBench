#!/bin/bash

set -euo pipefail

if [[ -t 1 ]]; then
    readonly BOLD=$(tput bold 2>/dev/null || echo)
    readonly RED=$(tput setaf 1 2>/dev/null || echo)
    readonly GREEN=$(tput setaf 2 2>/dev/null || echo)
    readonly YELLOW=$(tput setaf 3 2>/dev/null || echo)
    readonly BLUE=$(tput setaf 4 2>/dev/null || echo)
    readonly RESET=$(tput sgr0 2>/dev/null || echo)
else
    readonly BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

log_info() {
    echo "${BLUE}${BOLD}[INFO]${RESET} $*"
}

log_success() {
    echo "${GREEN}${BOLD}[SUCCESS]${RESET} $*"
}

log_warning() {
    echo "${YELLOW}${BOLD}[WARNING]${RESET} $*" >&2
}

log_error() {
    echo "${RED}${BOLD}[ERROR]${RESET} $*" >&2
    exit 1
}

INSTALL_MD=false
MD_SOURCE_PATH=""
SKIP_PACKAGES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --md)
            INSTALL_MD=true
            SKIP_PACKAGES=true
            shift
            ;;
        --path)
            if [[ -n $2 ]]; then
                if [[ "$INSTALL_MD" == false ]]; then
                    log_error "--path argument must follow --md argument"
                fi
                MD_SOURCE_PATH="$2"
                shift 2
            else
                log_error "--path requires a value"
            fi
            ;;
        -*)
            log_warning "Unknown option: $1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

package_list=(
    vllm absl-py accelerate aiohappyeyeballs aiohttp aiosignal
    altair annotated-types anyio asttokens async-timeout
    attrs bitsandbytes blessed blinker cachetools
    certifi chanfig charset-normalizer click coloredlogs
    comm contourpy cycler danling datasets
    debugpy decorator dill distro docker-pycreds
    einops exceptiongroup executing filelock fonttools
    frozenlist fsspec gitdb GitPython gpustat
    grpcio h11 httpcore httpx huggingface-hub
    humanfriendly idna importlib_metadata importlib_resources ipykernel
    ipython jedi Jinja2 jiter joblib
    jsonlines jsonschema jsonschema-specifications jupyter_client jupyter_core
    kiwisolver lazy-imports Markdown markdown-it-py MarkupSafe
    matplotlib matplotlib-inline mdurl mpmath multidict
    multimolecule multiprocess narwhals nest-asyncio networkx
    nltk numpy openai optimum packaging
    pandas parso peft pexpect pillow
    platformdirs prompt_toolkit protobuf psutil ptyprocess
    pure_eval pyarrow pydantic pydantic_core pydeck
    Pygments pyparsing python-dateutil pytz PyYAML
    pyzmq referencing regex requests rich
    rpds-py safetensors scikit-learn scipy seaborn
    sentence-transformers sentencepiece sentry-sdk seqeval setproctitle
    six smmap sniffio stack-data streamlit
    StrEnum sympy tenacity tensorboard tensorboard-data-server
    threadpoolctl tokenizers toml torch torchaudio
    torchvision tornado tqdm traitlets transformers
    triton typing_extensions tzdata urllib3 wandb
    watchdog wcwidth Werkzeug xxhash yarl zipp
)

if [[ "$SKIP_PACKAGES" == false ]]; then
    log_info "Starting package installation..."
    
    if pip install "${package_list[@]}"; then
        log_success "All packages installed successfully"
    else
        log_error "Failed to install packages"
    fi
fi

if [[ "$INSTALL_MD" == true ]]; then
    TARGET_DIR="./targets/md"
    
    mkdir -p "$(dirname "$TARGET_DIR")"
    
    if [[ -d "$TARGET_DIR" ]]; then
        log_warning "Removing existing MD directory at $TARGET_DIR"
        rm -rf "$TARGET_DIR"
    fi
    
    if [[ -n "$MD_SOURCE_PATH" ]]; then
        if [[ -d "$MD_SOURCE_PATH" ]]; then
            log_info "Copying MD model from $MD_SOURCE_PATH to $TARGET_DIR"
            cp -r "$MD_SOURCE_PATH" "$TARGET_DIR"
        else
            log_error "Local MD path $MD_SOURCE_PATH does not exist or is not a directory"
        fi
    else
        log_info "Cloning MD model from GitHub to $TARGET_DIR"
        if git clone https://github.com/metalearningnet/md.git "$TARGET_DIR"; then
            log_success "MD model cloned successfully"
        else
            log_error "Failed to clone MD model from GitHub"
        fi
    fi
    
    if [[ -d "$TARGET_DIR" ]]; then
        log_success "MD model successfully installed to $TARGET_DIR"
    else
        log_error "Failed to install MD model"
    fi
fi
