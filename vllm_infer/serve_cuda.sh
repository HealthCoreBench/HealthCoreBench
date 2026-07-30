#!/usr/bin/env bash
# =============================================================================
#  serve_cuda.sh  —  本地 vLLM 模型启动器 (medcrux/models)  [可指定显卡版]
# -----------------------------------------------------------------------------
#  机器: 4 x NVIDIA A800-SXM4-80GB  |  vLLM 0.11.2  |  端口固定 8077
#  用法: ./serve_cuda.sh help              查看完整说明和模型清单
#        ./serve_cuda.sh model <名称>      按精炼名称启动一个模型
#
#  与 serve.sh 的唯一区别:
#    * 若在命令前显式设置了 CUDA_VISIBLE_DEVICES, 则「优先使用你指定的卡」,
#      不再强制从 0 号卡开始; 未显式指定 -t 时 TP 会自动对齐到可见卡数。
#      例:  CUDA_VISIBLE_DEVICES=1 ./serve_cuda.sh model phi-4   # 只用 1 号卡
#           CUDA_VISIBLE_DEVICES=2,3 ./serve_cuda.sh model qwen3-32b
#    * 未设置 CUDA_VISIBLE_DEVICES 时, 行为与 serve.sh 完全一致(从 0 号卡起)。
#    * 设置环境：SERVE_CONDA_ENV=qwen35
#
#  说明: 每个模型的“默认 TP(张卡数)”与“max-model-len”是依据权重体积、模型架构
#        和 vLLM 显存行为推算出的稳妥值(已保证能整除注意力头数)，不是逐个实测。
#        真正的落地验证发生在你运行本脚本时——脚本会打印完整命令与依据，可用
#        -t / -l / -u 等参数随时覆盖。
# =============================================================================

set -euo pipefail

# ------------------------------- 基本常量 -----------------------------------
MODELS_DIR="/F00120250029/lixiang_share/wangrongsheng_share/medcrux/models"
DEFAULT_PORT=8077
DEFAULT_HOST="0.0.0.0"
DEFAULT_UTIL="0.90"
TOTAL_GPUS=4

# conda: 启动模型前自动激活的环境 (可用环境变量覆盖)
#   SERVE_CONDA_ENV=xxx   换用别的环境
#   SERVE_NO_ACTIVATE=1   完全跳过自动激活 (用当前 shell 的 vllm)
CONDA_ROOT="/F00120250029/lixiang_share/Data/conda"
CONDA_ENV="${SERVE_CONDA_ENV:-health-eval}"

# 颜色 (仅在 tty 下启用)
if [[ -t 1 ]]; then
  C_RST=$'\033[0m'; C_B=$'\033[1m'; C_DIM=$'\033[2m'
  C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_BL=$'\033[34m'; C_CY=$'\033[36m'
else
  C_RST=''; C_B=''; C_DIM=''; C_R=''; C_G=''; C_Y=''; C_BL=''; C_CY=''
fi

log()   { echo "${C_CY}[serve]${C_RST} $*"; }
ok()    { echo "${C_G}[ ok ]${C_RST} $*"; }
warn()  { echo "${C_Y}[warn]${C_RST} $*" >&2; }
err()   { echo "${C_R}[fail]${C_RST} $*" >&2; }
hr()    { echo "${C_DIM}--------------------------------------------------------------------------------${C_RST}"; }

# ---------------------------- 自动激活 conda 环境 ---------------------------
# 若当前已在目标环境则跳过; 否则 source conda 并 activate。
# 设 SERVE_NO_ACTIVATE=1 可完全跳过。
ensure_env() {
  if [[ "${SERVE_NO_ACTIVATE:-0}" == "1" ]]; then
    log "SERVE_NO_ACTIVATE=1, 跳过自动激活, 使用当前 shell 的 vllm"
    return 0
  fi
  # 已经在目标环境?
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return 0
  fi
  local hook="${CONDA_ROOT}/etc/profile.d/conda.sh"
  if [[ ! -f "$hook" ]]; then
    warn "找不到 conda 初始化脚本: $hook"
    warn "将尝试使用当前 shell 的 vllm (若已手动激活可忽略)"
    return 0
  fi
  # conda 的 source/activate 脚本在 set -eu 下可能误触发退出, 临时关闭
  set +eu
  # shellcheck disable=SC1090
  source "$hook"
  conda activate "$CONDA_ENV" 2>/dev/null
  local rc=$?
  set -eu
  if [[ $rc -eq 0 && "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    log "已激活 conda 环境: ${C_B}${CONDA_ENV}${C_RST}"
  else
    warn "无法激活环境 '${CONDA_ENV}' (不存在?)。将使用当前 shell 的 vllm。"
    warn "可设 SERVE_CONDA_ENV=<环境名> 指定, 或 SERVE_NO_ACTIVATE=1 跳过。"
  fi
  return 0
}

# ---------------------------- 定位 vllm 可执行 ------------------------------
find_vllm() {
  if command -v vllm >/dev/null 2>&1; then
    command -v vllm
  elif [[ -x "/F00120250029/lixiang_share/Data/conda/bin/vllm" ]]; then
    echo "/F00120250029/lixiang_share/Data/conda/bin/vllm"
  else
    echo ""
  fi
}

# =============================================================================
#  模型注册表
#  resolve_model <alias> 会填充以下全局变量:
#    R_DIR      模型目录名 (相对 MODELS_DIR)
#    R_TP       默认张量并行卡数 (默认启动卡数)
#    R_MINTP    理论最小卡数 (参考)
#    R_MAXLEN   默认 max-model-len
#    R_TYPE     文本 / 多模态 / MoE 等标签
#    R_CTX      原生最大上下文
#    R_SIZE     权重体积(粗略)
#    R_NOTE     备注
#    R_EXTRA    该模型额外的 vllm 参数 (字符串, 会被 eval 拆分)
#  返回 0 表示命中, 1 表示未知别名
# =============================================================================
resolve_model() {
  local a="$1"
  R_DIR=""; R_TP=""; R_MINTP=""; R_MAXLEN=""; R_TYPE=""; R_CTX=""; R_SIZE=""; R_NOTE=""; R_EXTRA=""; R_GROUP=""; R_CEIL=""
  # R_MAXLEN = 默认启动上下文 (取 config 原生可直接支持、且默认TP能稳定启动的值)
  # R_CEIL   = 上限说明 (要达到需要的条件: YaRN / 更多卡 等)
  case "$a" in
    # ============================ 文本 ============================
    qwen3-32b|qwen3)
      R_DIR="Qwen3-32B";                    R_TP=2; R_MINTP=1; R_MAXLEN=40960; R_TYPE="文本";     R_CTX="40960";   R_SIZE="65.5G"; R_GROUP="text|1"
      R_CEIL="40960 原生 / 131072 需YaRN"; R_NOTE="Dense 32B; 32K约1卡(紧), 默认2卡跑满40K原生" ;;
    gptoss-20b|gpt-oss-20b|oss-20b)
      R_DIR="gpt-oss-20b";                  R_TP=1; R_MINTP=1; R_MAXLEN=131072; R_TYPE="MoE(mxfp4)"; R_CTX="131072"; R_SIZE="13.8G mxfp4"; R_GROUP="text|1"
      R_CEIL="131072 (config内置YaRN)"; R_NOTE="MoE 21B/3.6B; 单卡足够, 可跑满131K" ;;
    phi4|phi-4)
      R_DIR="phi-4";                        R_TP=1; R_MINTP=1; R_MAXLEN=16384; R_TYPE="文本(14B)"; R_CTX="16384";   R_SIZE="29.3G"; R_GROUP="text|1"
      R_CEIL="16384 (原生上限)"; R_NOTE="Dense 14B; 单卡" ;;
    hunyuan-a13b|hunyuan)
      R_DIR="Hunyuan-A13B-Instruct";        R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="MoE/文本"; R_CTX="32768";   R_SIZE="161G"; R_GROUP="text|2"
      R_CEIL="config原生32768 / 官方262144需rope覆盖(256K建议8卡)"; R_NOTE="MoE 80B/13B; ≤128K用4卡" ;;
    qwen25-72b|qwen2.5-72b)
      R_DIR="Qwen2.5-72B-Instruct";         R_TP=4; R_MINTP=2; R_MAXLEN=32768; R_TYPE="文本";     R_CTX="32768";   R_SIZE="145G"; R_GROUP="text|2"
      R_CEIL="32768 原生 / 131072 需YaRN"; R_NOTE="Dense 72.7B; 最小2卡可起, 默认4卡更稳" ;;
    llama33-70b|llama3.3-70b|llama-3.3-70b)
      R_DIR="Llama-3.3-70B-Instruct";       R_TP=4; R_MINTP=2; R_MAXLEN=131072; R_TYPE="文本";    R_CTX="131072";  R_SIZE="141G"; R_GROUP="text|2"
      R_CEIL="131072 (config内置, 可直接跑满)"; R_NOTE="Dense 70B; 4卡跑满128K(并发约3路)" ;;
    deepseek-v4-flash|dsv4|deepseek-v4)
      R_DIR="DeepSeek-V4-Flash";            R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="MoE/文本"; R_CTX="1048576"; R_SIZE="160G FP4+FP8"; R_GROUP="text|3"
      R_CEIL="官方1M"; R_NOTE="!! MoE 284B/13B FP4混合; A800无FP4且vLLM0.11.2支持存疑, 当前不建议部署" ;;
    gptoss-120b|gpt-oss-120b|oss-120b)
      R_DIR="gpt-oss-120b";                 R_TP=2; R_MINTP=1; R_MAXLEN=131072; R_TYPE="MoE(mxfp4)"; R_CTX="131072"; R_SIZE="61G mxfp4"; R_GROUP="text|3"
      R_CEIL="131072 (config内置YaRN)"; R_NOTE="MoE 117B/5.1B; 1卡可起2卡更稳; 目录含metal/original备用格式, vLLM只读HF分片" ;;
    mistral-medium-128b|mistral-medium|mistral3)
      R_DIR="Mistral-Medium-3.5-128B";      R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="多模态*";  R_CTX="262144";  R_SIZE="134G fp8"; R_GROUP="text|3"
      R_CEIL="262144 原生 (官方建议8卡TP=8)"; R_NOTE="Dense FP8, 实际带视觉(*); 官方要8卡, 4卡仅建议短上下文" ;;
    # =========================== 多模态 ===========================
    gemma4-31b|gemma-4-31b|gemma4)
      R_DIR="gemma-4-31B-it";               R_TP=2; R_MINTP=1; R_MAXLEN=131072; R_TYPE="多模态";  R_CTX="262144";  R_SIZE="62.6G"; R_GROUP="mm|1"
      R_CEIL="262144 原生 (滑窗注意力, 2卡可达256K)"; R_NOTE="Dense; ≤64K约1卡, 默认2卡" ;;
    qwen36-27b|qwen3.6-27b)
      R_DIR="Qwen3.6-27B";                  R_TP=2; R_MINTP=1; R_MAXLEN=131072; R_TYPE="多模态";  R_CTX="262144";  R_SIZE="55.6G"; R_GROUP="mm|1"
      R_CEIL="262144 原生 / 1.01M 需YaRN(官方262K配方需8卡)"; R_NOTE="Dense; 32~64K约1卡, 默认2卡" ;;
    internvl35-38b|internvl3.5-38b|ivl35-38b)
      R_DIR="InternVL3_5-38B-Instruct";     R_TP=2; R_MINTP=2; R_MAXLEN=32768; R_TYPE="多模态";   R_CTX="40960";   R_SIZE="76.8G"; R_GROUP="mm|1"
      R_CEIL="配置上限40960, 官方建议32768"; R_NOTE="38.4B; 2卡" ;;
    qwen25vl-72b|qwen2.5-vl-72b|qwenvl-72b)
      R_DIR="Qwen2.5-VL-72B-Instruct";      R_TP=4; R_MINTP=2; R_MAXLEN=32768; R_TYPE="多模态";   R_CTX="128000";  R_SIZE="147G"; R_GROUP="mm|2"
      R_CEIL="32768默认 / 128000原生(mrope) / 131072需YaRN"; R_NOTE="VL; 图像tokens占上下文, 默认32K; 4卡" ;;
    llama32-90b|llama3.2-90b|llama-3.2-90b)
      R_DIR="Llama-3.2-90B-Vision-Instruct";R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="多模态(mllama)"; R_CTX="131072"; R_SIZE="180G"; R_GROUP="mm|2"
      R_CEIL="131072 (config内置)"; R_NOTE="视觉交叉注意力, 并发不宜过高; 4卡"; R_EXTRA="--max-num-seqs 16" ;;
    internvl3-78b|internvl3|ivl3-78b)
      R_DIR="InternVL3-78B-Instruct";       R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="多模态";   R_CTX="32768";   R_SIZE="157G"; R_GROUP="mm|2"
      R_CEIL="32768 (config原生上限)"; R_NOTE="4卡" ;;
    glm45v|glm-4.5v|glm45-v)
      R_DIR="GLM-4.5V";                     R_TP=4; R_MINTP=4; R_MAXLEN=65536; R_TYPE="多模态MoE(108B)"; R_CTX="65536"; R_SIZE="215G"; R_GROUP="mm|3"
      R_CEIL="65536 (config原生, 可直接跑满)"; R_NOTE="MoE 108B; 官方TP=4" ;;
    qwen35-122b|qwen3.5-122b|qwen35-122b-a10b)
      R_DIR="Qwen3.5-122B-A10B";            R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="MoE/多模态"; R_CTX="262144"; R_SIZE="250G"; R_GROUP="mm|3"
      R_CEIL="262144 原生(官方TP=8) / 1.01M 需YaRN"; R_NOTE="MoE 122B/10B; 4卡权重可载, 长上下文/吞吐建议8卡" ;;
    llama4-scout|llama4|scout)
      R_DIR="Llama-4-Scout-17B-16E-Instruct";R_TP=4; R_MINTP=4; R_MAXLEN=32768; R_TYPE="多模态MoE(109B)"; R_CTX="10485760"; R_SIZE="218G"; R_GROUP="mm|3"
      R_CEIL="架构10M / 4卡约32K, 64K建议8卡"; R_NOTE="MoE 109B/17B; 4卡只能跑短上下文" ;;
    *)
      return 1 ;;
  esac
  return 0
}

# 所有主别名, 按 [模态 -> 筛选层级] 有序排列, 供 help / 校验遍历
ALL_ALIASES=(
  # 文本
  qwen3-32b gptoss-20b phi4
  hunyuan-a13b qwen25-72b llama33-70b
  deepseek-v4-flash gptoss-120b mistral-medium-128b
  # 多模态
  gemma4-31b qwen36-27b internvl35-38b
  qwen25vl-72b llama32-90b internvl3-78b
  glm45v qwen35-122b llama4-scout
)

# =============================================================================
#  help
# =============================================================================
print_help() {
  cat <<EOF
${C_B}serve_cuda.sh${C_RST} — 本地 vLLM 模型启动器  (medcrux/models) [可指定显卡版]

${C_B}环境${C_RST}
  显卡      : ${TOTAL_GPUS} x NVIDIA A800-SXM4-80GB  (共 320GB)
  框架      : vLLM 0.11.2  (torch 2.9 / CUDA 12.8)
  端口      : 固定 ${DEFAULT_PORT}  (OpenAI 兼容 API: http://<host>:${DEFAULT_PORT}/v1)
  模型目录  : ${MODELS_DIR}
  conda环境 : 启动前自动激活 '${CONDA_ENV}' (无需手动 activate)
              SERVE_CONDA_ENV=<名> 换环境; SERVE_NO_ACTIVATE=1 跳过

${C_B}用法${C_RST}
  ./serve_cuda.sh help                       打印本说明和模型清单
  ./serve_cuda.sh list                       只打印模型清单表格
  ./serve_cuda.sh model <名称> [选项...]     启动指定模型 (每次只启动一个)

${C_B}指定显卡 (本脚本特性)${C_RST}
  在命令前设置 CUDA_VISIBLE_DEVICES 即可锁定使用哪几张卡, 优先级最高:
    CUDA_VISIBLE_DEVICES=1   ./serve_cuda.sh model phi-4        # 只用 1 号卡
    CUDA_VISIBLE_DEVICES=2,3 ./serve_cuda.sh model qwen3-32b    # 用 2,3 号卡(TP自动=2)
  未设置时行为与 serve.sh 一致 (从 0 号卡按 TP 顺序取)。
  显式 -t 与可见卡数不一致时会告警 (vLLM 要求 TP == 可见卡数)。

${C_B}选项 (可覆盖默认值)${C_RST}
  -t, --tp <N>        张量并行卡数 (覆盖默认 TP)
  -l, --max-len <N>   max-model-len 上下文长度上限
  -u, --util <F>      GPU 显存利用率 0~1 (默认 ${DEFAULT_UTIL})
  -p, --port <N>      端口 (默认 ${DEFAULT_PORT})
      --host <IP>     监听地址 (默认 ${DEFAULT_HOST})
      --name <str>    对外暴露的 served-model-name (默认=精炼名称)
      --dry-run       只打印将执行的命令, 不真正启动
      --extra "<...>" 追加任意原生 vllm serve 参数
  -h, --help          同 help

${C_B}示例${C_RST}
  ./serve_cuda.sh model qwen25-72b                     # 用默认4卡(0-3)启动 Qwen2.5-72B
  CUDA_VISIBLE_DEVICES=1 ./serve_cuda.sh model phi-4    # 只用 1 号卡, 不碰 GPU0
  ./serve_cuda.sh model qwen3-32b -t 1                 # 强制单卡(0号)启动
  ./serve_cuda.sh model llama33-70b -l 16384 -t 2      # 2卡 + 限制上下文16k
  ./serve_cuda.sh model gptoss-120b --dry-run          # 只看命令不启动

${C_B}说明${C_RST}
  * "默认TP" 是按权重体积+架构推算的稳妥卡数, 保证能整除注意力头数;
    "最小TP" 仅为参考。你每次只跑一个模型, 4卡都空闲, 故默认偏向多用卡换稳定。
  * 所有默认 max-model-len 压到 32768 以保证启动 (原生更短的按原生),
    需要更长上下文时用 -l 调大, 但可能因 KV cache 过大而 OOM。
  * mxfp4/fp8 量化模型体积已是量化后大小。

EOF
  print_table
}

# 模型清单表格 (按 模态 -> 筛选层级 分组)
_table_header() {
  printf "${C_B}%-20s %-32s %-12s %-3s %-3s %-9s %-16s${C_RST}\n" "精炼名称" "目录" "权重" "TP" "最" "默认len" "类型"
}
_table_row() {
  printf "%-20s %-32s %-12s %-3s %-3s %-9s %-16s\n" "$1" "$R_DIR" "$R_SIZE" "$R_TP" "$R_MINTP" "$R_MAXLEN" "$R_TYPE"
}

# 打印某个分组 (模态|层级) 下的所有模型
_print_group() {
  local want="$1" a
  for a in "${ALL_ALIASES[@]}"; do
    resolve_model "$a"
    if [[ "$R_GROUP" == "$want" ]]; then
      _table_row "$a"
    fi
  done
  return 0
}

print_table() {
  hr
  echo "${C_B}${C_BL}【文本模型】${C_RST}"
  echo "${C_DIM}  第一层筛选 (小: 单卡~双卡可跑)${C_RST}"
  _table_header; _print_group "text|1"
  echo "${C_DIM}  第二层筛选 (中: 70B 级)${C_RST}"
  _print_group "text|2"
  echo "${C_DIM}  第三层筛选 (大: 120B+ / MoE)${C_RST}"
  _print_group "text|3"
  hr
  echo "${C_B}${C_BL}【多模态模型】${C_RST}"
  echo "${C_DIM}  第一层筛选 (小: 27B~38B)${C_RST}"
  _table_header; _print_group "mm|1"
  echo "${C_DIM}  第二层筛选 (中: 72B~90B)${C_RST}"
  _print_group "mm|2"
  echo "${C_DIM}  第三层筛选 (大: 108B+ / MoE)${C_RST}"
  _print_group "mm|3"
  hr
  echo "${C_DIM}名称支持多种别名, 例: qwen25-72b / qwen2.5-72b 等价。完整别名见脚本 resolve_model()。${C_RST}"
  echo "${C_DIM}类型带 * 号者 (mistral-medium-128b) 实际具备视觉能力, 按你的分层归入文本。${C_RST}"
}

# =============================================================================
#  GPU 状态打印
# =============================================================================
print_gpu_status() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "当前 GPU 状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader | while IFS=, read -r idx name used total util; do
      echo "    GPU${idx}:${name} 已用${used} / ${total}  利用率${util}"
    done
  else
    warn "未找到 nvidia-smi, 跳过 GPU 状态检查"
  fi
}

# 检查是否有显卡已被占用 (>2GB 视为占用)
check_gpu_free() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '$1>2048{c++} END{print c+0}')
  if [[ "${busy:-0}" -gt 0 ]]; then
    warn "检测到 ${busy} 张卡显存占用 >2GB。你说过每次只跑一个模型——"
    warn "如果上一个模型还在运行, 请先停掉它 (端口 ${PORT} 也会冲突)。"
  fi
}

# =============================================================================
#  启动逻辑
# =============================================================================
launch() {
  local alias_name="$1"; shift

  # 捕获用户是否在命令前显式指定了 CUDA_VISIBLE_DEVICES (本脚本核心差异)
  local USER_CVD="${CUDA_VISIBLE_DEVICES:-}"
  local NDEV=0

  # 默认值
  local TP="" MAXLEN="" UTIL="$DEFAULT_UTIL" PORT="$DEFAULT_PORT" HOST="$DEFAULT_HOST"
  local SERVED_NAME="" DRY=0 USER_EXTRA=""

  # 解析选项
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--tp)       TP="$2"; shift 2 ;;
      -l|--max-len)  MAXLEN="$2"; shift 2 ;;
      -u|--util)     UTIL="$2"; shift 2 ;;
      -p|--port)     PORT="$2"; shift 2 ;;
      --host)        HOST="$2"; shift 2 ;;
      --name)        SERVED_NAME="$2"; shift 2 ;;
      --dry-run)     DRY=1; shift ;;
      --extra)       USER_EXTRA="$2"; shift 2 ;;
      *) err "未知选项: $1"; echo "用 ./serve_cuda.sh help 查看用法"; exit 2 ;;
    esac
  done
  PORT="${PORT:-$DEFAULT_PORT}"

  # 查表
  if ! resolve_model "$alias_name"; then
    err "未知模型别名: '${alias_name}'"
    echo "可用模型:"
    print_table
    exit 2
  fi

  # ---- 若用户指定了 CUDA_VISIBLE_DEVICES, 统计可见卡数; 未显式 -t 时自动对齐 ----
  if [[ -n "$USER_CVD" ]]; then
    NDEV=$(awk -F, '{c=0; for(i=1;i<=NF;i++) if($i!="") c++; print c}' <<<"$USER_CVD")
    if [[ "$NDEV" -lt 1 ]]; then
      err "CUDA_VISIBLE_DEVICES='${USER_CVD}' 解析不到有效卡号。"; exit 2
    fi
    if [[ -z "$TP" ]]; then
      TP="$NDEV"   # 未显式 -t 时, TP 自动对齐到可见卡数
    fi
  fi

  # 应用默认/覆盖
  TP="${TP:-$R_TP}"
  MAXLEN="${MAXLEN:-$R_MAXLEN}"
  SERVED_NAME="${SERVED_NAME:-$alias_name}"

  local MODEL_PATH="${MODELS_DIR}/${R_DIR}"

  # ---- 自动激活 conda 环境 (dry-run 也激活, 以便显示正确的 vllm 路径) ----
  ensure_env

  # ---- 前置校验 ----
  local vllm_bin; vllm_bin="$(find_vllm)"
  [[ -n "$vllm_bin" ]] || { err "找不到 vllm 可执行文件。请手动 activate 环境, 或检查 CONDA_ROOT/CONDA_ENV。"; exit 1; }
  [[ -d "$MODEL_PATH" ]] || { err "模型目录不存在: $MODEL_PATH"; exit 1; }
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    err "缺少 config.json: ${MODEL_PATH}/config.json"; exit 1
  fi
  if ! [[ "$TP" =~ ^[0-9]+$ ]] || (( TP < 1 || TP > TOTAL_GPUS )); then
    err "TP 非法: '$TP' (应为 1~${TOTAL_GPUS})"; exit 2
  fi

  # ---- 选卡: 用户指定 CUDA_VISIBLE_DEVICES 时优先使用它, 否则从 0 号卡按 TP 顺序取 ----
  local devices
  if [[ -n "$USER_CVD" ]]; then
    devices="$USER_CVD"
    if [[ "$NDEV" -ne "$TP" ]]; then
      warn "CUDA_VISIBLE_DEVICES=${USER_CVD} 提供 ${NDEV} 张卡, 但 TP=${TP} 不一致。"
      warn "vLLM 要求 tensor-parallel-size == 可见卡数, 否则会启动失败。"
      warn "请用 -t ${NDEV} 或调整 CUDA_VISIBLE_DEVICES 使二者相等。"
    fi
  else
    devices=$(seq -s, 0 $((TP-1)))
  fi

  # ---- 打印启动信息 ----
  echo
  hr
  echo "${C_B}准备启动模型${C_RST}"
  hr
  printf "  %-16s %s\n" "精炼名称"      "$alias_name"
  printf "  %-16s %s\n" "目录"          "$R_DIR"
  printf "  %-16s %s\n" "完整路径"      "$MODEL_PATH"
  printf "  %-16s %s\n" "类型"          "$R_TYPE"
  printf "  %-16s %s\n" "权重体积"      "$R_SIZE"
  printf "  %-16s %s\n" "原生上下文"    "$R_CTX"
  printf "  %-16s %s%s\n" "张卡数(TP)"  "$TP" "  ${C_DIM}(默认${R_TP} / 最小${R_MINTP})${C_RST}"
  if [[ -n "$USER_CVD" ]]; then
    printf "  %-16s %s%s\n" "使用GPU"    "$devices" "  ${C_DIM}(来自 CUDA_VISIBLE_DEVICES, 已优先采用)${C_RST}"
  else
    printf "  %-16s %s\n" "使用GPU"       "$devices"
  fi
  printf "  %-16s %s%s\n" "max-model-len" "$MAXLEN" "  ${C_DIM}(默认${R_MAXLEN})${C_RST}"
  [[ -n "$R_CEIL" ]]     && printf "  %-16s ${C_DIM}%s${C_RST}\n" "上下文上限" "$R_CEIL"
  printf "  %-16s %s\n" "显存利用率"    "$UTIL"
  printf "  %-16s %s\n" "端口"          "$PORT"
  printf "  %-16s %s\n" "监听地址"      "$HOST"
  printf "  %-16s %s\n" "对外模型名"    "$SERVED_NAME"
  [[ -n "$R_NOTE" ]]     && printf "  %-16s ${C_Y}%s${C_RST}\n" "备注" "$R_NOTE"
  [[ -n "$R_EXTRA" ]]    && printf "  %-16s %s\n" "内置额外参数" "$R_EXTRA"
  [[ -n "$USER_EXTRA" ]] && printf "  %-16s %s\n" "用户额外参数" "$USER_EXTRA"
  printf "  %-16s %s\n" "vllm 路径"     "$vllm_bin"
  hr

  print_gpu_status
  check_gpu_free

  # 对已知在本机难以部署的模型给出显式警告 + 需二次确认
  if [[ "$R_DIR" == "DeepSeek-V4-Flash" ]]; then
    echo
    warn "DeepSeek-V4-Flash 是 FP4+FP8 混合量化 MoE(284B)。A800 无原生 FP4 支持,"
    warn "且 vLLM 0.11.2 对该架构支持存疑, 极可能加载失败或性能极差。"
    if [[ "$DRY" -ne 1 ]]; then
      read -r -p "$(echo "${C_Y}仍要尝试启动吗? [y/N] ${C_RST}")" ans
      [[ "$ans" =~ ^[Yy]$ ]] || { err "已取消。"; exit 1; }
    fi
  fi

  # ---- 组装命令 ----
  local -a CMD=(
    "$vllm_bin" serve "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAXLEN"
    --gpu-memory-utilization "$UTIL"
    --host "$HOST"
    --port "$PORT"
    --trust-remote-code
  )
  # 追加内置额外参数
  if [[ -n "$R_EXTRA" ]]; then
    # shellcheck disable=SC2206
    local -a e=($R_EXTRA); CMD+=("${e[@]}")
  fi
  # 追加用户额外参数
  if [[ -n "$USER_EXTRA" ]]; then
    # shellcheck disable=SC2206
    local -a u=($USER_EXTRA); CMD+=("${u[@]}")
  fi

  echo
  log "即将执行的命令:"
  echo "    CUDA_VISIBLE_DEVICES=${devices} \\"
  printf '    %q ' "${CMD[@]}"; echo
  hr

  if [[ "$DRY" -eq 1 ]]; then
    ok "dry-run 模式, 未真正启动。"
    echo "启动成功后可这样测试:"
    echo "    curl http://${HOST}:${PORT}/v1/models"
    exit 0
  fi

  echo
  ok "开始启动 (Ctrl-C 停止)。首次加载大模型可能需要数分钟..."
  echo "启动完成后, OpenAI 兼容接口: http://${HOST}:${PORT}/v1  (model=${SERVED_NAME})"
  hr
  export CUDA_VISIBLE_DEVICES="$devices"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  exec "${CMD[@]}"
}

# =============================================================================
#  入口
# =============================================================================
main() {
  local cmd="${1:-help}"
  case "$cmd" in
    help|-h|--help|hlep|"") print_help ;;
    list|ls)                print_table ;;
    model|run|serve)
      shift
      [[ $# -ge 1 ]] || { err "缺少模型名称。用法: ./serve_cuda.sh model <名称>"; echo; print_table; exit 2; }
      launch "$@" ;;
    *)
      err "未知命令: '$cmd'"
      echo "用 ./serve_cuda.sh help 查看用法。"
      exit 2 ;;
  esac
}

main "$@"
