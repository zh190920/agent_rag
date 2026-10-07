#!/usr/bin/env bash
# =============================================================================
#  一键跑通 PDF 知识库：转换 → 入库 → 问答（支持真实 LLM）
#
#  用法：
#    bash scripts/run_kb.sh "你的问题"
#    bash scripts/run_kb.sh --config config.yaml "MOV 指令格式是什么"
#    bash scripts/run_kb.sh --reindex "H5U 定时器怎么用"
#
#  默认使用项目根目录下的 config.yaml（需包含真实 LLM 配置）。
#  若不存在 config.yaml 或其中无 llm 配置，自动退化为离线 EchoLLM。
#  首次运行会转换 PDF + 建索引；之后只需传入 --reindex 才会重建。
# =============================================================================
set -euo pipefail

# ---- 路径与变量 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/d/softword/anaconda/python.exe"
DATAS_DIR="$PROJECT_DIR/datas"
MD_DIR="$DATAS_DIR/md"
RAG_DATA="$DATAS_DIR/.rag-data"
KB_NAME="general"

export PYTHONPATH="$PROJECT_DIR"
export PYTHONIOENCODING=utf-8

CONFIG="$PROJECT_DIR/config.yaml"
REINDEX=0
QUESTION=""

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2"; shift 2 ;;
        --reindex)  REINDEX=1; shift ;;
        -h|--help)
            echo "用法: bash scripts/run_kb.sh [--config path] [--reindex] \"你的问题\""
            exit 0 ;;
        *)
            QUESTION="$1"; shift ;;
    esac
done

if [[ -z "$QUESTION" ]]; then
    QUESTION="H5U 梯形图编程的基本方法和扫描周期是什么"
    echo "[INFO] 未提供问题，使用默认示例：$QUESTION"
fi

echo "============================================================"
echo " FusionRAG 知识库一键流程"
echo "   项目：$PROJECT_DIR"
echo "   配置：$CONFIG"
echo "   数据：$RAG_DATA"
echo "============================================================"
echo ""

# ---- Step 1: PDF → Markdown ----
NEED_CONVERT=1
if [[ -d "$MD_DIR" ]] && [[ $(find "$MD_DIR" -name "*.md" 2>/dev/null | wc -l) -gt 0 ]] && [[ $REINDEX -eq 0 ]]; then
    NEED_CONVERT=0
    echo "[Step 1/3] PDF → MD: 已存在转换结果，跳过。"
fi

if [[ $NEED_CONVERT -eq 1 ]]; then
    echo "[Step 1/3] 转换 PDF → Markdown ..."
    "$PYTHON" "$SCRIPT_DIR/pdf2md.py" "$DATAS_DIR"
    echo ""
fi

# ---- Step 2: 索引入库 ----
NEED_INDEX=$REINDEX
if [[ $REINDEX -eq 0 ]]; then
    # 检查是否已有索引
    if [[ ! -f "$RAG_DATA/state/fusion_rag.db" ]] || [[ ! -d "$MD_DIR" ]]; then
        NEED_INDEX=1
    fi
fi

if [[ $NEED_INDEX -eq 1 ]]; then
    echo "[Step 2/3] 索引 MD → 知识库 (BM25 + HashEmbedding) ..."
    "$PYTHON" -m fusion_rag.cli index-dir "$MD_DIR" \
        --kb "$KB_NAME" \
        --data-dir "$RAG_DATA" \
        --config "$CONFIG" \
        --globs "**/*.md" \
        --force
    echo ""
else
    echo "[Step 2/3] 索引已存在，跳过（加 --reindex 强制重建）。"
    echo ""
fi

# ---- Step 3: 问答 ----
echo "[Step 3/3] 提问：$QUESTION"
echo "------------------------------------------------------------"
"$PYTHON" -m fusion_rag.cli ask "$QUESTION" \
    --kb "$KB_NAME" \
    --data-dir "$RAG_DATA" \
    --config "$CONFIG" \
    --json
echo "------------------------------------------------------------"
echo ""
echo "✅ 完成！"
