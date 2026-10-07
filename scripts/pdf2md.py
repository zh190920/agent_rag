#!/usr/bin/env python3
"""PDF → Markdown 转换脚本（pdfplumber 提取文本，pypdf 兜底）。

用法::

    python scripts/pdf2md.py datas/                      # 转目录下所有 PDF
    python scripts/pdf2md.py datas/book1.pdf             # 转单个 PDF
    python scripts/pdf2md.py datas/ -o datas/md/         # 指定输出目录

输出格式：每页一个 `## Page N` 标题 + 文本内容，空页跳过。
适合 PLC 指令/编程手册这类**文字为主**的 PDF；图片/纯扫描件只能拿到有限文字。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def extract_text_pdfplumber(pdf_path: Path) -> list[str]:
    """用 pdfplumber 逐页提取文本（优先，对表格/布局支持更好）。"""
    import pdfplumber  # noqa: PLC0415

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            if i % 50 == 0 or i == total:
                print(f"  pdfplumber: {i}/{total} 页", file=sys.stderr)
    return pages


def extract_text_pypdf(pdf_path: Path) -> list[str]:
    """pypdf 兜底提取（轻量，不依赖 pdfminer）。"""
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(text)
        if i % 50 == 0 or i == total:
            print(f"  pypdf: {i}/{total} 页", file=sys.stderr)
    return pages


def convert_one(pdf_path: Path, output_dir: Path) -> Path:
    """转换一个 PDF → MD，返回输出路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    # 清理文件名中不适合做标题的特殊字符
    safe_stem = stem.replace(" ", "-").replace("《", "").replace("》", "")[:80]
    md_path = output_dir / f"{safe_stem}.md"

    if md_path.exists():
        print(f"  已存在，跳过：{md_path.name}")
        return md_path

    print(f"  开始转换：{pdf_path.name}")
    t0 = time.time()
    pages: list[str] = []
    try:
        pages = extract_text_pdfplumber(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  pdfplumber 失败({exc})，尝试 pypdf 兜底...", file=sys.stderr)
        try:
            pages = extract_text_pypdf(pdf_path)
        except Exception as exc2:  # noqa: BLE001
            print(f"  两种方法都失败: {exc2}", file=sys.stderr)
            raise

    elapsed = time.time() - t0
    # 组装 markdown
    lines: list[str] = []
    lines.append(f"# {pdf_path.stem}\n\n")
    non_empty = 0
    for page_num, text in enumerate(pages, 1):
        text = text.strip()
        if not text:
            continue
        non_empty += 1
        lines.append(f"\n## 第 {page_num} 页\n\n")
        lines.append(text)
        lines.append("\n\n")

    md_content = "".join(lines)
    md_path.write_text(md_content, encoding="utf-8")

    total_pages = len(pages)
    total_chars = len(md_content)
    print(
        f"  ✅ 完成：{md_path.name}  "
        f"({total_pages} 页，{non_empty} 有效，{total_chars:,} 字符，{elapsed:.1f}s)"
    )
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF → Markdown 转换器")
    parser.add_argument("path", help="PDF 文件或包含 PDF 的目录")
    parser.add_argument(
        "-o", "--output", default=None,
        help="输出目录（默认 <path>/md 或同目录下 md/）",
    )
    args = parser.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"路径不存在：{src}", file=sys.stderr)
        return 2

    # 确定输出目录
    if args.output:
        out_dir = Path(args.output)
    elif src.is_file():
        out_dir = src.parent / "md"
    else:
        out_dir = src / "md"

    # 收集 PDF
    if src.is_file() and src.suffix.lower() == ".pdf":
        pdfs = [src]
    elif src.is_dir():
        pdfs = sorted(src.glob("*.pdf"))
    else:
        print(f"非 PDF 文件：{src}", file=sys.stderr)
        return 2

    if not pdfs:
        print("未找到任何 PDF 文件。", file=sys.stderr)
        return 1

    print(f"发现 {len(pdfs)} 个 PDF，输出目录：{out_dir}\n")
    for pdf in pdfs:
        try:
            convert_one(pdf, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ 失败：{pdf.name} -> {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
