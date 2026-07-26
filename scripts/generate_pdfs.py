"""将 data/ 目录下(含子目录)的 txt 知识文件批量转换为同名 PDF

中文字体使用 reportlab 内置的 CID 字体 STSong-Light,无需外部字体文件。
用法: python scripts/generate_pdfs.py [相对data的目录名,缺省为全部]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from utils.file_handler import listdir_with_allowed_type
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

STYLE_H1 = ParagraphStyle('h1', fontName='STSong-Light', fontSize=18, leading=26, spaceAfter=12)
STYLE_H2 = ParagraphStyle('h2', fontName='STSong-Light', fontSize=14, leading=20, spaceBefore=10, spaceAfter=6)
STYLE_H3 = ParagraphStyle('h3', fontName='STSong-Light', fontSize=12, leading=18, spaceBefore=8, spaceAfter=4)
STYLE_BODY = ParagraphStyle('body', fontName='STSong-Light', fontSize=10.5, leading=16, spaceAfter=3)


def txt_to_pdf(txt_path: str, pdf_path: str):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    story = []
    for line in lines:
        text = line.strip()
        if not text:
            story.append(Spacer(1, 0.15 * cm))
            continue
        if text.startswith('### '):
            story.append(Paragraph(text[4:], STYLE_H3))
        elif text.startswith('## '):
            story.append(Paragraph(text[3:], STYLE_H2))
        elif text.startswith('# '):
            story.append(Paragraph(text[2:], STYLE_H1))
        else:
            story.append(Paragraph(text, STYLE_BODY))

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=os.path.splitext(os.path.basename(txt_path))[0],
    )
    doc.build(story)


def main():
    data_dir = get_abs_path('data')
    target = sys.argv[1] if len(sys.argv) > 1 else ''
    if target:
        data_dir = os.path.join(data_dir, target)

    txt_files = listdir_with_allowed_type(data_dir, ('.txt',))
    if not txt_files:
        logger.warning(f'[生成PDF]{data_dir} 下未找到 txt 文件')
        return

    for txt_path in txt_files:
        pdf_path = os.path.splitext(txt_path)[0] + '.pdf'
        try:
            txt_to_pdf(txt_path, pdf_path)
            logger.info(f'[生成PDF]{txt_path} -> {pdf_path}')
        except Exception as e:
            logger.error(f'[生成PDF]{txt_path} 转换失败: {str(e)}', exc_info=True)


if __name__ == '__main__':
    main()
