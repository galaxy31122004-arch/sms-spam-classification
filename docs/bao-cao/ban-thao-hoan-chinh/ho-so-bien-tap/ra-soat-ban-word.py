"""Clean inherited metadata and audit the generated Word report.

The cover template carries the title and author of an earlier report, so those fields are reset
here. The audit then looks for leftover placeholders, tables wider than the text column, empty
paragraphs, missing figures and a table of contents that was never updated.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from docx.oxml.ns import qn
from docx import Document
from docx.shared import RGBColor

AUDIT = Path(__file__).resolve().parent
DEFAULT = AUDIT.parent / "drafts" / "bao-cao-sms-spam.docx"
TITLE = "Phát hiện tin nhắn SMS rác bằng mạng nơ-ron MLP, LSTM và Transformer"
TEXT_WIDTH_CM = 16.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT)
    parser.add_argument("--fix-metadata", action="store_true", help="ghi lai title/author cua tai lieu")
    parser.add_argument("--fix-styles", action="store_true",
                        help="ep Times New Roman mau den cho cac style Word tu sinh (TOC, Hyperlink)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    doc = Document(str(args.file))

    properties = doc.core_properties
    print(f"title cu   : {properties.title!r}")
    print(f"author cu  : {properties.author!r}")
    if args.fix_styles:
        # Word creates TOC 1-3 and Hyperlink when it builds the table of contents; they come with
        # theme fonts and blue underlined links, which do not belong in a printed report.
        names = {style.name for style in doc.styles}
        touched = []
        # Word names the generated entry styles "toc 1".."toc 3" in lower case.
        for name in ("toc 1", "toc 2", "toc 3", "TOC Heading1", "TOC Heading",
                     "Hyperlink", "FollowedHyperlink", "Caption"):
            if name not in names:
                continue
            style = doc.styles[name]
            rpr = style.element.get_or_add_rPr()
            rfonts = rpr.get_or_add_rFonts()
            for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
                rfonts.attrib.pop(qn(f"w:{attribute}"), None)
            for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
                rfonts.set(qn(f"w:{attribute}"), "Times New Roman")
            style.font.color.rgb = RGBColor(0, 0, 0)
            if name.startswith(("Hyperlink", "Followed")):
                style.font.underline = False
            touched.append(name)
        doc.save(str(args.file))
        print(f"-> da ep phong/mau cho style: {', '.join(touched) if touched else 'khong co style nao'}")

    if args.fix_metadata:
        properties.title = TITLE
        properties.subject = ""
        properties.keywords = ""
        properties.comments = ""
        properties.category = ""
        properties.author = ""
        properties.last_modified_by = ""
        doc.save(str(args.file))
        print(f"-> da dat title: {TITLE!r}; author de trong de nguoi dung tu dien")

    problems = []
    placeholder = re.compile(r"\[(LOẠI TÀI LIỆU|TÊN/CHỦ ĐỀ|PHỤ ĐỀ|HỌ VÀ TÊN|MÃ SỐ|LỚP|GIẢNG VIÊN"
                             r" HƯỚNG DẪN|ĐỊA DANH, NĂM|CẦN BỔ SUNG)\]")
    for index, paragraph in enumerate(doc.paragraphs):
        if placeholder.search(paragraph.text):
            problems.append(f"placeholder con lai o doan {index}: {paragraph.text.strip()[:70]}")
        if "```" in paragraph.text or paragraph.text.strip().startswith("|"):
            problems.append(f"markdown chua chuyen o doan {index}: {paragraph.text.strip()[:70]}")
        if re.search(r"\\(frac|times|approx|text|tag|lfloor|rfloor|ldots)\b", paragraph.text):
            problems.append(f"cong thuc con dang LaTeX o doan {index}: {paragraph.text.strip()[:70]}")
        # An asterisk is only suspicious in prose; in code it is multiplication.
        prose_stars = any("*" in run.text and run.font.name != "Consolas" for run in paragraph.runs)
        if prose_stars:
            problems.append(f"dau sao Markdown con lai o doan {index}: {paragraph.text.strip()[:70]}")

    for number, table in enumerate(doc.tables, 1):
        widths = [cell.width.cm if cell.width else None for cell in table.rows[0].cells]
        total = sum(width for width in widths if width) if any(widths) else None
        if total and total > TEXT_WIDTH_CM + 0.3:
            problems.append(f"bang {number} rong {total:.1f} cm > cot chu {TEXT_WIDTH_CM} cm")
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    # Word drops direct formatting that repeats the style, so check the style too.
                    spacing = (paragraph.paragraph_format.line_spacing
                               or doc.styles[paragraph.style.name].paragraph_format.line_spacing)
                    if spacing is None:
                        problems.append(f"bang {number}: o thieu gian dong (truc tiep lan theo style)")
                        break

    images = doc.inline_shapes
    body_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    toc_updated = "MỤC LỤC" in body_text and "CHƯƠNG 1" in body_text
    empty = sum(1 for paragraph in doc.paragraphs if not paragraph.text.strip()
                and not paragraph._element.findall(".//{*}drawing"))

    print(f"\ndoan van   : {len(doc.paragraphs)} (trong do {empty} doan trong)")
    print(f"bang       : {len(doc.tables)}")
    print(f"hinh       : {len(images)}")
    print(f"muc luc    : {'da sinh' if toc_updated else 'CHUA CAP NHAT'}")
    print(f"sections   : {len(doc.sections)}")
    for index, section in enumerate(doc.sections):
        print(f"  [{index}] le L{section.left_margin.cm:.1f} R{section.right_margin.cm:.1f} "
              f"T{section.top_margin.cm:.1f} B{section.bottom_margin.cm:.1f} "
              f"first_page={section.different_first_page_header_footer}")

    if problems:
        print(f"\nVAN DE ({len(problems)}):")
        for problem in dict.fromkeys(problems):
            print("  -", problem)
    else:
        print("\nKhong thay van de ve placeholder, markdown sot, be rong bang hay gian dong.")


if __name__ == "__main__":
    main()
