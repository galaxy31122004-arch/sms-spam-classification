"""Build the Word report from the approved Markdown chapters.

Layout follows docs/bao-cao/ban-thao-hoan-chinh/ho-so-bien-tap/quy-cach-trinh-bay.md, measured
from the group-20 machine-learning report. Content is not rewritten here: headings, tables,
captions, lists, code, formulas and figures are carried over as they stand in the Markdown.

Formulas are written in UnicodeMath between sentinels; cap-nhat-truong.ps1 asks Word to build
them into real equations, and to fill the table of contents and page-number fields.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cong_thuc import unconverted, wrap as wrap_math  # noqa: E402
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

AUDIT = Path(__file__).resolve().parent
REPORT = AUDIT.parent
FIGURES = REPORT / "hinh"
CHAPTERS = sorted(REPORT.glob("0*.md"))

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(14)
COVER_SIZE = Pt(13)
TABLE_SIZE = Pt(12)
CAPTION_SIZE = Pt(13)
SOURCE_SIZE = Pt(12)
CODE_FONT = "Consolas"
CODE_SIZE = Pt(10)
FOOTER_SIZE = Pt(11)
LINE_SPACING = 1.5
MARGINS = {"left": Cm(3), "right": Cm(2), "top": Cm(2), "bottom": Cm(2)}
TEXT_WIDTH = Cm(16.0)
# Formulas whose conversion still holds a backslash: they would print raw LaTeX, so the build
# reports them instead of letting them reach the document unnoticed.
LEFTOVER: list[str] = []
# The frame, the logo and the first-page footer of the cover live in this template's header.
TEMPLATE = Path(r"C:\Users\letha\.claude\skills\document-authoring\references\cover-template.docx")

# Cover wording and the student details come from D:\Randombullsheet\bcaohocmaynhom20.docx;
# only the student this report belongs to is carried over.
COVER = {
    "school": "TRƯỜNG ĐẠI HỌC THỦ ĐÔ HÀ NỘI",
    "faculty": "KHOA TOÁN – CÔNG NGHỆ THÔNG TIN",
    "rule": "-----    -----",
    "kind": "ĐỒ ÁN MÔN HỌC",
    "topic": "ĐỀ TÀI: PHÁT HIỆN TIN NHẮN SMS RÁC BẰNG MẠNG NƠ-RON MLP, LSTM VÀ TRANSFORMER",
    "lecturer": "Giảng viên: Nguyễn Quốc Tuấn",
    "student": "Sinh viên thực hiện: Ninh Thị Hạnh Chi",
    "code": "MSSV: 223001698",
    "place": "Hà Nội, năm 2026",
}


# --------------------------------------------------------------------------- inline formatting

def looks_like_math(body: str) -> bool:
    """Decide whether "$...$" is a formula or two literal dollar signs in a sentence.

    A single symbol such as $V$ or $t$ is a formula and carries no operator, so the test cannot
    look for operators. It uses whitespace instead: prose between two prices ("$300 for rent $")
    always contains a space, a formula written in this report never does — unless it has a
    backslash command, which only a formula has.
    """
    if "\n" in body or not body.strip():
        return False
    return "\\" in body or not any(character.isspace() for character in body)


def add_inline(paragraph, text: str, size=BODY_SIZE, italic=False, bold=False) -> None:
    """Render one Markdown span: code, bold, italics, links and inline math become Word runs."""
    for chunk in re.split(r"(`[^`]+`)", text):
        if not chunk:
            continue
        if chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 1:
            run = paragraph.add_run(chunk[1:-1])
            run.font.name = CODE_FONT
            run.font.size = Pt(size.pt - 2)
            run.italic, run.bold = italic, bold
            continue
        chunk = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", chunk)
        chunk = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", chunk)
        for part in re.split(r"(\*\*[^*]+\*\*|\*(?!\s)[^*\n]{1,80}(?<!\s)\*|\$[^$\n]{1,160}\$)", chunk):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                run = paragraph.add_run(part[2:-2])
                run.bold, run.italic = True, italic
                run.font.size = size
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                run = paragraph.add_run(part[1:-1])
                run.italic, run.bold = True, bold
                run.font.size = size
            elif part.startswith("$") and part.endswith("$") and looks_like_math(part[1:-1]):
                formula = wrap_math(part[1:-1], inline=True)
                if unconverted(formula):
                    LEFTOVER.append(part)
                run = paragraph.add_run(formula)
                run.font.size = size
            else:
                run = paragraph.add_run(part.replace("\\*", "*").replace("\\_", "_"))
                run.bold, run.italic = bold, italic
                run.font.size = size


# --------------------------------------------------------------------------- document helpers

def force_font(style, size=None, color=RGBColor(0, 0, 0)) -> None:
    """Pin a style to Times New Roman in black; a theme attribute would otherwise win."""
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rfonts.attrib.pop(qn(f"w:{attribute}"), None)
    for attribute in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attribute}"), BODY_FONT)
    if size is not None:
        style.font.size = size
    if color is not None:
        style.font.color.rgb = color


def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    force_font(normal, size=BODY_SIZE)
    fmt = normal.paragraph_format
    fmt.line_spacing = LINE_SPACING
    fmt.space_after = Pt(6)
    fmt.space_before = Pt(0)
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for level, before, after in ((1, 18, 6), (2, 12, 6), (3, 10, 4)):
        style = doc.styles[f"Heading {level}"]
        force_font(style, size=BODY_SIZE)
        style.font.bold = True
        heading_fmt = style.paragraph_format
        heading_fmt.space_before = Pt(before)
        heading_fmt.space_after = Pt(after)
        heading_fmt.line_spacing = LINE_SPACING
        heading_fmt.keep_with_next = True
        heading_fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for name in ("Hyperlink", "FollowedHyperlink"):
        if name in (style.name for style in doc.styles):
            link = doc.styles[name]
            link.font.color.rgb = RGBColor(0, 0, 0)
            link.font.underline = False


def apply_margins(section) -> None:
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = MARGINS["left"]
    section.right_margin = MARGINS["right"]
    section.top_margin = MARGINS["top"]
    section.bottom_margin = MARGINS["bottom"]


def cover_line(doc: Document, text: str, bold=False, center=True, before=0, after=0, indent=None):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = LINE_SPACING
    if indent is not None:
        fmt.left_indent = indent
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = COVER_SIZE
    return paragraph


def new_document() -> Document:
    """Open the cover template and empty its body, keeping the framed first-page header."""
    doc = Document(str(TEMPLATE))
    for paragraph in list(doc.paragraphs):
        paragraph._element.getparent().remove(paragraph._element)
    return doc


def build_cover(doc: Document) -> None:
    """Cover text of the reference report, inside the frame of the template."""
    cover_line(doc, COVER["school"], after=2)
    cover_line(doc, COVER["faculty"], bold=True, after=2)
    cover_line(doc, COVER["rule"], after=0)
    cover_line(doc, COVER["kind"], bold=True, before=90, after=10)
    cover_line(doc, COVER["topic"], bold=True, after=0)
    indent = Cm(2.0)
    cover_line(doc, COVER["lecturer"], center=False, before=90, after=6, indent=indent)
    cover_line(doc, COVER["student"], center=False, after=6, indent=indent)
    cover_line(doc, COVER["code"], center=False, after=0, indent=indent)
    # The place and year belong in the first-page footer, where the template's frame leaves room.
    footer = doc.sections[0].first_page_footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in list(paragraph.runs)[1:]:
        run._element.getparent().remove(run._element)
    if paragraph.runs:
        paragraph.runs[0].text = COVER["place"]
    else:
        paragraph.add_run(COVER["place"])
    paragraph.runs[0].bold = True
    paragraph.runs[0].font.size = COVER_SIZE
    paragraph.runs[0].font.name = BODY_FONT


def add_field(paragraph, instruction: str, placeholder: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, separate, text, end):
        run._r.append(node)
    return run


def start_body_section(doc: Document):
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    apply_margins(section)
    section.different_first_page_header_footer = False
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.line_spacing = 1.0
    run = add_field(paragraph, " PAGE   \\* MERGEFORMAT ", "1")
    run.font.size = FOOTER_SIZE
    run.font.name = BODY_FONT
    return section


def repeat_header_row(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def keep_row_whole(row) -> None:
    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))


def keep_table_together(table) -> None:
    for row in table.rows[:-1]:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = True


# --------------------------------------------------------------------------- Markdown blocks

def add_paragraph_block(doc: Document, text: str) -> None:
    stripped = text.strip()
    if not stripped:
        return
    caption = re.fullmatch(r"\*\*((?:Bảng|Hình) [^*]+)\*\*", stripped)
    if caption:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(6)
        paragraph.paragraph_format.keep_with_next = True
        add_inline(paragraph, caption.group(1), size=CAPTION_SIZE, bold=True)
        return
    source = re.fullmatch(r"\*(Nguồn:[^*]+)\*", stripped, re.S)
    if source:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(10)
        add_inline(paragraph, re.sub(r"\s+", " ", source.group(1)), size=SOURCE_SIZE, italic=True)
        return
    paragraph = doc.add_paragraph()
    add_inline(paragraph, stripped)


def add_list_item(doc: Document, text: str, level: int) -> None:
    paragraph = doc.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.left_indent = Cm(1.13 + 0.63 * level)
    fmt.first_line_indent = Cm(-0.63)
    fmt.space_before = Pt(4)
    fmt.space_after = Pt(0)
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_inline(paragraph, f"{'-' if level == 0 else '+'}\t{text}")


def add_numbered_item(doc: Document, number: str, text: str) -> None:
    paragraph = doc.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.left_indent = Cm(1.13)
    fmt.first_line_indent = Cm(-0.63)
    fmt.space_before = Pt(4)
    fmt.space_after = Pt(0)
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_inline(paragraph, f"{number}.\t{text}")


def add_code_block(doc: Document, lines: list[str]) -> None:
    for line in lines:
        paragraph = doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.left_indent = Cm(0.75)
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        fmt.line_spacing = 1.0
        fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = paragraph.add_run(line if line.strip() else " ")
        run.font.name = CODE_FONT
        run.font.size = CODE_SIZE
    doc.paragraphs[-1].paragraph_format.space_after = Pt(8)


def add_display_math(doc: Document, lines: list[str]) -> None:
    """One centred equation per paragraph, with its number on the right margin."""
    body = " ".join(line.strip() for line in lines if line.strip())
    tag = re.search(r"\\tag\{([^}]*)\}", body)
    body = re.sub(r"\\tag\{[^}]*\}", "", body).strip().rstrip(".")
    if unconverted(wrap_math(body)):
        LEFTOVER.append(body)
    paragraph = doc.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(10)
    run = paragraph.add_run(wrap_math(body))
    run.font.size = BODY_SIZE
    if tag:
        fmt.tab_stops.add_tab_stop(TEXT_WIDTH, WD_TAB_ALIGNMENT.RIGHT)
        number = paragraph.add_run(f"\t({tag.group(1)})")
        number.font.size = BODY_SIZE


def add_picture_block(doc: Document, path: Path, width: Cm = None) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=width or TEXT_WIDTH)


def picture_width(path: Path, max_height_cm: float = 19.0) -> Cm:
    from PIL import Image

    with Image.open(path) as image:
        ratio = image.height / image.width
    width_cm = TEXT_WIDTH.cm
    if width_cm * ratio > max_height_cm:
        width_cm = max_height_cm / ratio
    return Cm(width_cm)


def add_table_block(doc: Document, rows: list[str]) -> None:
    grid = [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows]
    alignment_row = next((index for index, row in enumerate(grid)
                          if all(re.fullmatch(r":?-{2,}:?", cell) for cell in row)), None)
    aligns = []
    if alignment_row is not None:
        for cell in grid[alignment_row]:
            aligns.append(WD_ALIGN_PARAGRAPH.RIGHT if cell.endswith(":") and not cell.startswith(":")
                          else WD_ALIGN_PARAGRAPH.CENTER if cell.startswith(":") and cell.endswith(":")
                          else WD_ALIGN_PARAGRAPH.LEFT)
        grid.pop(alignment_row)
    columns = max(len(row) for row in grid)
    table = doc.add_table(rows=len(grid), cols=columns)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for row_index, row in enumerate(grid):
        for column_index in range(columns):
            cell = table.cell(row_index, column_index)
            paragraph = cell.paragraphs[0]
            fmt = paragraph.paragraph_format
            fmt.line_spacing = 1.15
            fmt.space_before = Pt(2)
            fmt.space_after = Pt(2)
            fmt.alignment = (WD_ALIGN_PARAGRAPH.CENTER if row_index == 0
                             else aligns[column_index] if column_index < len(aligns)
                             else WD_ALIGN_PARAGRAPH.LEFT)
            add_inline(paragraph, row[column_index] if column_index < len(row) else "",
                       size=TABLE_SIZE, bold=row_index == 0)
    repeat_header_row(table.rows[0])
    for row in table.rows:
        keep_row_whole(row)
    if len(table.rows) <= 12:
        keep_table_together(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


# --------------------------------------------------------------------------- chapter conversion

def convert_chapter(doc: Document, path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").split("\n")
    counts = {"heading": 0, "table": 0, "figure": 0, "code": 0, "math": 0}
    paragraph_buffer: list[str] = []
    last_figure = None
    index = 0

    def flush():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            add_paragraph_block(doc, " ".join(paragraph_buffer))
            paragraph_buffer = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            language = stripped[3:].strip()
            block, index = [], index + 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            if language == "mermaid":
                name = f"hinh-{last_figure.replace('.', '-')}.png" if last_figure else None
                image = FIGURES / name if name else None
                if not image or not image.is_file():
                    raise SystemExit(f"Thieu anh cho {last_figure} trong {path.name}; chay render-so-do.py truoc")
                add_picture_block(doc, image, picture_width(image))
                counts["figure"] += 1
            else:
                add_code_block(doc, block)
                counts["code"] += 1
            continue

        if stripped == "$$":
            flush()
            block, index = [], index + 1
            while index < len(lines) and lines[index].strip() != "$$":
                block.append(lines[index])
                index += 1
            index += 1
            add_display_math(doc, block)
            counts["math"] += 1
            continue

        if stripped.startswith("|"):
            flush()
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            add_table_block(doc, block)
            counts["table"] += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush()
            level = min(len(heading.group(1)), 3)
            paragraph = doc.add_heading("", level=level)
            add_inline(paragraph, heading.group(2), bold=True)
            counts["heading"] += 1
            index += 1
            continue

        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            flush()
            target = (REPORT / image.group(2)).resolve()
            if not target.is_file():
                raise SystemExit(f"Thieu anh {target} tham chieu trong {path.name}")
            add_picture_block(doc, target, picture_width(target))
            counts["figure"] += 1
            index += 1
            continue

        bullet = re.match(r"^(\s*)-\s+(.*)$", line)
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if bullet and not stripped.startswith("---"):
            flush()
            add_list_item(doc, bullet.group(2), level=len(bullet.group(1)) // 2)
            index += 1
            continue
        if numbered:
            flush()
            add_numbered_item(doc, numbered.group(1), numbered.group(2))
            index += 1
            continue

        if not stripped or stripped == "---":
            flush()
            index += 1
            continue

        caption = re.match(r"^\*\*(?:Bảng|Hình) ([\d.]+)\.", stripped)
        if caption:
            last_figure = caption.group(1).rstrip(".")
        if re.fullmatch(r"\*\*[^*]+\*\*", stripped) or re.fullmatch(r"\*[^*]+\*", stripped, re.S):
            flush()
            add_paragraph_block(doc, stripped)
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "drafts" / "bao-cao-sms-spam.docx")
    args = parser.parse_args()

    doc = new_document()
    style_document(doc)
    apply_margins(doc.sections[0])
    build_cover(doc)
    start_body_section(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(12)
    title.paragraph_format.keep_with_next = True
    run = title.add_run("MỤC LỤC")
    run.bold = True
    run.font.size = BODY_SIZE
    add_field(doc.add_paragraph(), ' TOC \\o "1-3" \\h \\z \\u ',
              "Mục lục sẽ hiện sau khi cập nhật trường trong Word.")

    totals = {"heading": 0, "table": 0, "figure": 0, "code": 0, "math": 0}
    for chapter in CHAPTERS:
        counts = convert_chapter(doc, chapter)
        print(f"{chapter.name}: " + ", ".join(f"{key} {value}" for key, value in counts.items()))
        for key, value in counts.items():
            totals[key] += value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(args.output))
    print("\nTong: " + ", ".join(f"{key} {value}" for key, value in totals.items()))
    if LEFTOVER:
        print(f"CANH BAO: {len(LEFTOVER)} cong thuc van con lenh LaTeX chua chuyen:")
        for formula in dict.fromkeys(LEFTOVER):
            print(f"   {formula[:90]}")
    print(f"Da luu {args.output}")


if __name__ == "__main__":
    main()
