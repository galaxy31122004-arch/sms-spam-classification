"""Build the SMS spam presentation as an editable 16:9 PPTX (light theme, no content truncation)."""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

BG = RGBColor(0xFF, 0xFF, 0xFF)
FG = RGBColor(0x0F, 0x17, 0x2A)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
MUTED = RGBColor(0x64, 0x74, 0x8B)
PANEL = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

W, H = Inches(13.333), Inches(7.5)


def box(slide, x, y, w, h, fill=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.line.fill.background()
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    return shape


def text(slide, x, y, w, h, value, size, color=FG, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP):
    frame = slide.shapes.add_textbox(x, y, w, h).text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    lines = value if isinstance(value, list) else [value]
    for i, line in enumerate(lines):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color
    return frame


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.layout = self.prs.slide_layouts[6]  # blank: no placeholders
        self.number = 0

    def new(self):
        self.number += 1
        slide = self.prs.slides.add_slide(self.layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BG
        return slide

    def footer(self, slide):
        text(slide, Inches(0.6), Inches(6.95), Inches(9), Inches(0.4),
             "Phát hiện SMS rác bằng MLP, LSTM và Transformer", 12, MUTED)
        text(slide, Inches(11.7), Inches(6.95), Inches(1.0), Inches(0.4),
             f"{self.number:02d}", 12, MUTED, align=PP_ALIGN.RIGHT)

    def content(self, title):
        slide = self.new()
        box(slide, 0, 0, W, Inches(0.12), ACCENT)
        text(slide, Inches(0.6), Inches(0.45), Inches(12.1), Inches(0.9), title, 32, FG, True)
        box(slide, Inches(0.6), Inches(1.3), Inches(1.0), Inches(0.06), ACCENT)
        self.footer(slide)
        return slide

    def title(self, title, lines):
        slide = self.new()
        box(slide, 0, 0, Inches(0.35), H, ACCENT)
        text(slide, Inches(1.0), Inches(0.6), Inches(11.5), Inches(1.0),
             ["TRƯỜNG ĐẠI HỌC THỦ ĐÔ HÀ NỘI", "KHOA TOÁN – CÔNG NGHỆ THÔNG TIN"], 18, MUTED, True)
        text(slide, Inches(1.0), Inches(2.3), Inches(11.5), Inches(2.0), title, 44, FG, True,
             anchor=MSO_ANCHOR.MIDDLE)
        box(slide, Inches(1.0), Inches(4.45), Inches(1.4), Inches(0.08), ACCENT)
        text(slide, Inches(1.0), Inches(4.8), Inches(11.5), Inches(1.6), lines, 22, FG)

    def section(self, name):
        slide = self.new()
        box(slide, 0, 0, W, H, ACCENT)
        text(slide, Inches(1.0), Inches(2.8), Inches(11.3), Inches(1.6), name, 44, WHITE, True,
             anchor=MSO_ANCHOR.MIDDLE)

    def bullets(self, title, items, size=24):
        slide = self.content(title)
        frame = slide.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.8), Inches(5.0)).text_frame
        frame.word_wrap = True
        first = True
        for item in items:
            level = 1 if item.startswith("  ") else 0
            p = frame.paragraphs[0] if first else frame.add_paragraph()
            first = False
            p.space_after = Pt(10)
            run = p.add_run()
            run.text = ("–  " if level else "•  ") + item.strip()
            run.font.size = Pt(size - 4 * level)
            run.font.name = FONT
            run.font.color.rgb = MUTED if level else FG
            if level:
                p.level = 1
        return slide

    def table(self, title, header, rows, note=None, top=Inches(1.8), widths=None, size=20):
        slide = self.content(title)
        height = Inches(0.55) * (len(rows) + 1)
        shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(0.8), top,
                                       Inches(11.7), height)
        table = shape.table
        widths = widths or [11.7 / len(header)] * len(header)
        for i, width in enumerate(widths):
            table.columns[i].width = Inches(width)
        for r, values in enumerate([header] + rows):
            for c, value in enumerate(values):
                cell = table.cell(r, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT if r == 0 else (PANEL if r % 2 == 0 else WHITE)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                run = p.add_run()
                run.text = value
                run.font.size, run.font.name = Pt(size), FONT
                run.font.bold = r == 0 or c == 0
                run.font.color.rgb = WHITE if r == 0 else FG
        if note:
            text(slide, Inches(0.8), top + height + Inches(0.7), Inches(11.7), Inches(1.2),
                 note, 20, FG)
        return slide


def build(out: Path):
    d = Deck()
    d.title("Phát hiện tin nhắn SMS rác bằng MLP, LSTM và Transformer",
            ["Sinh viên: Ninh Thị Hạnh Chi – MSSV 223001698",
             "Giảng viên hướng dẫn: Nguyễn Quốc Tuấn"])
    d.bullets("Bài toán và mục tiêu", [
        "Phân loại SMS tiếng Anh thành HAM (0) hoặc SPAM (1)",
        "Xây dựng ba mô hình học từ đầu trên cùng dữ liệu: MLP, BiLSTM, Transformer nhỏ",
        "So sánh theo cùng quy trình: cùng split, cùng ngưỡng 0,5, cùng metric",
        "Minh họa bằng website demo phân loại tin nhắn",
    ])
    d.section("Dữ liệu và tiền xử lý")
    d.bullets("Bộ dữ liệu SMS Spam Collection", [
        "5.572 SMS, chia train / validation / test = 3.900 / 836 / 836 (seed 42)",
        "Test gồm 724 HAM và 112 SPAM",
        "Hai tập đánh giá bổ sung:",
        "  387 SMS thật ngoài nguồn (Mendeley SMS Phishing)",
        "  689 SMS lừa đảo tài chính tổng hợp (africa-smishing)",
    ])
    d.bullets("Chuẩn hóa văn bản dùng chung", [
        "Chữ thường, chuẩn hóa Unicode",
        "URL → <URL>, số điện thoại → <PHONE>, số tiền → <MONEY>",
        "Giữ dấu câu ! $ %",
        "MLP dùng TF-IDF từ và ký tự",
        "BiLSTM và Transformer dùng token ID từ vocab học trên train",
    ])
    d.section("Ba mô hình")
    d.table("Ba kiến trúc được so sánh", ["Mô hình", "Đầu vào", "Khối chính", "Đầu ra"], [
        ["MLP", "Vector TF-IDF", "Các lớp kết nối đầy đủ", "Sigmoid"],
        ["BiLSTM", "Token ID → embedding", "LSTM hai chiều", "Sigmoid"],
        ["Transformer", "Embedding token + vị trí", "2 tầng encoder, masked mean pooling", "Sigmoid"],
    ], note="Chọn checkpoint theo F1 SPAM cao nhất trên validation; ngưỡng quyết định 0,5.",
       widths=[2.2, 3.2, 4.3, 2.0])
    d.section("Kết quả")
    d.table("Kết quả trên test UCI (836 SMS)", ["Chỉ số", "MLP", "BiLSTM", "Transformer"], [
        ["Accuracy", "99,16%", "98,68%", "98,33%"],
        ["Precision SPAM", "100,00%", "99,03%", "94,55%"],
        ["Recall SPAM", "93,75%", "91,07%", "92,86%"],
        ["F1 SPAM", "96,77%", "94,88%", "93,69%"],
        ["ROC-AUC", "98,13%", "98,43%", "99,38%"],
        ["FP / FN", "0 / 7", "1 / 10", "6 / 8"],
    ], note="MLP có F1 cao nhất; Transformer có ROC-AUC cao nhất.")
    d.table("Chi phí tính toán", ["Chỉ số", "MLP", "BiLSTM", "Transformer"], [
        ["Số tham số", "3.873.281", "1.158.273", "1.155.457"],
        ["Thời gian train (giây)", "8,9", "29,8", "104,9"],
        ["Latency batch 1 (ms/tin)", "2,56", "1,65", "2,26"],
        ["Latency batch 32 (ms/tin)", "0,44", "0,47", "0,81"],
    ], note="Đo trên CPU Intel Core i7-12700H, một lần đo.")
    d.table("Dữ liệu thật ngoài nguồn: thứ hạng đảo ngược",
            ["Mô hình", "F1 SPAM", "Recall SPAM", "Bỏ sót SPAM"], [
        ["Transformer", "97,56%", "96,39%", "6 / 166"],
        ["BiLSTM", "94,60%", "89,76%", "17 / 166"],
        ["MLP", "83,92%", "72,29%", "46 / 166"],
    ], note="387 SMS thật. F1 cao trên test cùng nguồn không bảo đảm khả năng tổng quát.")
    d.table("Bổ sung dữ liệu lừa đảo tổng hợp",
            ["F1 SPAM (trước → sau)", "Test UCI", "Test tổng hợp", "Test ngoài nguồn"], [
        ["MLP", "96,77 → 95,37", "34,28 → 100", "83,92 → 89,04"],
        ["BiLSTM", "94,88 → 92,45", "47,40 → 100", "94,60 → 93,25"],
        ["Transformer", "93,69 → 93,46", "38,43 → 100", "97,56 → 94,97"],
    ], note="Thêm 3.906 SMS tổng hợp vào train: đạt 100% trên test tổng hợp nhưng giảm trên "
            "test UCI; trên dữ liệu thật chỉ MLP cải thiện.")
    d.table("Kiểm tra với 66 SMS tự soạn", ["Mô hình", "Nhận đúng", "Báo nhầm HAM", "Bỏ sót SPAM"], [
        ["Transformer baseline", "59/66", "5", "2"],
        ["Transformer augmented", "56/66", "5", "5"],
        ["BiLSTM baseline", "54/66", "4", "8"],
        ["MLP baseline", "50/66", "5", "11"],
    ], note=["Báo nhầm: tin giao dịch có số tiền, mã vận đơn bị nhận là <PHONE>",
             "Bỏ sót: lừa đảo giả làm người thân, không có liên kết"], size=18)
    d.section("Website demo")
    d.bullets("Website phân loại SMS", [
        "Flask, dùng Transformer baseline – tốt nhất trên dữ liệu ngoài nguồn",
        "Hiển thị các bước: chuẩn hóa → mã hóa token → logit → sigmoid → nhãn",
        "Đã kiểm thử các nhánh GET/POST và kiểm tra đầu vào",
    ])
    d.bullets("Kết luận", [
        "MLP tốt nhất trên test cùng nguồn; Transformer tổng quát tốt nhất trên SMS thật",
        "Thứ hạng mô hình phụ thuộc vào dữ liệu đánh giá",
        "Hạn chế: 84 tin test trùng với train, một seed, tập ngoài nguồn nhỏ",
        "Hướng phát triển:",
        "  Chia lại dữ liệu theo nhóm nội dung",
        "  Thu thập SMS lừa đảo thật, hiện đại",
        "  Sửa luật nhận số điện thoại",
    ], size=22)
    d.section("Cảm ơn thầy và các bạn đã lắng nghe")
    d.prs.save(out)
    print(f"Saved {out} | slides: {d.number}")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
