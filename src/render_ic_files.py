import json, os
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from docx import Document

sections = json.load(open("artifacts/ic_sections.json", encoding="utf-8"))

class ZethetaPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(95, 105, 115)
        self.cell(0, 5, "Project 1A: Bayesian Regime Detection Engine for Equity Direction Forecasting (ENSEMBLE_v3.2)", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Zetheta Algorithms Private Limited | CIN: U62012MH2023PTC410415 | Candidate: Gangadharan R | Page {self.page_no()}", align="C")

pdf = ZethetaPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_fill_color(20, 52, 92)
pdf.rect(10, 13, 190, 24, "F")
pdf.set_y(15)
pdf.set_font("Helvetica", "B", 13)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 6.5, "BAYESIAN REGIME DETECTION ENGINE FOR EQUITY DIRECTION FORECASTING", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 9.5)
pdf.cell(0, 5.5, "Complete Technical Report, Model Card, Calibration Audit & Investment Committee Dossier (Deliverables D3.1-D3.6)", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(7)

for title, items in sections:
    pdf.set_x(10)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 52, 92)
    pdf.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(20, 52, 92)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    for k, v in items:
        pdf.set_x(10)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(190, 4.8, f"- {k}: {v}")
        pdf.ln(0.8)
    pdf.ln(2)

pdf.add_page()
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(20, 52, 92)
pdf.cell(0, 7, "APPENDIX A: EMPIRICAL DIAGNOSTIC FIGURES (REGIMES, BACKTEST, SHAP & RS-VAR IRF)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.image("artifacts/figures/fig1_regimes_bocpd.png", x=10, w=190)
pdf.ln(2)
pdf.image("artifacts/figures/fig2_backtest_allocation.png", x=10, w=190)
pdf.add_page()
pdf.image("artifacts/figures/fig3_shap_irf.png", x=10, w=190)

pdf_path = "Project_1A_Bayesian_Regime_Detection_Report.pdf"
pdf.output(pdf_path)

doc = Document()
doc.sections[0].footer.paragraphs[0].text = "Zetheta Algorithms Private Limited | CIN: U62012MH2023PTC410415 | Candidate: Gangadharan R"
doc.add_heading("Project 1A: Bayesian Regime Detection Engine for Equity Direction Forecasting", level=0)
for title, items in sections:
    doc.add_heading(title, level=1)
    for k, v in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{k}: ").bold = True
        p.add_run(v)
for img in ["artifacts/figures/fig1_regimes_bocpd.png", "artifacts/figures/fig2_backtest_allocation.png", "artifacts/figures/fig3_shap_irf.png"]:
    doc.add_picture(img)
doc.save("docs/Project_1A_Bayesian_Regime_Detection_Report.docx")
print(f"[ALL DONE] {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB, {pdf.page_no()} pages) + DOCX saved")
