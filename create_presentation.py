from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_THEME_COLOR

ROOT = Path(__file__).parent
OUT = ROOT / "PricePilot_AI_Presentation.pptx"
NAVY = RGBColor(14, 38, 55)
TEAL = RGBColor(20, 125, 146)
GOLD = RGBColor(220, 164, 73)
MINT = RGBColor(226, 241, 238)
INK = RGBColor(42, 55, 62)
MUTED = RGBColor(91, 108, 116)
WHITE = RGBColor(255, 255, 255)
PALE = RGBColor(247, 250, 249)
RED = RGBColor(183, 83, 79)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def box(slide, x, y, w, h, fill=WHITE, line=None, radius=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    if radius:
        shape.adjustments[0] = 0.08
    return shape


def text(slide, value, x, y, w, h, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, font="Aptos"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.clear(); tf.word_wrap = True; tf.margin_left = Inches(.03); tf.margin_right = Inches(.03)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = value; r.font.name = font; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return tb


def add_notes(slide, value):
    try:
        slide.notes_slide.notes_text_frame.text = value
    except Exception:
        pass


def base(title, kicker, number):
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    text(slide, kicker.upper(), .65, .38, 5, .25, 9, TEAL, True)
    text(slide, title, .65, .72, 11.5, .55, 27, NAVY, True, font="Aptos Display")
    text(slide, f"{number:02d}", 12.05, .42, .6, .3, 10, MUTED, True, PP_ALIGN.RIGHT)
    divider = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.65), Inches(1.43), Inches(12.05), Inches(.015)); divider.fill.solid(); divider.fill.fore_color.rgb = MINT
    return slide


def bullet(slide, value, x, y, w, color=INK):
    text(slide, "•  " + value, x, y, w, .42, 15, color)


def metric(slide, label, value, x, y, w=2.2, accent=TEAL):
    box(slide, x, y, w, .95, PALE, MINT)
    text(slide, label.upper(), x+.16, y+.14, w-.3, .18, 8, MUTED, True)
    text(slide, value, x+.16, y+.4, w-.3, .35, 19, accent, True)

# 1
s = prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.7), Inches(0), Inches(4.65), Inches(7.5)); accent.fill.solid(); accent.fill.fore_color.rgb = TEAL
for i, (x, y, w, h, c) in enumerate([(9.35, 1.05, 2.9, .65, GOLD), (9.85, 2.1, 2.25, .65, WHITE), (9.2, 3.2, 3.05, .65, MINT), (9.8, 4.3, 2.4, .65, GOLD)]):
    box(s, x, y, w, h, c, c)
    text(s, ["PRICE", "DEMAND", "REVENUE", "PROFIT"][i], x, y+.18, w, .25, 13, NAVY, True, PP_ALIGN.CENTER)
text(s, "PRICEPILOT AI", .8, 1.35, 6.8, .35, 13, GOLD, True)
text(s, "Intelligent Retail\nPrice Recommendation\nSystem", .8, 1.9, 7.1, 2.05, 34, WHITE, True, font="Aptos Display")
text(s, "Turning sales data into practical pricing decisions.", .85, 4.45, 6.4, .45, 18, MINT)
text(s, "Presented by [Name]  |  [Institution]  |  [Date]", .85, 6.55, 7, .25, 11, WHITE)
add_notes(s, "Introduce PricePilot AI as a decision-support tool that helps businesses choose a practical price using historical demand, costs, competition, and inventory.")

# 2
s = base("The pricing decision is a balancing act", "The business problem", 2)
text(s, "A price that looks attractive in one direction can create a problem somewhere else.", .75, 1.8, 7, .4, 18, MUTED)
for x, title, desc, color in [(1, "Too low", "Demand may rise, but margin disappears.", RED), (4.35, "Too high", "Margin improves, but sales may slow.", GOLD), (7.7, "Guesswork", "Decisions become inconsistent and hard to defend.", TEAL)]:
    box(s, x, 2.55, 2.8, 2.15, PALE, MINT); text(s, title, x+.22, 2.85, 2.3, .35, 20, color, True); text(s, desc, x+.22, 3.45, 2.3, .75, 15, INK)
text(s, "The goal is not simply the highest price. It is the strongest expected outcome under real business constraints.", 1, 5.45, 10.8, .55, 20, NAVY, True, PP_ALIGN.CENTER)
add_notes(s, "Explain that pricing affects customer demand, profitability, competitiveness, and inventory at the same time. This is why a structured recommendation is useful.")

# 3
s = base("One recommendation, four business priorities", "The objective", 3)
text(s, "PricePilot compares possible prices and makes the trade-off visible before a decision is approved.", .75, 1.8, 8.8, .4, 18, MUTED)
for x, y, label, desc, color in [(1.0, 2.8, "Demand", "Expected units sold", TEAL), (4.1, 2.8, "Revenue", "Sales value", GOLD), (7.2, 2.8, "Profit", "Value after cost", NAVY), (10.3, 2.8, "Inventory", "Available supply", RED)]:
    box(s, x, y, 2.0, 1.65, color, color); text(s, label, x+.1, y+.42, 1.8, .3, 19, WHITE, True, PP_ALIGN.CENTER); text(s, desc, x+.12, y+1.02, 1.76, .25, 11, WHITE, False, PP_ALIGN.CENTER)
text(s, "Business priority", 5.15, 5.15, 3.0, .3, 13, TEAL, True, PP_ALIGN.CENTER)
text(s, "Maximize revenue  •  Maximize profit  •  Balanced", 3.2, 5.55, 7, .35, 16, NAVY, True, PP_ALIGN.CENTER)
add_notes(s, "Emphasize that the user selects the business priority. The system supports the decision; it does not make an autonomous decision without human review.")

# 4
s = base("The model learns from the business context", "The data", 4)
items = [("Cost", "price floor"), ("Current price", "starting point"), ("Competitor", "market context"), ("Inventory", "supply limit"), ("Promotion", "sales signal"), ("Season", "time pattern"), ("Rating", "customer signal"), ("History", "observed demand")]
for i, (label, desc) in enumerate(items):
    x = .85 + (i % 4) * 3.05; y = 1.95 + (i // 4) * 1.65
    box(s, x, y, 2.65, 1.15, PALE, MINT)
    text(s, label, x+.18, y+.22, 2.25, .3, 17, NAVY, True); text(s, desc, x+.18, y+.67, 2.25, .2, 11, MUTED)
metric(s, "Historical records", "5,000", .85, 5.5, 2.55, TEAL); metric(s, "Data period", "2023–2024", 3.65, 5.5, 2.55, GOLD); metric(s, "Target", "Units sold", 6.45, 5.5, 2.55, NAVY); metric(s, "Validation", "Chronological", 9.25, 5.5, 2.55, RED)
add_notes(s, "Describe the inputs in business language. The model uses 5,000 historical records from 2023 through 2024 and predicts units sold as the demand target.")

# 5
s = base("From historical data to an actionable price", "How it works", 5)
steps = [("01", "Historical\nsales", TEAL), ("02", "Prepare\ndata", GOLD), ("03", "Predict\ndemand", NAVY), ("04", "Simulate\nprices", TEAL), ("05", "Recommend\nprice", GOLD)]
for i, (num, label, color) in enumerate(steps):
    x = .75 + i * 2.48
    box(s, x, 2.45, 1.85, 1.55, color, color); text(s, num, x+.15, 2.67, .5, .25, 11, WHITE, True); text(s, label, x+.15, 3.05, 1.55, .6, 17, WHITE, True, PP_ALIGN.CENTER)
    if i < 4: text(s, "→", x+1.92, 2.98, .55, .35, 25, GOLD, True, PP_ALIGN.CENTER)
text(s, "The recommendation is based on the best expected outcome among tested scenarios, subject to cost and competitor constraints.", 1.2, 5.25, 10.9, .55, 18, NAVY, True, PP_ALIGN.CENTER)
add_notes(s, "Walk through the five-step pipeline. The important idea for a non-technical audience is that the system evaluates several realistic price options instead of guessing one number.")

# 6
s = base("Gradient Boosting gives the strongest test performance", "The machine learning approach", 6)
chart = CategoryChartData(); chart.categories = ["Mean baseline", "Linear Regression", "Random Forest", "Gradient Boosting"]; chart.add_series("R² score", [-.005, .723, .747, .762])
chart_shape = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(.85), Inches(1.95), Inches(7.1), Inches(4.3), chart)
chart_obj = chart_shape.chart; chart_obj.has_legend = False; chart_obj.value_axis.minimum_scale = 0; chart_obj.value_axis.maximum_scale = 1; chart_obj.value_axis.has_major_gridlines = True
chart_obj.category_axis.tick_labels.font.size = Pt(12); chart_obj.value_axis.tick_labels.font.size = Pt(10)
series = chart_obj.series[0]; series.format.fill.solid(); series.format.fill.fore_color.rgb = TEAL
box(s, 8.45, 2.1, 3.65, 2.45, PALE, MINT); text(s, "Best model", 8.75, 2.45, 2.8, .25, 11, MUTED, True); text(s, "Gradient\nBoosting", 8.75, 2.85, 2.9, .75, 24, NAVY, True); text(s, "R² = 0.762", 8.75, 3.85, 2.8, .3, 18, TEAL, True)
text(s, "Lower error and higher R² indicate better agreement with held-out historical demand.", 8.55, 5.15, 3.45, .65, 15, INK)
add_notes(s, "Explain the chart simply: the baseline is a useful reference, and Gradient Boosting performs best on the chronological test set with an R² of 0.762.")

# 7 dashboard mockup
s = base("A focused workflow for the decision maker", "The application", 7)
box(s, .8, 1.8, 11.7, 4.85, RGBColor(250, 252, 251), MINT)
text(s, "PricePilot", 1.1, 2.08, 2.3, .35, 21, NAVY, True); text(s, "A clear, data-informed price recommendation for every business decision.", 1.1, 2.53, 5.8, .25, 11, MUTED)
text(s, "SET THE SCENARIO", 1.1, 3.08, 2.5, .2, 9, TEAL, True)
for x, label, val in [(1.1, "Product category", "Grocery"), (3.8, "Cost price", "$18.00"), (6.5, "Current price", "$24.00"), (9.2, "Competitor price", "$25.00")]:
    text(s, label, x, 3.48, 2.2, .18, 10, MUTED, True); box(s, x, 3.72, 2.15, .48, WHITE, RGBColor(205, 220, 216)); text(s, val, x+.15, 3.87, 1.8, .18, 13, INK)
text(s, "Inventory: 240 units", 1.1, 4.62, 2.4, .25, 12, INK); text(s, "Promotion active  □", 3.8, 4.62, 2.4, .25, 12, INK); text(s, "Priority: Balanced", 6.5, 4.62, 2.5, .25, 12, INK)
box(s, 9.2, 4.45, 2.15, .58, TEAL, TEAL); text(s, "GENERATE", 9.35, 4.64, 1.85, .18, 12, WHITE, True, PP_ALIGN.CENTER)
text(s, "Recommendation results appear here after one click, with no code or technical files in the user experience.", 1.1, 5.75, 9.8, .3, 13, NAVY, True)
add_notes(s, "Show how the user interacts with the application. This is the clean front end: the audience enters business conditions and receives a recommendation without seeing the underlying project files.")

# 8
s = base("The output is a decision, not just a number", "Example recommendation", 8)
metric(s, "Recommended price", "$26.40", .85, 1.9, 2.25, TEAL); metric(s, "Predicted demand", "184 units", 3.3, 1.9, 2.25, NAVY); metric(s, "Expected revenue", "$4,857.60", 5.75, 1.9, 2.25, GOLD); metric(s, "Expected profit", "$1,545.60", 8.2, 1.9, 2.25, TEAL); metric(s, "Profit margin", "31.8%", 10.65, 1.9, 1.85, RED)
chart = CategoryChartData(); chart.categories = ["$20", "$22", "$24", "$26", "$28", "$30"]; chart.add_series("Expected revenue", [3900, 4300, 4550, 4858, 4700, 4420]); chart.add_series("Expected profit", [900, 1120, 1320, 1546, 1510, 1430])
ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(1.1), Inches(3.35), Inches(7.1), Inches(2.65), chart).chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.value_axis.has_major_gridlines = True
ch.series[0].format.line.color.rgb = TEAL; ch.series[1].format.line.color.rgb = GOLD
box(s, 8.65, 3.55, 3.2, 2.1, PALE, MINT); text(s, "Why this option?", 8.95, 3.86, 2.5, .25, 13, TEAL, True); text(s, "It produces the strongest expected outcome among the tested prices while respecting cost, competition, and inventory assumptions.", 8.95, 4.35, 2.5, .95, 14, INK)
add_notes(s, "Use this slide to demonstrate the result. The figures are an illustrative scenario layout; replace them with a screenshot of the actual recommendation generated by the app during the live demo.")

# 9
s = base("Useful guidance with transparent boundaries", "Business value and limitations", 9)
box(s, .85, 1.95, 5.55, 3.85, MINT, MINT); text(s, "Business value", 1.2, 2.35, 4.5, .35, 22, NAVY, True)
for i, v in enumerate(["Faster, more consistent pricing decisions", "Better visibility into demand and margin trade-offs", "Practical support for inventory planning", "Clear assumptions for human review"]): bullet(s, v, 1.2, 3.0 + i*.55, 4.65)
box(s, 6.95, 1.95, 5.55, 3.85, PALE, MINT); text(s, "Important limitations", 7.3, 2.35, 4.5, .35, 22, NAVY, True)
for i, v in enumerate(["Historical patterns may change", "Competitor prices can become stale", "Predictions are estimates, not guarantees", "A human approves the final decision"]): bullet(s, v, 7.3, 3.0 + i*.55, 4.65)
add_notes(s, "Be direct about responsible use. PricePilot is decision support, not autonomous pricing. The recommendation should be reviewed alongside current market knowledge and business judgment.")

# 10
s = base("From data to better decisions", "Conclusion and future work", 10)
text(s, "PricePilot AI turns sales data into practical pricing decisions.", .85, 1.85, 11, .55, 25, NAVY, True, PP_ALIGN.CENTER)
for x, num, label in [(1.4, "01", "Predict demand"), (4.75, "02", "Compare outcomes"), (8.1, "03", "Support human review")]:
    box(s, x, 3.0, 2.55, 1.45, MINT, MINT); text(s, num, x+.18, 3.25, .45, .25, 11, TEAL, True); text(s, label, x+.18, 3.75, 2.15, .3, 18, NAVY, True, PP_ALIGN.CENTER)
text(s, "Next: live sales data  •  real-time competitor prices  •  customer segments  •  A/B price testing  •  continuous retraining", .95, 5.35, 11.5, .45, 15, MUTED, False, PP_ALIGN.CENTER)
text(s, "Thank you", .85, 6.35, 11.6, .35, 20, GOLD, True, PP_ALIGN.CENTER)
add_notes(s, "Close by summarizing the value: PricePilot predicts demand, compares pricing outcomes, and gives decision makers a transparent starting point. Mention the future improvements briefly.")

prs.save(OUT)
print(f"Created {OUT}")
