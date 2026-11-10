"""
ビジネス関係者（数値計算が弱い方）向けの説明スライドを生成する。
make_slide_charts.py で作った3枚の図を使い、python-pptxでpptxを組み立てる。
PowerPointでもGoogle Slides（pptxをインポート）でも開ける形式。
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --- パレット（dataviz skill の reference palette より） ---
INK_PRIMARY = RGBColor(0x0B, 0x0B, 0x0B)
INK_SECONDARY = RGBColor(0x52, 0x51, 0x4E)
PAGE_PLANE = RGBColor(0xF9, 0xF9, 0xF7)
SERIES_BLUE = RGBColor(0x2A, 0x78, 0xD6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Meiryo"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide():
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = PAGE_PLANE
    bg.line.fill.background()
    bg.shadow.inherit = False
    # 背面に送る
    spTree = slide.shapes._spTree
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)
    return slide


def add_title_bar(slide, title, accent=True):
    if accent:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), prs.slide_height)
        bar.fill.solid()
        bar.fill.fore_color.rgb = SERIES_BLUE
        bar.line.fill.background()
        bar.shadow.inherit = False

    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.2), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.name = FONT
    p.font.color.rgb = INK_PRIMARY


def add_bullets(slide, bullets, left=0.7, top=1.5, width=11.8, height=2.8, size=20, line_spacing=1.3):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"●  {b}"
        p.font.size = Pt(size)
        p.font.name = FONT
        p.font.color.rgb = INK_SECONDARY
        p.line_spacing = line_spacing
        p.space_after = Pt(10)


def add_picture_centered(slide, path, top, max_width, max_height):
    from PIL import Image
    im = Image.open(path)
    w, h = im.size
    ratio = min(max_width / w, max_height / h)
    disp_w, disp_h = w * ratio, h * ratio
    left = (prs.slide_width - Emu(int(disp_w))) / 2
    slide.shapes.add_picture(path, left, Inches(top), width=Emu(int(disp_w)), height=Emu(int(disp_h)))


def add_footer(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.7), Inches(7.05), Inches(11.8), Inches(0.35))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(11)
    p.font.name = FONT
    p.font.color.rgb = INK_SECONDARY


# ============================================================
# Slide 1: 表紙
# ============================================================
slide = add_slide()
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6), prs.slide_width, Inches(2.3))
bar.fill.solid()
bar.fill.fore_color.rgb = SERIES_BLUE
bar.line.fill.background()
bar.shadow.inherit = False

tb = slide.shapes.add_textbox(Inches(0.9), Inches(2.9), Inches(11.5), Inches(1.0))
p = tb.text_frame.paragraphs[0]
p.text = "保険販売量予測モデルのご説明"
p.font.size = Pt(40)
p.font.bold = True
p.font.name = FONT
p.font.color.rgb = WHITE

tb2 = slide.shapes.add_textbox(Inches(0.9), Inches(3.9), Inches(11.5), Inches(0.8))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "「積み上がり方のクセ」を使って、月末の着地を先読みする仕組み"
p2.font.size = Pt(20)
p2.font.name = FONT
p2.font.color.rgb = WHITE

add_footer(slide, "2026年7月")


# ============================================================
# Slide 2: 背景
# ============================================================
slide = add_slide()
add_title_bar(slide, "保険の「計上」には特殊なクセがある")
add_bullets(slide, [
    "21日には、その直前の数日分（15〜20日ごろ）の実売分をまとめて登録する運用があり、21日の数字は「その日1日分の実績」ではなく、まとめ球（スパイク）",
    "1日時点で実績を見直すと、想定より多い/少ないことが分かり、予測（①→②）が変わることがある",
], top=1.2, height=1.3, size=18)
add_picture_centered(slide, "chart_spike_calendar.png", top=2.5, max_width=Inches(11.3), max_height=Inches(4.6))


# ============================================================
# Slide 3: 課題
# ============================================================
slide = add_slide()
add_title_bar(slide, "単純な「日割り」では読み違える")
add_bullets(slide, [
    "「経過日数 ÷ サイクル全体の日数」で単純に延ばすと、スパイクの分だけ数字が過大に見えてしまう",
    "保険は「立ち上がりが早く、その後はゆっくり積み上がる」という独特のクセを持っている",
    "→ このクセを反映しないと、21日時点の見立てが大きくズレる",
], top=1.6, height=3.0, size=22)


# ============================================================
# Slide 4: 発想の転換
# ============================================================
slide = add_slide()
add_title_bar(slide, "発想：過去の「進み方のクセ」に当てはめる")
add_bullets(slide, [
    "過去の月について、「その時点の実績が最終的に何割まで来ていたか」をあらかじめ調べておく",
    "今月の実績を、その「何割」で割り戻すと、月末の姿が見えてくる",
    "たとえるなら、マラソンの30km地点の通過タイムから、過去のペース配分を参考にゴール時間を読むのと同じ考え方",
], top=1.6, height=3.4, size=22)


# ============================================================
# Slide 5: 進捗ゲージ
# ============================================================
slide = add_slide()
add_title_bar(slide, "過去のデータから見えた「進み方」")
add_bullets(slide, [
    "統括部ごとに多少の差はあるが、おおよそ次のペースで積み上がる（5月の実績を基に算出）",
], top=1.4, height=0.8, size=20)
add_picture_centered(slide, "chart_progress_gauge.png", top=2.3, max_width=Inches(11.0), max_height=Inches(3.8))


# ============================================================
# Slide 6: 計算のイメージ
# ============================================================
slide = add_slide()
add_title_bar(slide, "予測の計算はこれだけ")

box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.7), Inches(10.3), Inches(1.5))
box.fill.solid()
box.fill.fore_color.rgb = WHITE
box.line.color.rgb = SERIES_BLUE
box.line.width = Pt(1.5)
box.shadow.inherit = False
tf = box.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
p.text = "今の実績　÷　進み具合（％）　＝　月末の予測"
p.font.size = Pt(26)
p.font.bold = True
p.font.name = FONT
p.font.color.rgb = INK_PRIMARY

add_bullets(slide, [
    "例：21日時点で実績が60件、進み具合が6割（＝約61%）の統括部の場合",
    "　　60件 ÷ 0.61 ≒ 約100件 → これが月末の予測着地",
    "難しい統計モデルではなく、割り算1つで計算できる",
    "実績が新しく分かるたびに、同じ計算をやり直して予測を更新する（ローリング再予測）",
], top=3.6, height=2.8, size=20)


# ============================================================
# Slide 7: 検証結果
# ============================================================
slide = add_slide()
add_title_bar(slide, "実際に当てはめてみた結果（5月のクセ→6月で検証）")
add_picture_centered(slide, "chart_comparison_bar.png", top=1.35, max_width=Inches(11.3), max_height=Inches(4.6))
add_bullets(slide, [
    "5月の「進み方のクセ」を使って6月の着地を予測し、実際の6月の実績と比較",
    "概ね数%以内の誤差で着地を捉えられている",
], top=6.0, height=1.2, size=17)


# ============================================================
# Slide 8: 注意点
# ============================================================
slide = add_slide()
add_title_bar(slide, "今の精度についての注意点")
add_bullets(slide, [
    "参考にできる実績がまだ2ヶ月分しかなく、「進み方のクセ」自体は暫定値",
    "支社ごとの違いはまだ反映しきれておらず、まずは統括部（5部）単位で共通のクセを使っている",
    "今後、データが半年〜1年分と増えるにつれて、支社別・より高度な統計/AIモデルへ段階的に移行していく計画",
], top=1.6, height=3.4, size=22)


# ============================================================
# Slide 9: まとめ
# ============================================================
slide = add_slide()
add_title_bar(slide, "まとめ")
add_bullets(slide, [
    "保険特有の「先取り計上（スパイク）」を踏まえた予測ロジックを構築した",
    "21日時点・1日時点の2段階で、その都度予測を更新する（ローリング再予測）",
    "計算の中身は割り算1つとシンプルだが、過去の「進み方のクセ」を反映することで単純な日割りより精度が高い",
    "データが増えるほど、クセの精度も自動的に上がっていく設計",
], top=1.6, height=3.6, size=22)


prs.save("insurance_forecast_explainer.pptx")
print("saved: insurance_forecast_explainer.pptx")
