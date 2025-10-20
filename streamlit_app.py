# streamlit_app.py
# とどランURL×2 → 都道府県データ抽出、相関分析（外れ値あり／なし散布図）
# ・割合列も許可（オプション）
# ・「偏差値」や「順位」列は除外
# ・外れ値は「X軸で外れ値」「Y軸で外れ値」のみ横並び2カラム表示
# ・グレースケールデザイン／中央寄せ／アクセシビリティ配慮／タイトル余白修正
# ・結果分析（計算結果をSessionに保存→ボタン外置き）
# ・「クリア」ボタンで2つのURLと計算結果をリセット（on_click方式）
# ・結合後データのCSV保存ボタンを追加／外れ値CSV保存ボタンは削除
# ・結果分析ボタンを押しても散布図・結合表が消えないよう永続表示
# ・結合表の直下に「外れ値を含む散布図＋周辺箱ひげ図（左・下）」を追加

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import requests
from bs4 import BeautifulSoup
from pandas.api.types import is_scalar
from pathlib import Path

# === フォント設定 ===
fp = Path("fonts/SourceHanCodeJP-Regular.otf")
if fp.exists():
    fm.fontManager.addfont(str(fp))
    plt.rcParams["font.family"] = "Source Han Code JP"
else:
    for name in ["Noto Sans JP","IPAexGothic","Yu Gothic","Hiragino Sans","Meiryo"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="CorrGraph", layout="wide")

# タイトルの上に余白を追加
st.markdown("""
<style>
h1 { margin-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("CorrGraph")
st.write("とどランの **各ランキング記事のURL** を2つ貼り付けてください。")

# ====== UIテーマ（グレースケール＆アクセシビリティ） ======
plt.style.use("grayscale")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#888",
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})
DEFAULT_MARKER_SIZE = 36
DEFAULT_LINE_WIDTH = 2.0

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  color: #111 !important;
  background: #f5f5f5 !important;
}
.block-container {
  max-width: 980px;
  padding-top: 1.2rem;
  padding-bottom: 3rem;
}
h1, h2, h3 { color: #111 !important; letter-spacing: .01em; }
h1 { font-weight: 800; }
h2, h3 { font-weight: 700; }
p, li, .stMarkdown { line-height: 1.8; font-size: 1.02rem; }
input, textarea, select, .stTextInput > div > div > input {
  border: 1.5px solid #333 !important; background: #fff !important; color: #111 !important;
}
:focus-visible, input:focus, textarea:focus, select:focus,
button:focus, [role="button"]:focus {
  outline: 3px solid #000 !important; outline-offset: 2px !important;
}
button[kind="primary"], .stButton>button {
  background: #222 !important; color: #fff !important; border: 1.5px solid #000 !important; box-shadow: none !important;
}
button[kind="primary"]:hover, .stButton>button:hover { filter: brightness(1.2); }
[data-testid="stDataFrame"] thead tr th {
  background: #e8e8e8 !important; color: #111 !important; font-weight: 700 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: #fafafa !important; }
.small-font, .caption, .stCaption, figcaption { font-size: 0.98rem !important; color: #222 !important; }
a, a:visited { color: #000 !important; text-decoration: underline !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- 定数 --------------------
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
EXPORT_DPI = 200
SCATTER_WIDTH_PX = 480

PREFS = ["北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"]
PREF_SET = set(PREFS)

TOTAL_KEYWORDS = ["総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数",
    "金額","額","費用","支出","収入","販売額","生産額","生産量","面積","延べ","延","数"]
RATE_WORDS = ["率","割合","比率","％","パーセント","人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり","戸建て率"]
EXCLUDE_WORDS = ["順位","偏差値"]

# -------------------- セッション初期化 --------------------
if "url_a" not in st.session_state:
    st.session_state["url_a"] = ""
if "url_b" not in st.session_state:
    st.session_state["url_b"] = ""
if "display_df" not in st.session_state:
    st.session_state["display_df"] = None
if "calc" not in st.session_state:
    st.session_state["calc"] = None

# -------------------- ユーティリティ --------------------
def show_fig(fig, width_px: int):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)

def to_number(x) -> float:
    if not is_scalar(x):
        try:
            x = x.item()
        except Exception:
            return np.nan
    s = str(x).replace(",", "").replace("　", " ").replace("％", "%").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def iqr_mask(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=bool)
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (arr >= lo) & (arr <= hi)

def fmt(v: float) -> str:
    return "-" if (v is None or not np.isfinite(v)) else f"{v:.4f}"

def draw_scatter_reg_with_metrics(x, y, la, lb, title, width_px):
    fig, ax = plt.subplots(figsize=(BASE_W_INCH, BASE_H_INCH))
    ax.scatter(x, y, label="データ点", s=DEFAULT_MARKER_SIZE)
    r = r2 = None
    varx = float(np.nanstd(x)) if len(x) else 0.0
    vary = float(np.nanstd(y)) if len(y) else 0.0
    if len(x) >= 2:
        if varx > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, slope * xs + intercept, label="回帰直線", linewidth=DEFAULT_LINE_WIDTH)
        if varx > 0 and vary > 0:
            r = float(np.corrcoef(x, y)[0, 1]); r2 = r**2
    if r is not None and np.isfinite(r):
        ax.legend(loc="best", frameon=False, title=f"相関係数 r = {r:.3f}／決定係数 r2 = {r2:.3f}")
    else:
        ax.legend(loc="best", frameon=False)
    ax.set_xlabel(la if str(la).strip() else "横軸")
    ax.set_ylabel(lb if str(lb).strip() else "縦軸")
    ax.set_title(title if str(title).strip() else "散布図")
    show_fig(fig, width_px)
    st.caption(f"n = {len(x)}")
    st.caption(f"相関係数 r = {fmt(r)}")
    st.caption(f"決定係数 r2 = {fmt(r2)}")

# === 追加：散布図＋周辺箱ひげ図 ===
def draw_scatter_with_marginal_boxplots(x, y, la, lb, title, width_px):
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]
    y = np.asarray(y)[ok]
    if x.size == 0 or y.size == 0:
        st.warning("描画できるデータがありません。")
        return

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(BASE_W_INCH * 1.2, BASE_H_INCH * 1.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0.05, hspace=0.05)
    ax_box_y = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_box_x = fig.add_subplot(gs[1, 1])
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis("off")

    ax_main.scatter(x, y, s=DEFAULT_MARKER_SIZE)
    ax_main.set_xlabel(la)
    ax_main.set_ylabel(lb)
    ax_main.set_title(title)
    xlim, ylim = ax_main.get_xlim(), ax_main.get_ylim()

    ax_box_x.boxplot(x, vert=False, widths=0.6)
    ax_box_x.set_xlim(xlim)
    ax_box_y.boxplot(y, vert=True, widths=0.6)
    ax_box_y.set_ylim(ylim)
    ax_box_y.xaxis.set_visible(False)
    ax_box_x.yaxis.set_visible(False)
    show_fig(fig, width_px)

# --- 省略: load_todoran_table, safe_pearson など（元と同じ） ---
# ※ ここ以降の部分も前回と同様に続けてください
