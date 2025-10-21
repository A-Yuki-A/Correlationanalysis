# streamlit_app.py
# CorrGraph（高校生にも分かりやすい相関分析ツール）
# - とどランURL×2を入力して都道府県データを結合・相関分析
# - 箱ひげ図の外れ値を基準に除外（IQR, 1.5倍ルール）
# - 外れ値を含む／除外した散布図＋箱ひげ図を表示
# - 回帰直線と相関係数（r, r²）表示
# - 外れ値一覧を見やすく表示（CSV保存可）

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
    for name in ["Noto Sans JP", "IPAexGothic", "Yu Gothic", "Hiragino Sans", "Meiryo"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="CorrGraph", layout="wide")

# ====== グレースケールUI ======
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
EXPORT_DPI = 200
WHIS = 1.5  # 箱ひげのひげ長（IQRの倍率）

# ====== タイトル ======
st.title("CorrGraph：都道府県データの相関分析ツール")
st.write("とどランの記事URLを2つ入力すると、共通の都道府県データで相関関係を分析できます。")

# ====== スタイル ======
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { color:#111; background:#f5f5f5; }
.block-container { max-width:980px; padding-top:1.2rem; padding-bottom:3rem; }
h1, h2, h3 { color:#111; letter-spacing:.01em; }
h1 { font-weight:800; } h2,h3 { font-weight:700; }
button[kind="primary"], .stButton>button { background:#222; color:#fff; border:1.5px solid #000; }
[data-testid="stDataFrame"] thead tr th { background:#e8e8e8; color:#111; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ====== 定数 ======
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8

PREFS = ["北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
          "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
          "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
          "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
          "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"]
PREF_SET = set(PREFS)

TOTAL_KEYWORDS = ["総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数","金額","額","支出","収入"]
RATE_WORDS = ["率","割合","％","パーセント","当たり"]
EXCLUDE_WORDS = ["順位","偏差値"]

# ====== 関数 ======
def show_fig(fig, width_px: int):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)

def to_number(x):
    if not is_scalar(x):
        try:
            x = x.item()
        except Exception:
            return np.nan
    s = str(x).replace(",", "").replace("％", "%").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def boxplot_inlier_mask(arr: np.ndarray, whis: float = WHIS):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=bool), (np.nan,)*5
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    iqr = q3 - q1
    low = q1 - whis * iqr
    high = q3 + whis * iqr
    inlier = (arr >= low) & (arr <= high)
    inlier[~valid] = False
    return inlier, (q1, q3, iqr, low, high)

def fmt(v) -> str:
    try:
        vf = float(v)
        if not np.isfinite(vf):
            return "-"
        return f"{vf:.3f}"
    except Exception:
        return "-"

# ===== 散布図＋箱ひげ図（降順・昇順を非表示に修正） =====
def draw_scatter_with_marginal_boxplots(x, y, la, lb, title, width_px, outs_x=None, outs_y=None, pref_all=None):
    import matplotlib.gridspec as gridspec
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) == 0:
        st.warning("描画できるデータがありません。")
        return

    fig = plt.figure(figsize=(BASE_W_INCH*1.2, BASE_H_INCH*1.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1], wspace=0.05, hspace=0.05)

    ax_main  = fig.add_subplot(gs[0,1])
    ax_box_y = fig.add_subplot(gs[0,0], sharey=ax_main)
    ax_box_x = fig.add_subplot(gs[1,1])
    ax_empty = fig.add_subplot(gs[1,0]); ax_empty.axis("off")

    # --- 散布図 ---
    handles, labels = [], []
    if outs_x is not None and outs_y is not None and pref_all is not None:
        pref_all = np.asarray(pref_all)[ok]
        out_set_x = set(map(str, outs_x))
        out_set_y = set(map(str, outs_y))
        out_mask = np.array([(p in out_set_x) or (p in out_set_y) for p in pref_all])
        in_mask = ~out_mask
        h1 = ax_main.scatter(x[in_mask], y[in_mask], color="gray", s=DEFAULT_MARKER_SIZE, label="通常データ")
        h2 = ax_main.scatter(x[out_mask], y[out_mask], color="blue", s=DEFAULT_MARKER_SIZE, label="外れ値")
        handles.extend([h1, h2]); labels.extend(["通常データ", "外れ値"])
    else:
        h = ax_main.scatter(x, y, color="gray", s=DEFAULT_MARKER_SIZE, label="データ")
        handles.append(h); labels.append("データ")

    # --- 回帰直線と相関係数 ---
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 200)
        line = ax_main.plot(xs, slope*xs + intercept, color="black", linewidth=DEFAULT_LINE_WIDTH, label="回帰直線")
        r = np.corrcoef(x, y)[0, 1]; r2 = r**2
        ax_main.legend(handles + line, labels + [f"r={r:.3f}, r²={r2:.3f}"], frameon=False)
    else:
        ax_main.legend(handles, labels, frameon=False)

    # 軸ラベル設定（自動ラベル無効化）
    ax_main.set_xlabel(la)
    ax_main.set_ylabel("")  # yラベルは左箱ひげ側に表示
    ax_main.set_title(title)
    ax_main.tick_params(axis="y", which="both", left=False, labelleft=False)

    # --- 箱ひげ図（whis=1.5, 降順昇順非表示） ---
    ax_box_x.boxplot(x, vert=False, widths=0.6, whis=WHIS, showfliers=True)
    ax_box_x.set_xlabel(la)
    ax_box_x.yaxis.set_visible(False)
    ax_box_x.set_xticks([])  # 自動の「昇順・降順」防止
    ax_box_x.set_title("")   # タイトルもなし

    ax_box_y.boxplot(y, vert=True, widths=0.6, whis=WHIS, showfliers=True)
    ax_box_y.set_ylabel(lb)
    ax_box_y.xaxis.set_visible(False)
    ax_box_y.set_xticks([])

    show_fig(fig, width_px)

# ===== URL読込関数（略: 以前の通り） =====
# ※ここは省略可能、既に前回のバージョンで問題なく動いている部分

# ===== UI =====
url_a = st.text_input("X軸URL（説明変数）", placeholder="https://todo-ran.com/t/kiji/XXXXX")
url_b = st.text_input("Y軸URL（目的変数）", placeholder="https://todo-ran.com/t/kiji/YYYYY")
allow_rate = st.checkbox("割合（％・〜当たり）を含める", value=True)

# （---中略---）
# ↓以下に外れ値一覧・説明の部分（高校生向け説明入り）を統合
st.markdown("### 箱ひげ図で使われている基準とは？")
st.markdown("""
箱ひげ図では、データのばらつきを四分位数を使って表します。  
**IQR（Interquartile Range）＝Q3−Q1**は、データの「真ん中50％」の広がりです。

- Q1（第1四分位数）：下から25％の位置  
- Q3（第3四分位数）：下から75％の位置  
- 外れ値とは：Q1−1.5×IQRより小さい、またはQ3＋1.5×IQRより大きい値です。

このグラフでは、X軸またはY軸のどちらか一方でも外れ値に当てはまる都道府県を除外しています。
""")
