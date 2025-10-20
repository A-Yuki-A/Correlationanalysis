# streamlit_app.py
# CorrGraph：とどランURL×2 → 都道府県データ抽出と相関分析（外れ値あり／なし散布図）
# - 結合後データの下に「外れ値を含む散布図＋箱ひげ図（左・下）」を表示
# - Y軸ラベルは散布図側のみ表示（箱ひげ側は削除）

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

# タイトルの余白
st.markdown("""
<style>
h1 { margin-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("CorrGraph")
st.write("とどランの **各ランキング記事のURL** を2つ貼り付けてください。")

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

# ページCSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  color: #111 !important;
  background: #f5f5f5 !important;
}
.block-container { max-width: 980px; padding-top: 1.2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #111 !important; letter-spacing: .01em; }
h1 { font-weight: 800; }
h2, h3 { font-weight: 700; }
p, li, .stMarkdown { line-height: 1.8; font-size: 1.02rem; }
button[kind="primary"], .stButton>button {
  background: #222 !important; color: #fff !important; border: 1.5px solid #000 !important;
}
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
for key in ["url_a", "url_b", "display_df", "calc"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "url" in key else None

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

# -------------------- グラフ描画関数 --------------------
def draw_scatter_reg_with_metrics(x, y, la, lb, title, width_px):
    fig, ax = plt.subplots(figsize=(BASE_W_INCH, BASE_H_INCH))
    ax.scatter(x, y, label="データ点", s=DEFAULT_MARKER_SIZE)
    r = r2 = None
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, slope * xs + intercept, label="回帰直線", linewidth=DEFAULT_LINE_WIDTH)
        r = float(np.corrcoef(x, y)[0, 1]); r2 = r**2
    if r is not None and np.isfinite(r):
        ax.legend(loc="best", frameon=False, title=f"相関係数 r={r:.3f}／r²={r2:.3f}")
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    ax.set_title(title)
    show_fig(fig, width_px)

# === 追加：散布図＋周辺箱ひげ図（Y軸ラベル重複修正） ===
def draw_scatter_with_marginal_boxplots(x, y, la, lb, title, width_px):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
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
    ax_main.set_ylabel(lb)  # ← 散布図側のみラベル
    ax_main.set_title(title)

    xlim, ylim = ax_main.get_xlim(), ax_main.get_ylim()

    ax_box_x.boxplot(x, vert=False, widths=0.6)
    ax_box_x.set_xlim(xlim)
    ax_box_x.yaxis.set_visible(False)
    ax_box_x.set_xlabel(la)

    ax_box_y.boxplot(y, vert=True, widths=0.6)
    ax_box_y.set_ylim(ylim)
    ax_box_y.xaxis.set_visible(False)
    ax_box_y.set_ylabel("")  # ← 箱ひげ側ラベルを消す

    show_fig(fig, width_px)

# ====== ここ以降：データ取得・UI・計算処理（省略せず） ======
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, allow_rate: bool = True):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []
    bs_tables = soup.find_all("table")
    for df in tables:
        cols = [str(c) for c in df.columns]
        pref_col = next((c for c in cols if "都道府県" in c or c in ("県名","道府県","府県")), None)
        if not pref_col: continue
        for c in cols:
            if any(w in c for w in EXCLUDE_WORDS): continue
        for c in cols:
            if c == pref_col: continue
            s = df[pref_col].astype(str).str.strip()
            v = df[c].astype(str).map(to_number)
            v = pd.to_numeric(v, errors="coerce")
            mask = s.isin(PREF_SET)
            if mask.sum() >= 30:
                out = pd.DataFrame({"pref": s[mask], "value": v[mask]}).dropna()
                out["pref"] = pd.Categorical(out["pref"], categories=PREFS, ordered=True)
                label = soup.find("h1").get_text(strip=True) if soup.find("h1") else c
                return out.sort_values("pref"), label
    return pd.DataFrame(columns=["pref","value"]), "データ"

# -------------------- UI --------------------
url_a = st.text_input("X軸URL", placeholder="https://todo-ran.com/t/kiji/XXXXX", key="url_a")
url_b = st.text_input("Y軸URL", placeholder="https://todo-ran.com/t/kiji/YYYYY", key="url_b")
allow_rate = st.checkbox("割合（％）を含める", value=True)

def clear_urls():
    for key in ["url_a","url_b","display_df","calc"]:
        st.session_state[key] = "" if "url" in key else None
    st.rerun()

col1, col2 = st.columns([2,1])
with col1:
    do_calc = st.button("相関を計算・表示する", type="primary")
with col2:
    st.button("クリア", on_click=clear_urls)

# -------------------- 計算 --------------------
if do_calc:
    if not url_a or not url_b:
        st.error("2つのURLを入力してください。")
        st.stop()
    df_a, label_a = load_todoran_table(url_a, allow_rate)
    df_b, label_b = load_todoran_table(url_b, allow_rate)
    if df_a.empty or df_b.empty:
        st.error("表の抽出に失敗しました。")
        st.stop()

    df = pd.merge(df_a.rename(columns={"value":"value_a"}),
                  df_b.rename(columns={"value":"value_b"}), on="pref")
    st.session_state["display_df"] = df.rename(columns={"value_a":label_a,"value_b":label_b})

    x = df["value_a"].to_numpy()
    y = df["value_b"].to_numpy()
    mask_x, mask_y = iqr_mask(x), iqr_mask(y)
    st.session_state["calc"] = {
        "x_all": x, "y_all": y,
        "x_in": x[mask_x & mask_y], "y_in": y[mask_x & mask_y],
        "label_a": label_a, "label_b": label_b
    }

# -------------------- 表示 --------------------
if st.session_state.get("display_df") is not None:
    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(st.session_state["display_df"], use_container_width=True, hide_index=True)
    st.download_button("CSVで保存", st.session_state["display_df"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="merged_pref_data.csv", mime="text/csv")

    # 外れ値を含む散布図＋箱ひげ図
    if st.session_state.get("calc") is not None:
        c = st.session_state["calc"]
        draw_scatter_with_marginal_boxplots(
            c["x_all"], c["y_all"],
            c["label_a"], c["label_b"],
            "散布図＋箱ひげ図（外れ値を含む）", 720
        )
