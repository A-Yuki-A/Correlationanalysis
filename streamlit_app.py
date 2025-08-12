# streamlit_app.py
# とどランURL×2 → 都道府県データ抽出、相関分析（外れ値あり／なし散布図）
# ・割合列も許可（オプション）
# ・「偏差値」や「順位」列は除外
# ・外れ値は「X軸で外れ値」「Y軸で外れ値」のみ横並び2カラム表示
# ・グレースケールデザイン／中央寄せ／アクセシビリティ配慮／タイトル余白修正

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

# -------------------- ユーティリティ --------------------
def show_fig(fig, width_px: int):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)

def to_number(x) -> float:
    if not is_scalar(x):
        try: x = x.item()
        except Exception: return np.nan
    s = str(x).replace(",", "").replace("　", " ").replace("％", "%").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m: return np.nan
    try: return float(m.group(0))
    except Exception: return np.nan

def iqr_mask(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    if arr.size == 0: return np.array([], dtype=bool)
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

def flatten_columns(cols):
    def _normalize(c: str) -> str:
        return re.sub(r"\s+", "", str(c).strip())
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x) for x in tup if pd.notna(x)]
            parts = [p for p in parts if not p.startswith("Unnamed")]
            name = " ".join(parts).strip()
            flat.append(name if name else "col")
        return [_normalize(c) for c in flat]
    return [_normalize(c) for c in cols]

def make_unique(seq):
    seen, out = {}, []
    for c in seq:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 1
            out.append(c)
    return out

def is_rank_like(nums):
    s = pd.to_numeric(nums, errors="coerce").dropna()
    if s.empty: return False
    ints = (np.abs(s - np.round(s)) < 1e-9)
    share_int = float(ints.mean())
    in_range = float(((s >= 1) & (s <= 60)).mean())
    unique_close = (s.nunique() >= min(30, len(s)))
    return (share_int >= 0.8) and (in_range >= 0.9) and unique_close

def compose_label(caption, val_col, page_title):
    for s in (caption, page_title, val_col, "データ"):
        if s and str(s).strip():
            return str(s).strip()
    return "データ"

# -------------------- URL読み込み --------------------
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, allow_rate: bool = True):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    page_h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else None
    page_title = soup.title.get_text(strip=True) if soup.title else None
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html, flavor="bs4")
        except Exception:
            tables = []
    bs_tables = soup.find_all("table")
    def pick_value_dataframe(df):
        df = df.copy()
        df.columns = make_unique(flatten_columns(df.columns))
        df = df.loc[:, ~df.columns.duplicated()]
        cols = list(df.columns)
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名","道府県","府県"))]
        if not pref_cols: return None, None
        def bad_name(name: str) -> bool:
            return any(w in str(name) for w in EXCLUDE_WORDS)
        raw_value_candidates = [c for c in cols if (c not in ("順位","都道府県","道府県","県名","府県")) and (not bad_name(c))]
        total_name_candidates = [c for c in raw_value_candidates if any(k in c for k in TOTAL_KEYWORDS)]
        if allow_rate:
            fallback_candidates = raw_value_candidates[:]
        else:
            fallback_candidates = [c for c in raw_value_candidates if not any(rw in c for rw in RATE_WORDS)]
        def score_and_build(pref_col, candidate_cols):
            best_score, best_df, best_vc = -1, None, None
            pref_series = df[pref_col]
            if isinstance(pref_series, pd.DataFrame):
                pref_series = pref_series.iloc[:, 0]
            pref_series = pref_series.map(lambda x: str(x).strip())
            mask = pref_series.isin(PREF_SET).to_numpy()
            if not mask.any(): return None, None
            for vc in candidate_cols:
                if vc not in df.columns: continue
                col = df[vc]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                col_num = pd.to_numeric(col.map(to_number), errors="coerce").loc[mask]
                if is_rank_like(col_num): continue
                base = int(col_num.notna().sum())
                bonus = 15 if any(k in vc for k in TOTAL_KEYWORDS) else 0
                score = base + bonus
                if score > best_score and base >= 30:
                    tmp = pd.DataFrame({"pref": pref_series.loc[mask].values, "value": col_num.values})
                    tmp = tmp.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
                    best_score, best_df, best_vc = score, tmp, vc
            return best_df, best_vc
        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, total_name_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                return got.sort_values("pref").reset_index(drop=True), val_col
        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, fallback_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                return got.sort_values("pref").reset_index(drop=True), val_col
        return None, None
    for idx, raw in enumerate(tables):
        got, val_col = pick_value_dataframe(raw)
        if got is not None:
            caption_text = None
            if idx < len(bs_tables):
                cap = bs_tables[idx].find("caption")
                if cap: caption_text = cap.get_text(strip=True)
            label = compose_label(caption_text, val_col, page_h1 or page_title)
            return got, label
    return pd.DataFrame(columns=["pref","value"]), "データ"

# -------------------- UI --------------------
url_a = st.text_input("X軸（説明変数）URL", placeholder="https://todo-ran.com/t/kiji/XXXXX")
url_b = st.text_input("Y軸（目的変数）URL", placeholder="https://todo-ran.com/t/kiji/YYYYY")
allow_rate = st.checkbox("割合（率・％・当たり）も対象にする", value=True)

# -------------------- メイン処理 --------------------
if st.button("相関を計算・表示する", type="primary"):
    if not url_a or not url_b:
        st.error("2つのURLを入力してください。"); st.stop()
    try:
        df_a, label_a = load_todoran_table(url_a, allow_rate=allow_rate)
        df_b, label_b = load_todoran_table(url_b, allow_rate=allow_rate)
    except requests.RequestException as e:
        st.error(f"ページの取得に失敗しました：{e}"); st.stop()
    if df_a.empty or df_b.empty:
        st.error("表の抽出に失敗しました。"); st.stop()
    df = pd.merge(df_a.rename(columns={"value":"value_a"}), df_b.rename(columns={"value":"value_b"}), on="pref", how="inner")
    display_df = df.rename(columns={"value_a": label_a, "value_b": label_b})
    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    if len(df) < 3:
        st.warning("共通データが少ないため、相関係数が不安定です。"); st.stop()
    x0 = pd.to_numeric(df["value_a"], errors="coerce")
    y0 = pd.to_numeric(df["value_b"], errors="coerce")
    mask0 = x0.notna() & y0.notna()
    x_all = x0[mask0].to_numpy()
    y_all = y0[mask0].to_numpy()
    pref_all = df.loc[mask0, "pref"].astype(str).to_numpy()
    mask_x_in = iqr_mask(x_all, 1.5)
    mask_y_in = iqr_mask(y_all, 1.5)
    mask_inlier = mask_x_in & mask_y_in
    x_in = x_all[mask_inlier]
    y_in = y_all[mask_inlier]
       st.subheader("散布図（左：外れ値を含む／右：外れ値除外）")
    col_l, col_r = st.columns(2)
    with col_l:
        draw_scatter_reg_with_metrics(x_all, y_all, label_a, label_b, "散布図（外れ値を含む）", SCATTER_WIDTH_PX)
    with col_r:
        draw_scatter_reg_with_metrics(x_in, y_in, label_a, label_b, "散布図（外れ値除外）", SCATTER_WIDTH_PX)

