# streamlit_app.py
# CorrGraph：とどランURL×2 → 都道府県データ抽出と相関分析
# - 結合後データの下に「散布図＋箱ひげ図（外れ値を含む）」と
#   その下に「散布図＋箱ひげ図（外れ値除外）」を表示
# - Y軸ラベル＆目盛は箱ひげ図（左）に集約、散布図側は非表示
# - 一番下に「外れ値として処理した都道府県一覧（X軸／Y軸）」と「外れ値の定義（IQR）」を表示

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

# タイトル余白
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

# ページCSS（簡略）
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

# === 散布図＋周辺箱ひげ図（Y軸は箱ひげ側に集約、散布図側のY目盛を非表示） ===
def draw_scatter_with_marginal_boxplots(x, y, la, lb, title, width_px):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if x.size == 0 or y.size == 0:
        st.warning("描画できるデータがありません。")
        return

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(BASE_W_INCH * 1.2, BASE_H_INCH * 1.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0.05, hspace=0.05)

    # 先に散布図を作成 → 左の箱ひげは sharey にする
    ax_main  = fig.add_subplot(gs[0, 1])
    ax_box_y = fig.add_subplot(gs[0, 0], sharey=ax_main)  # Y軸共有
    ax_box_x = fig.add_subplot(gs[1, 1])
    ax_empty = fig.add_subplot(gs[1, 0]); ax_empty.axis("off")

    # 散布図
    ax_main.scatter(x, y, s=DEFAULT_MARKER_SIZE)
    ax_main.set_xlabel(la)
    ax_main.set_title(title)
    ax_main.set_ylabel("")  # Yラベル消す（箱ひげ側に集約）
    ax_main.tick_params(axis="y", which="both", left=False, labelleft=False)

    # 箱ひげ：下（X）
    xlim = ax_main.get_xlim()
    ax_box_x.boxplot(x, vert=False, widths=0.6)
    ax_box_x.set_xlim(xlim)
    ax_box_x.yaxis.set_visible(False)
    ax_box_x.set_xlabel(la)

    # 箱ひげ：左（Y）※共有Y軸なので目盛はここにだけ表示
    ax_box_y.boxplot(y, vert=True, widths=0.6)
    ax_box_y.xaxis.set_visible(False)
    ax_box_y.set_ylabel(lb)  # Y軸ラベルはここだけ

    show_fig(fig, width_px)

# ===== 相関ユーティリティ =====
def strength_label(r: float) -> str:
    if r is None or not np.isfinite(r): return "判定不可"
    a = abs(r)
    if a >= 0.7: return "強い"
    if a >= 0.4: return "中程度"
    if a >= 0.2: return "弱い"
    return "ほとんどない"

def safe_pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])

def safe_spearman(x, y):
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return safe_pearson(xr, yr)

# -------------------- URL読み込み --------------------
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, allow_rate: bool = True):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []
    bs_tables = soup.find_all("table")

    def flatten_columns(cols):
        def _normalize(c: str) -> str:
            return re.sub(r"\s+", "", str(c).strip())
        if isinstance(cols, pd.MultiIndex):
            flat = []
            for tup in cols:
                parts = [str(x) for x in tup if pd.notna(x)]
                parts = [p for p in parts if not str(p).startswith("Unnamed")]
                nm = " ".join(parts).strip()
                flat.append(nm if nm else "col")
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
        for s in (caption, val_col, page_title, "データ"):
            if s and str(s).strip():
                return str(s).strip()
        return "データ"

    def pick_value_dataframe(df):
        df = df.copy()
        df.columns = make_unique(flatten_columns(df.columns))
        df = df.loc[:, ~df.columns.duplicated()]
        cols = list(df.columns)
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名","道府県","府県"))]
        if not pref_cols:
            return None, None

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
            if not mask.any():
                return None, None
            for vc in candidate_cols:
                if vc not in df.columns: continue
                col = df[vc]
                if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
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

    page_h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else None
    page_title = soup.title.get_text(strip=True) if soup.title else None
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
url_a = st.text_input("X軸URL（説明変数）", placeholder="https://todo-ran.com/t/kiji/XXXXX", key="url_a")
url_b = st.text_input("Y軸URL（目的変数）", placeholder="https://todo-ran.com/t/kiji/YYYYY", key="url_b")
allow_rate = st.checkbox("割合（％・〜当たり）を含める", value=True)

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
    try:
        df_a, label_a = load_todoran_table(url_a, allow_rate)
        df_b, label_b = load_todoran_table(url_b, allow_rate)
    except requests.RequestException as e:
        st.error(f"ページの取得に失敗しました：{e}")
        st.stop()
    if df_a.empty or df_b.empty:
        st.error("表の抽出に失敗しました。")
        st.stop()

    df = pd.merge(
        df_a.rename(columns={"value":"value_a"}),
        df_b.rename(columns={"value":"value_b"}),
        on="pref", how="inner"
    )

    st.session_state["display_df"] = df.rename(columns={"value_a":label_a,"value_b":label_b})

    # 数値化＆外れ値判定用ベクトル
    x0 = pd.to_numeric(df["value_a"], errors="coerce")
    y0 = pd.to_numeric(df["value_b"], errors="coerce")
    mask0 = x0.notna() & y0.notna()

    x_all = x0[mask0].to_numpy()
    y_all = y0[mask0].to_numpy()
    pref_all = df.loc[mask0, "pref"].astype(str).to_numpy()

    # IQR外れ値（軸ごと）
    mask_x_in = iqr_mask(x_all, 1.5)
    mask_y_in = iqr_mask(y_all, 1.5)
    mask_in = mask_x_in & mask_y_in

    # 外れ値リスト（都道府県名）
    outs_x = pref_all[~mask_x_in]
    outs_y = pref_all[~mask_y_in]

    # 参考：IQR境界値（全データベース）
    def iqr_bounds(arr):
        q1 = float(np.nanpercentile(arr, 25))
        q3 = float(np.nanpercentile(arr, 75))
        iqr = q3 - q1
        return q1, q3, iqr, q1 - 1.5*iqr, q3 + 1.5*iqr

    q1x, q3x, iqrx, lox, hix = iqr_bounds(x_all) if len(x_all) else (np.nan,)*5
    q1y, q3y, iqry, loy, hiy = iqr_bounds(y_all) if len(y_all) else (np.nan,)*5

    st.session_state["calc"] = {
        "x_all": x_all, "y_all": y_all,
        "x_in": x_all[mask_in], "y_in": y_all[mask_in],
        "label_a": label_a, "label_b": label_b,
        "outs_x": outs_x, "outs_y": outs_y,
        "iqr_info": {
            "x": {"Q1": q1x, "Q3": q3x, "IQR": iqrx, "LOW": lox, "HIGH": hix},
            "y": {"Q1": q1y, "Q3": q3y, "IQR": iqry, "LOW": loy, "HIGH": hiy},
        }
    }

# -------------------- 表示 --------------------
if st.session_state.get("display_df") is not None:
    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(st.session_state["display_df"], use_container_width=True, hide_index=True)
    st.download_button("CSVで保存",
        st.session_state["display_df"].to_csv(index=False).encode("utf-8-sig"),
        file_name="merged_pref_data.csv", mime="text/csv"
    )

    # 1) 外れ値を含む散布図＋箱ひげ図
    if st.session_state.get("calc") is not None:
        c = st.session_state["calc"]
        draw_scatter_with_marginal_boxplots(
            c["x_all"], c["y_all"], c["label_a"], c["label_b"],
            "散布図＋箱ひげ図（外れ値を含む）", width_px=720
        )

        # 2) 外れ値除外の散布図＋箱ひげ図
        if len(c["x_in"]) >= 2 and len(c["y_in"]) >= 2:
            draw_scatter_with_marginal_boxplots(
                c["x_in"], c["y_in"], c["label_a"], c["label_b"],
                "散布図＋箱ひげ図（外れ値除外）", width_px=720
            )
        else:
            st.info("外れ値除外後のデータ数が少ないため、除外版の図は省略しました。")

       # ---- 一番下：外れ値一覧と外れ値の定義（IQR） ----
st.markdown("---")
st.subheader("外れ値として処理した都道府県（一覧）")

colx, coly = st.columns(2)
with colx:
    st.markdown("**X軸で外れ値**")
    st.write("\n".join(map(str, c["outs_x"])) if len(c["outs_x"]) else "なし")
with coly:
    st.markdown("**Y軸で外れ値**")
    st.write("\n".join(map(str, c["outs_y"])) if len(c["outs_y"]) else "なし")

# ▼▼▼ ここから安全化：iqr_info が無い場合は再計算 ▼▼▼
iqr_info = c.get("iqr_info")

def _iqr_bounds(arr: np.ndarray):
    if arr is None or len(arr) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    q1 = float(np.nanpercentile(arr, 25))
    q3 = float(np.nanpercentile(arr, 75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return (q1, q3, iqr, low, high)

if not iqr_info:
    q1x, q3x, iqrx, lox, hix = _iqr_bounds(c["x_all"])
    q1y, q3y, iqry, loy, hiy = _iqr_bounds(c["y_all"])
    xi = {"Q1": q1x, "Q3": q3x, "IQR": iqrx, "LOW": lox, "HIGH": hix}
    yi = {"Q1": q1y, "Q3": q3y, "IQR": iqry, "LOW": loy, "HIGH": hiy}
else:
    xi = iqr_info.get("x", {})
    yi = iqr_info.get("y", {})

# fmt() はあなたのコードに既にある補助関数（None/NaN安全）
st.caption(
    "X軸 IQR基準: "
    f"Q1={fmt(xi.get('Q1'))}, Q3={fmt(xi.get('Q3'))}, IQR={fmt(xi.get('IQR'))}, "
    f"下限={fmt(xi.get('LOW'))}, 上限={fmt(xi.get('HIGH'))} / "
    "Y軸 IQR基準: "
    f"Q1={fmt(yi.get('Q1'))}, Q3={fmt(yi.get('Q3'))}, IQR={fmt(yi.get('IQR'))}, "
    f"下限={fmt(yi.get('LOW'))}, 上限={fmt(yi.get('HIGH'))}"
)

st.markdown("#### 外れ値の定義（IQR法）")
st.markdown(
    "- 四分位範囲（**IQR**）を **IQR = Q3 − Q1** とします。  \n"
    "- **下限 = Q1 − 1.5×IQR**, **上限 = Q3 + 1.5×IQR** を超えるデータを**外れ値**と判定しました。  \n"
    "- 本ツールでは、X軸またはY軸の**どちらか一方でも外れ値**になった都道府県を、外れ値として除外しています。"
)
