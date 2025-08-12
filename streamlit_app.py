# streamlit_app.py
# とどランURL×2 → 都道府県の「総数」データを抽出し、
# 外れ値あり／外れ値除外の散布図（横並び・回帰直線つき）を表示。
# 各散布図の直下に n・相関係数r・決定係数r2 を表示。
# 外れ値（都道府県名）は散布図の下に表示。CSV保存も可。
# 最下部に外れ値の定義（IQR法）を記載。
# ※ 修正点：率（割合）列も採用できるオプションを追加。ただし「偏差値」は除外。

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

# === フォント設定：同梱フォント最優先（なければシステム） ===
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
# ====== ここから追加：UIテーマ（グレースケール＆アクセシビリティ） ======
# Matplotlib をグレースケール化（凡例・線・点をモノトーンに）
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
# 点や線を少し太くして視認性UP（お好みで調整）
DEFAULT_MARKER_SIZE = 36
DEFAULT_LINE_WIDTH = 2.0

# draw_scatter_reg_with_metrics 内の scatter/plot に引数を渡せるように、関数の先頭に以下を追加
# ax.scatter(x, y, s=DEFAULT_MARKER_SIZE)
# ax.plot(xs, slope * xs + intercept, linewidth=DEFAULT_LINE_WIDTH)

# ページの中央寄せ・左右余白、文字のコントラスト、フォーカス枠などをCSSで調整
st.markdown("""
<style>
/* 画面全体の背景：薄いグレー、本文は濃いグレーで高コントラスト */
html, body, [data-testid="stAppViewContainer"] {
  color: #111 !important;
  background: #f5f5f5 !important;
}

/* メインコンテナを中央寄せ＋左右余白（max-width） */
.block-container {
  max-width: 980px;      /* ページの横幅上限＝左右に余白が生まれる */
  padding-top: 1.2rem;
  padding-bottom: 3rem;
}

/* 見出しの余白と太さを少し強調（モノトーン） */
h1, h2, h3 {
  color: #111 !important;
  letter-spacing: .01em;
}
h1 { font-weight: 800; }
h2 { font-weight: 700; }
h3 { font-weight: 700; }

/* 本文の行間を広げて読みやすく */
p, li, .stMarkdown {
  line-height: 1.8;
  font-size: 1.02rem;
}

/* 入力UIのコントラストとフォーカス可視化（キーボード操作に配慮） */
input, textarea, select, .stTextInput > div > div > input {
  border: 1.5px solid #333 !important;
  background: #fff !important;
  color: #111 !important;
}
:focus-visible, input:focus, textarea:focus, select:focus,
button:focus, [role="button"]:focus {
  outline: 3px solid #000 !important;   /* 黒いフォーカスリング */
  outline-offset: 2px !important;
}

/* ボタンをモノトーンで高コントラストに */
button[kind="primary"], .stButton>button {
  background: #222 !important;
  color: white !important;
  border: 1.5px solid #000 !important;
  box-shadow: none !important;
}
button[kind="primary"]:hover, .stButton>button:hover {
  filter: brightness(1.2);
}

/* テーブルの見やすさ（ヘッダ濃色・行の縞） */
[data-testid="stDataFrame"] thead tr th {
  background: #e8e8e8 !important;
  color: #111 !important;
  font-weight: 700 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
  background: #fafafa !important;
}

/* キャプションの文字サイズを少し上げる（読取りやすさ） */
.small-font, .caption, .stCaption, figcaption {
  font-size: 0.98rem !important;
  color: #222 !important;
}

/* リンクは下線付き（色弱にも判別しやすく） */
a, a:visited {
  color: #000 !important;
  text-decoration: underline !important;
}

/* 余白を少しゆったり */
section[data-testid="stSidebar"], .block-container {
  scroll-behavior: smooth;
}
</style>
""", unsafe_allow_html=True)
# ====== ここまで追加 ======


st.title("CorrGraph")
st.write("とどランの **各ランキング記事のURL** を2つ貼り付けてください。")

# -------------------- 表示サイズ --------------------
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
EXPORT_DPI = 200
SCATTER_WIDTH_PX = 480

def show_fig(fig, width_px: int):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)

# -------------------- 47都道府県 --------------------
PREFS = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"
]
PREF_SET = set(PREFS)

# -------------------- キーワード --------------------
TOTAL_KEYWORDS = [
    "総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数",
    "金額","額","費用","支出","収入","販売額","生産額","生産量","面積","延べ","延","数",
]
RATE_WORDS = ["率","割合","比率","％","パーセント","人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり","戸建て率"]
EXCLUDE_WORDS = ["順位","偏差値"]  # ← 偏差値は引き続き除外

# -------------------- ユーティリティ --------------------
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

def draw_scatter_reg_with_metrics(x: np.ndarray, y: np.ndarray, la: str, lb: str, title: str, width_px: int):
    fig, ax = plt.subplots(figsize=(BASE_W_INCH, BASE_H_INCH))
    ax.scatter(x, y, label="データ点")

    r = r2 = None
    varx = float(np.nanstd(x)) if len(x) else 0.0
    vary = float(np.nanstd(y)) if len(y) else 0.0
    if len(x) >= 2:
        if varx > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, slope * xs + intercept, label="回帰直線")
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

def flatten_columns(cols) -> list:
    """列名を平坦化し、空白や改行を除去して正規化（例：'総 数'→'総数'）。"""
    def _normalize(c: str) -> str:
        c = str(c).strip()
        c = re.sub(r"\s+", "", c)  # 半角/全角スペース・改行を詰める
        return c

    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x) for x in tup if pd.notna(x)]
            parts = [p for p in parts if not p.startswith("Unnamed")]
            name = " ".join(parts).strip()
            flat.append(name if name else "col")
        return [_normalize(c) for c in flat]
    return [_normalize(c) for c in cols]

def make_unique(seq: list) -> list:
    seen, out = {}, []
    for c in seq:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 1
            out.append(c)
    return out

def is_rank_like(nums: pd.Series) -> bool:
    s = pd.to_numeric(nums, errors="coerce").dropna()
    if s.empty:
        return False
    ints = (np.abs(s - np.round(s)) < 1e-9)
    share_int = float(ints.mean())
    in_range = float(((s >= 1) & (s <= 60)).mean())
    unique_close = (s.nunique() >= min(30, len(s)))
    return (share_int >= 0.8) and (in_range >= 0.9) and unique_close

def compose_label(caption: str | None, val_col: str | None, page_title: str | None) -> str:
    for s in (caption, page_title, val_col, "データ"):
        if s and str(s).strip():
            return str(s).strip()
    return "データ"

# -------------------- URL → (DataFrame, ラベル) --------------------
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, allow_rate: bool = True, version: int = 28):
    """allow_rate=True で率・割合列も候補に含める。偏差値/順位は除外のまま。"""
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

    def pick_value_dataframe(df: pd.DataFrame):
        df = df.copy()
        df.columns = make_unique(flatten_columns(df.columns))
        df = df.loc[:, ~df.columns.duplicated()]

        cols = list(df.columns)
        # 都道府県列の候補
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名","道府県","府県"))]
        if not pref_cols:
            return None, None

        def bad_name(name: str) -> bool:
            n = str(name)
            return any(w in n for w in EXCLUDE_WORDS)

        # 「順位」「偏差値」などを除いた値候補（率を含むかは後で分岐）
        raw_value_candidates = [
            c for c in cols
            if (c not in ("順位","都道府県","道府県","県名","府県")) and (not bad_name(c))
        ]

        # 総数キーワードにヒットする列（優先）
        total_name_candidates = [c for c in raw_value_candidates if any(k in c for k in TOTAL_KEYWORDS)]

        # フォールバック候補：allow_rate で分岐
        if allow_rate:
            fallback_candidates = raw_value_candidates[:]   # 率・割合も許可
        else:
            fallback_candidates = [c for c in raw_value_candidates if not any(rw in c for rw in RATE_WORDS)]

        def score_and_build(pref_col: str, candidate_cols: list):
            best_score, best_df, best_vc = -1, None, None
            pref_series = df[pref_col]
            if isinstance(pref_series, pd.DataFrame):
                pref_series = pref_series.iloc[:, 0]
            pref_series = pref_series.map(lambda x: str(x).strip())
            mask = pref_series.isin(PREF_SET).to_numpy()
            if not mask.any():
                return None, None
            for vc in candidate_cols:
                if vc not in df.columns:
                    continue
                col = df[vc]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                col_num = pd.to_numeric(col.map(to_number), errors="coerce").loc[mask]
                # 順位っぽい列は除外
                if is_rank_like(col_num):
                    continue
                base = int(col_num.notna().sum())
                bonus = 15 if any(k in vc for k in TOTAL_KEYWORDS) else 0
                score = base + bonus
                if score > best_score and base >= 30:
                    tmp = pd.DataFrame({"pref": pref_series.loc[mask].values, "value": col_num.values})
                    tmp = tmp.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
                    best_score, best_df, best_vc = score, tmp, vc
            return best_df, best_vc

        # まず総数優先で探索
        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, total_name_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                return got.sort_values("pref").reset_index(drop=True), val_col
        # 見つからなければ（allow_rate に応じて）フォールバック
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
                if cap:
                    caption_text = cap.get_text(strip=True)
            label = compose_label(caption_text, val_col, page_h1 or page_title)
            return got, label

    # フォールバック（ページ全文から簡易抽出）
    lines = []
    for tag in soup.find_all(text=True):
        t = str(tag).strip()
        if t:
            lines.append(t)
    text = "\n".join(lines)
    rows = []
    for line in text.splitlines():
        m = re.search(r"(北海道|..県|..府|東京都)\s+(-?\d+(?:\.\d+)?)", line)
        if m:
            pref = m.group(1); val = float(m.group(2))
            if pref in PREF_SET:
                rows.append((pref, val))
    if rows:
        work = pd.DataFrame(rows, columns=["pref","value"]).drop_duplicates("pref")
        work["pref"] = pd.Categorical(work["pref"], categories=PREFS, ordered=True)
        work = work.sort_values("pref").reset_index(drop=True)
        label = compose_label(None, None, page_h1 or page_title)
        return work, label

    return pd.DataFrame(columns=["pref","value"]), "データ"

# -------------------- UI（縦並びフォーム） --------------------
url_a = st.text_input(
    "X軸（説明変数）URL ＝ 原因・条件の指標",
    placeholder="https://todo-ran.com/t/kiji/XXXXX",
    help="総数の指標を優先して使います（順位・偏差値は不可）"
)
url_b = st.text_input(
    "Y軸（目的変数）URL ＝ 結果・反応の指標",
    placeholder="https://todo-ran.com/t/kiji/YYYYY",
    help="総数の指標を優先して使います（順位・偏差値は不可）"
)
# 率・割合も許可するか（既定：許可）
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
        st.error("表の抽出に失敗しました。URLがランキング記事であること、表に『都道府県』列があることを確認してください。"); st.stop()

    # 共通都道府県で結合
    df = pd.merge(
        df_a.rename(columns={"value":"value_a"}),
        df_b.rename(columns={"value":"value_b"}),
        on="pref", how="inner",
    )

    # 表示用
    display_df = df.rename(columns={"value_a": label_a, "value_b": label_b})
    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(df) < 3:
        st.warning("共通データが少ないため、相関係数が不安定です。別の指標でお試しください。"); st.stop()

    # 数値配列（NaN除去）
    x0 = pd.to_numeric(df["value_a"], errors="coerce")
    y0 = pd.to_numeric(df["value_b"], errors="coerce")
    mask0 = x0.notna() & y0.notna()
    x_all = x0[mask0].to_numpy()
    y_all = y0[mask0].to_numpy()
    pref_all = df.loc[mask0, "pref"].astype(str).to_numpy()

    # 外れ値除外（x または y のどちらかが外れ値なら除外）
    mask_x_in = iqr_mask(x_all, 1.5)
    mask_y_in = iqr_mask(y_all, 1.5)
    mask_inlier = mask_x_in & mask_y_in
    x_in = x_all[mask_inlier]
    y_in = y_all[mask_inlier]

    # 散布図：外れ値あり／除外（横並び）
    st.subheader("散布図（左：外れ値を含む／右：外れ値除外）")
    col_l, col_r = st.columns(2)
    with col_l:
        draw_scatter_reg_with_metrics(x_all, y_all, label_a, label_b, "散布図（外れ値を含む）", SCATTER_WIDTH_PX)
    with col_r:
        draw_scatter_reg_with_metrics(x_in,  y_in,  label_a, label_b, "散布図（外れ値除外）", SCATTER_WIDTH_PX)

    # 外れ値（都道府県名）表示
    # --- ここから変更 ---
st.subheader("外れ値（都道府県名）")

outs_x = pref_all[~mask_x_in]
outs_y = pref_all[~mask_y_in]
outs_any = pref_all[~mask_inlier]

st.markdown("**X軸で外れ値**")
st.write("\n".join(map(str, outs_x)) if len(outs_x) else "なし")

st.markdown("**Y軸で外れ値**")
st.write("\n".join(map(str, outs_y)) if len(outs_y) else "なし")

st.markdown("**除外対象（XまたはY）**")
st.write("\n".join(map(str, outs_any)) if len(outs_any) else "なし")
# --- ここまで変更 ---


    # ページ末尾に外れ値の定義
    st.markdown("---")
    st.markdown(
        "#### 外れ値の定義\n"
        "本ツールでは **IQR法** を用いて外れ値を判定しています。四分位範囲 IQR = Q3 − Q1 とし、\n"
        "**下限 = Q1 − 1.5×IQR、上限 = Q3 + 1.5×IQR** を超える値を外れ値とします。\n"
        "散布図では、**x または y のどちらかが外れ値**に該当する都道府県を除外して「外れ値除外」図を作成しています。"
    )
else:
    st.info("上の2つの入力欄に とどラン記事のURL を貼ってから「相関を計算・表示する」を押してください。")
