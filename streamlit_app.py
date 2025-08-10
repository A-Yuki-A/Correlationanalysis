# streamlit_app.py
# とどランのランキング記事URLを2つ貼り付けて、
# 「都道府県 × 実数値（偏差値や順位は除外）」を自動抽出。
# 表タイトル（<caption>/<h1> 等）をラベルに反映し、
# 相関係数・決定係数・散布図（回帰直線つき）・箱ひげ図を表示。
# 散布図は幅640px（前より2倍）、箱ひげ図は左右に横並び。
# Matplotlibは japanize-matplotlib で日本語化（無い環境はフォールバック）。

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

# --- 日本語化（japanize があれば使う） ---
try:
    import japanize_matplotlib  # noqa: F401
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    # フォールバック：手元にある日本語フォントを自動検出
    def _set_jp_font_fallback():
        candidates = [
            "Yu Gothic", "Yu Gothic UI", "Noto Sans CJK JP", "Noto Sans JP",
            "IPAexGothic", "IPAGothic", "Hiragino Sans", "Meiryo",
        ]
        for name in candidates:
            try:
                prop = fm.FontProperties(family=name)
                fm.findfont(prop, fallback_to_default=False)
                plt.rcParams["font.family"] = name
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False
    _set_jp_font_fallback()

st.set_page_config(page_title="都道府県データ 相関ツール（URL版）", layout="wide")
st.title("都道府県データ 相関ツール（URL版）")
st.write(
    "とどランの **各ランキング記事のURL** を2つ貼り付けてください。"
    "表の「偏差値」「順位」は使わず、**総数（件数・人数・金額などの実数値）**を自動抽出し、"
    "ページ内の **表タイトル** をグラフのラベルに反映します。"
)

# -------------------- 画像表示サイズ --------------------
# Matplotlib 既定換算: 6.4inch * 100dpi = 640px
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
EXPORT_DPI = 200                 # PNG保存時のDPI（高精細）
SCATTER_WIDTH_PX = 640           # 散布図は前の2倍（640px）
BOX_WIDTH_PX = 320               # 箱ひげ図は左右に横並び用（各320px）

def show_fig(fig, width_px: int):
    """figをPNGにして、指定px幅で確実に表示。"""
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

# -------------------- 総数系／除外系キーワード --------------------
TOTAL_KEYWORDS = [
    "総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数",
    "金額","額","費用","支出","収入","販売額","生産額","生産量","面積","延べ","延","数",
]
RATE_WORDS = [
    "率","割合","比率","％","パーセント",
    "人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり"
]
EXCLUDE_WORDS = ["順位","偏差値"]

# -------------------- ユーティリティ --------------------
def to_number(x) -> float:
    """文字列から数値（小数含む）を抜き出して float 化。配列/Seriesが来ても安全。"""
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

def draw_scatter_with_reg(df: pd.DataFrame, la: str, lb: str):
    """散布図＋回帰直線（日本語ラベル）。"""
    x = pd.to_numeric(df["value_a"], errors="coerce")
    y = pd.to_numeric(df["value_b"], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()

    fig, ax = plt.subplots(figsize=(BASE_W_INCH, BASE_H_INCH))
    ax.scatter(x, y)

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, slope * xs + intercept, label=f"回帰直線: y = {slope:.3g}x + {intercept:.3g}")
        r = np.corrcoef(x, y)[0, 1]
        ax.legend(loc="best", frameon=False, title=f"r = {r:.3f}, R² = {r**2:.3f}")

    ax.set_xlabel(la)  # 日本語
    ax.set_ylabel(lb)  # 日本語
    ax.set_title("散布図（回帰直線つき）")
    show_fig(fig, SCATTER_WIDTH_PX)

def draw_boxplot(series: pd.Series, label: str):
    """箱ひげ図（日本語ラベル）。"""
    fig, ax = plt.subplots(figsize=(BASE_W_INCH, BASE_H_INCH))
    ax.boxplot(pd.to_numeric(series, errors="coerce").dropna(), vert=True)
    ax.set_title(f"箱ひげ図：{label}")
    ax.set_ylabel("値")
    ax.set_xticks([1])
    ax.set_xticklabels([label])  # 日本語ラベル
    show_fig(fig, BOX_WIDTH_PX)

def flatten_columns(cols) -> list:
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x) for x in tup if pd.notna(x)]
            parts = [p for p in parts if not p.startswith("Unnamed")]
            name = " ".join(parts).strip()
            flat.append(name if name else "col")
        return flat
    return [str(c).strip() for c in cols]

def make_unique(seq: list) -> list:
    seen = {}
    out = []
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
def load_todoran_table(url: str, version: int = 17):
    """
    とどラン記事URLから、
    - df: columns = ['pref','value']（都道府県と総数系の実数値）
    - label: グラフに使う日本語ラベル（caption > h1/title > 値列名）
    を返す。
    """
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
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名", "道府県", "府県"))]
        if not pref_cols:
            return None, None

        def bad_name(name: str) -> bool:
            n = str(name)
            return any(w in n for w in EXCLUDE_WORDS)

        raw_value_candidates = [
            c for c in cols
            if (c not in ("順位","都道府県","道府県","県名","府県")) and (not bad_name(c))
        ]
        total_name_candidates = [
            c for c in raw_value_candidates
            if any(k in c for k in TOTAL_KEYWORDS) and not any(r in c for r in RATE_WORDS)
        ]
        fallback_candidates = [
            c for c in raw_value_candidates
            if not any(r in c for r in RATE_WORDS)
        ]

        def score_and_build(pref_col: str, candidate_cols: list):
            best_score = -1
            best_df = None
            best_vc = None

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
                col_num = pd.to_numeric(col.map(to_number), errors="coerce")
                col_num = col_num.loc[mask]

                if is_rank_like(col_num):
                    continue

                base = int(col_num.notna().sum())
                bonus = 15 if any(k in vc for k in TOTAL_KEYWORDS) else 0
                score = base + bonus

                if score > best_score and base >= 30:
                    tmp = pd.DataFrame({"pref": pref_series.loc[mask].values, "value": col_num.values})
                    tmp = tmp.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
                    best_score = score
                    best_df = tmp
                    best_vc = vc

            return best_df, best_vc

        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, total_name_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                got = got.sort_values("pref").reset_index(drop=True)
                return got, val_col

        for pref_col in pref_cols:
            got, val_col = score_and_build(pref_col, fallback_candidates)
            if got is not None:
                got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
                got = got.sort_values("pref").reset_index(drop=True)
                return got, val_col

        return None, None

    # read_html の各表を試し、caption をラベル候補に使う
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

    # フォールバック（簡易抽出）
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
            pref = m.group(1)
            val = float(m.group(2))
            if pref in PREF_SET:
                rows.append((pref, val))

    if rows:
        # 👇 ここを修正：余分な ']' を削除
        work = pd.DataFrame(rows, columns=["pref", "value"]).drop_duplicates("pref")
        work["pref"] = pd.Categorical(work["pref"], categories=PREFS, ordered=True)
        work = work.sort_values("pref").reset_index(drop=True)
        label = compose_label(None, None, page_h1 or page_title)
        return work, label

    return pd.DataFrame(columns=["pref","value"]), "データ"

# -------------------- UI --------------------
c1, c2 = st.columns(2)
with c1:
    url_a = st.text_input("データAのURL（とどラン記事）", placeholder="https://todo-ran.com/t/kiji/XXXXX")
with c2:
    url_b = st.text_input("データBのURL（とどラン記事）", placeholder="https://todo-ran.com/t/kiji/YYYYY")

if st.button("相関を計算・表示する", type="primary"):
    if not url_a or not url_b:
        st.error("2つのURLを入力してください。")
        st.stop()
    try:
        df_a, label_a = load_todoran_table(url_a)
        df_b, label_b = load_todoran_table(url_b)
    except requests.RequestException as e:
        st.error(f"ページの取得に失敗しました：{e}")
        st.stop()

    if df_a.empty or df_b.empty:
        st.error("表の抽出に失敗しました。URLがランキング記事であること、表に『都道府県』列があることを確認してください。")
        st.stop()

    # 共通都道府県で結合
    df = pd.merge(
        df_a.rename(columns={"value": "value_a"}),
        df_b.rename(columns={"value": "value_b"}),
        on="pref",
        how="inner",
    )

    # 表示用：列名にラベルを使う（内部計算は value_a/value_b）
    display_df = df.rename(columns={"value_a": label_a, "value_b": label_b})

    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(df) < 3:
        st.warning("共通データが少ないため、相関係数が不安定です。別の指標でお試しください。")
        st.stop()

    # 相関（指定の表記で出力）
    x = pd.to_numeric(df["value_a"], errors="coerce")
    y = pd.to_numeric(df["value_b"], errors="coerce")
    mask = x.notna() & y.notna()
    r = float(x[mask].corr(y[mask]))
    r2 = r ** 2

    st.subheader("相関の結果")
    st.markdown(f"**相関係数 r = {r:.4f}**")
    st.markdown(f"**決定係数 r2 = {r2:.4f}**")

    # 散布図（回帰直線つき：幅640px）
    st.subheader("散布図")
    draw_scatter_with_reg(df, label_a, label_b)

    # 箱ひげ図（左右に横並び：各320px）
    st.subheader("箱ひげ図")
    col_left, col_right = st.columns(2)
    with col_left:
        draw_boxplot(df["value_a"], label_a)
    with col_right:
        draw_boxplot(df["value_b"], label_b)

    # CSVダウンロード（内部名のまま保存：分析向け）
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結合データをCSVで保存（内部列名：value_a/value_b）",
        data=csv,
        file_name="todoran_merged.csv",
        mime="text/csv",
    )
else:
    st.info("上の2つの入力欄に とどラン記事のURL を貼ってから「相関を計算・表示する」を押してください。")
