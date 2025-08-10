# streamlit_app.py
# とどランのランキング記事URLを2つ貼り付けて、
# 「都道府県 × 実数値（偏差値は除外）」を自動抽出し、
# 相関係数・決定係数・散布図・箱ひげ図・5数要約を表示するアプリ

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="都道府県データ 相関ツール（URL版）", layout="wide")
st.title("都道府県データ 相関ツール（URL版）")
st.write(
    "とどランの **各ランキング記事のURL** を2つ貼り付けてください。"
    "表の「偏差値」列は使わず、**実数の値**（人数・金額・割合など）を自動抽出します。"
)

# ---- 47都道府県 --------------------------------------------------------------
PREFS = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"
]
PREF_SET = set(PREFS)

# ---- ユーティリティ ----------------------------------------------------------
def to_number(s: str) -> float:
    """文字列から数値（小数含む）を抜き出して float 化。単位や%は無視。失敗時はNaN。"""
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "").replace("　", " ").replace("％", "%")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def five_number_summary(series: pd.Series):
    """最小, Q1, 中央(Q2), Q3, 最大 を辞書で返す"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return dict(最小値=np.nan, 第1四分位=np.nan, 中央値=np.nan, 第3四分位=np.nan, 最大値=np.nan)
    return dict(
        最小値=float(s.min()),
        第1四分位=float(s.quantile(0.25)),
        中央値=float(s.median()),
        第3四分位=float(s.quantile(0.75)),
        最大値=float(s.max()),
    )

def draw_scatter(df: pd.DataFrame, la: str, lb: str):
    fig, ax = plt.subplots()
    ax.scatter(df["value_a"], df["value_b"])
    ax.set_xlabel(la)
    ax.set_ylabel(lb)
    ax.set_title("散布図")
    st.pyplot(fig)

def draw_boxplot(series: pd.Series, label: str):
    fig, ax = plt.subplots()
    ax.boxplot(series.dropna(), vert=True)
    ax.set_title(f"箱ひげ図：{label}")
    st.pyplot(fig)

def flatten_columns(cols) -> list:
    """MultiIndex列や 'Unnamed' を含む列名を1段の文字列リストにフラット化"""
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x) for x in tup if pd.notna(x)]
            parts = [p for p in parts if not p.startswith("Unnamed")]
            name = " ".join(parts).strip()
            flat.append(name if name else "col")
        return flat
    return [str(c).strip() for c in cols]

# ---- とどランURL → (pref, value) DataFrame -----------------------------------
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, version: int = 4) -> pd.DataFrame:
    """とどラン記事URLから、『都道府県』と『偏差値でない実数値の列』を自動抽出して返す。"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    # 候補テーブルから「都道府県」列＋「偏差値でない数値列」を選ぶ
    def pick_value_dataframe(df: pd.DataFrame):
        df = df.copy()
        df.columns = flatten_columns(df.columns)
        df = df.loc[:, ~df.columns.duplicated()]  # 同名列を除外

        cols = list(df.columns)
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名", "道府県", "府県"))]
        if not pref_cols:
            return None

        value_candidates = [c for c in cols if ("偏差値" not in c) and (c not in ("順位", "都道府県", "道府県", "県名", "府県"))]
        if not value_candidates:
            return None

        best_score = -1
        best_df = None
        for pref_col in pref_cols:
            use_cols = [pref_col] + value_candidates
            work = df[use_cols].copy()

            # Series.str は使わず安全にトリム
            work[pref_col] = work[pref_col].map(lambda x: str(x).strip())

            # 47都道府県でフィルタ
            work = work[work[pref_col].isin(PREF_SET)]
            if work.empty:
                continue

            for vc in value_candidates:
                nums = pd.to_numeric(work[vc].apply(to_number), errors="coerce")
                score = int(nums.notna().sum())
                if score > best_score:
                    w2 = pd.DataFrame({"pref": work[pref_col], "value": nums})
                    w2 = w2.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
                    best_score = score
                    best_df = w2

        # 「十分に数値が取れている」ものだけ採用（目安：30件以上）
        if best_df is not None and best_score >= 30 and not best_df.empty:
            best_df["pref"] = pd.Categorical(best_df["pref"], categories=PREFS, ordered=True)
            best_df = best_df.sort_values("pref").reset_index(drop=True)
            return best_df
        return None

    # 1) pandas.read_html で表抽出（複数ある想定）
    tables = []
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html, flavor="bs4")
        except Exception:
            tables = []

    for raw in tables:
        got = pick_value_dataframe(raw)
        if got is not None:
            return got

    # 2) フォールバック：ページテキストからの簡易抽出
    soup = BeautifulSoup(html, "lxml")
    lines = []
    for tag in soup.find_all(text=True):
        t = str(tag).strip()
        if t:
            lines.append(t)
    text = "\n".join(lines)

    rows = []
    for line in text.splitlines():
        # 例: "1 滋賀県 81.78歳 69.77" から「滋賀県 81.78」を拾う
        m = re.search(r"(北海道|..県|..府|東京都)\s+(-?\d+(?:\.\d+)?)", line)
        if m:
            pref = m.group(1)
            val = float(m.group(2))
            if pref in PREF_SET:
                rows.append((pref, val))

    if rows:
        work = pd.DataFrame(rows, columns=["pref", "value"]).drop_duplicates("pref")
        work["pref"] = pd.Categorical(work["pref"], categories=PREFS, ordered=True)
        work = work.sort_values("pref").reset_index(drop=True)
        return work

    # 3) それでも無理なら空
    return pd.DataFrame(columns=["pref", "value"])

# ---- UI ----------------------------------------------------------------------
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
        df_a = load_todoran_table(url_a)
        df_b = load_todoran_table(url_b)
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

    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(df) < 3:
        st.warning("共通データが少ないため、相関係数が不安定です。別の指標でお試しください。")
        st.stop()

    # 相関
    r = float(pd.Series(df["value_a"]).corr(pd.Series(df["value_b"])))
    r2 = r ** 2

    st.subheader("相関の結果")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("相関係数 r（ピアソン）", f"{r:.4f}")
    with m2:
        st.metric("決定係数 R²", f"{r2:.4f}")

    # 散布図
    st.subheader("散布図")
    draw_scatter(df, "データA", "データB")

    # 箱ひげ図と5数要約
    st.subheader("箱ひげ図 と 四分位数（A と B）")
    b1, b2 = st.columns(2)
    with b1:
        draw_boxplot(df["value_a"], "データA")
        a_summary = five_number_summary(df["value_a"])
        st.markdown("**データAの5数要約**")
        st.table(pd.DataFrame(a_summary, index=["値"]))
    with b2:
        draw_boxplot(df["value_b"], "データB")
        b_summary = five_number_summary(df["value_b"])
        st.markdown("**データBの5数要約**")
        st.table(pd.DataFrame(b_summary, index=["値"]))

    # CSVダウンロード
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "結合データをCSVで保存",
        data=csv,
        file_name="todoran_merged.csv",
        mime="text/csv",
    )
else:
    st.info("上の2つの入力欄に とどラン記事のURL を貼ってから「相関を計算・表示する」を押してください。")
