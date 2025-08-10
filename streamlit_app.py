# streamlit_app.py
# 2つの とどラン URL から「都道府県 × 実数値（偏差値は除外）」を自動抽出して
# 相関係数・決定係数・散布図・箱ひげ図・四分位数を表示するアプリ

import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="都道府県データ 相関ツール（URL版）", layout="wide")
st.title("都道府県データ 相関ツール（URL版）")
st.write("とどランの **各ランキングページのURL** を2つ貼り付けてください。表の「偏差値」列は使わず、**実数の値**を自動抽出します。")

# ---- 共通ユーティリティ ----
PREFS = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"
]
PREF_SET = set(PREFS)

def to_number(s: str) -> float:
    """文字列から数値（小数含む）を抜き出して float 化。単位や%は無視。失敗時はNaN。"""
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace(",", "").replace("　", " ").replace("％", "%")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except:
        return np.nan

def five_number_summary(series: pd.Series):
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

# ---- とどランURL → (pref, value) DataFrame ----
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str) -> pd.DataFrame:
    """とどラン記事URLから、「都道府県」と「実数値（偏差値除外）」を抽出。"""
    # 1) 取得
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    # 2) まずは pandas.read_html で表を抽出（lxml/bs4ベース）
    #    一部記事で複数表があるため、候補を走査して最適なものを選ぶ。
    tables = []
    try:
        tables = pd.read_html(html, flavor="lxml")  # lxml優先
    except Exception:
        try:
            tables = pd.read_html(html, flavor="bs4")
        except Exception:
            tables = []

    # 3) ヘッダ判定関数
    def pick_value_column(df: pd.DataFrame):
        """dfの中から『都道府県』列と、『偏差値でない数値列』を推定して返す (value_col名 or None)。"""
        cols = [str(c) for c in df.columns]
        # 都道府県っぽい列名
        pref_col_candidates = [c for c in cols if "都道府県" in c or c in ("県名","道府県","府県")]
        if not pref_col_candidates:
            return None, None
        # 「偏差値」を含まない列名のうち、数値らしい列を優先
        value_candidates = [c for c in cols if ("偏差値" not in c) and (c not in ("順位","都道府県","道府県","県名","府県"))]
        # 列名が決めにくい場合、都道府県列の右隣を優先
        for pref_col in pref_col_candidates:
            # 候補列をスコアリング：数値化できるセルが多いほど高評価
            best_col, best_score = None, -1
            for vc in value_candidates:
                nums = pd.to_numeric(df[vc].apply(to_number), errors="coerce")
                score = nums.notna().sum()
                if score > best_score:
                    best_col, best_score = vc, score
            if best_col and best_score >= 30:  # 47都道府県のうち一定数が数値化できる
                return pref_col, best_col
        return None, None

    # 4) 表ごとに試す
    for df in tables:
        # 列名の重複・Unnamed対策
        df = df.loc[:, ~df.columns.duplicated()].copy()
        pref_col, val_col = pick_value_column(df)
        if pref_col and val_col:
            # 前処理：全国行などを除外
            work = df[[pref_col, val_col]].rename(columns={pref_col:"pref", val_col:"value"})
            # 前後の空白や注釈除去
            work["pref"] = work["pref"].astype(str).str.strip()
            work = work[work["pref"].isin(PREF_SET)]  # 47都道府県のみ
            work["value"] = work["value"].apply(to_number)
            work = work.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
            if len(work) >= 30:
                # 都道府県順に並べ替え
                work["pref"] = pd.Categorical(work["pref"], categories=PREFS, ordered=True)
                work = work.sort_values("pref").reset_index(drop=True)
                return work

    # 5) もし read_html で拾えない場合、簡易パーサ（aタグ＋テキスト行）で拾う最後の手段
    soup = BeautifulSoup(html, "lxml")
    # テーブルのテキストを走査
    lines = []
    for tag in soup.find_all(text=True):
        t = str(tag).strip()
        if t:
            lines.append(t)
    # 連結して行に分割
    text = "\n".join(lines)
    rows = []
    for line in text.splitlines():
        # パターン例: "1 滋賀県 81.78歳 69.77"
        m = re.search(r"(北海道|..県|..府|東京都)\s+(-?\d+(?:\.\d+)?)", line)
        if m:
            pref = m.group(1)
            val = float(m.group(2))
            if pref in PREF_SET:
                rows.append((pref, val))
    if rows:
        work = pd.DataFrame(rows, columns=["pref","value"]).drop_duplicates("pref")
        work["pref"] = pd.Categorical(work["pref"], categories=PREFS, ordered=True)
        work = work.sort_values("pref").reset_index(drop=True)
        return work

    # 6) それでも無理なら空
    return pd.DataFrame(columns=["pref","value"])

# ---- UI ----
c1, c2 = st.columns(2)
with c1:
    url_a = st.text_input("データAのURL（とどラン記事URL）", placeholder="https://todo-ran.com/t/kiji/XXXXX")
with c2:
    url_b = st.text_input("データBのURL（とどラン記事URL）", placeholder="https://todo-ran.com/t/kiji/YYYYY")

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
        st.error("表の抽出に失敗しました。URLがランキング記事であること、ページの表に『都道府県』列があることを確認してください。")
        st.stop()

    # 結合（共通都道府県のみ）
    df = pd.merge(df_a.rename(columns={"value":"value_a"}),
                  df_b.rename(columns={"value":"value_b"}),
                  on="pref", how="inner")

    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(df) < 3:
        st.warning("共通データが少ないため、相関係数が不安定です。別の指標でお試しください。")
        st.stop()

    # 相関
    r = float(pd.Series(df["value_a"]).corr(pd.Series(df["value_b"])))
    r2 = r ** 2
    c3, c4 = st.columns(2)
    with c3:
        st.metric("相関係数 r（ピアソン）", f"{r:.4f}")
    with c4:
        st.metric("決定係数 R²", f"{r2:.4f}")

    st.subheader("散布図")
    draw_scatter(df, "データA", "データB")

    st.subheader("箱ひげ図 と 四分位数（A と B）")
    c5, c6 = st.columns(2)
    with c5:
        draw_boxplot(df["value_a"], "データA")
        a_summary = five_number_summary(df["value_a"])
        st.markdown("**データAの5数要約**")
        st.table(pd.DataFrame(a_summary, index=["値"]))
    with c6:
        draw_boxplot(df["value_b"], "データB")
        b_summary = five_number_summary(df["value_b"])
        st.markdown("**データBの5数要約**")
        st.table(pd.DataFrame(b_summary, index=["値"]))

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("結合データをCSVで保存", data=csv, file_name="todoran_merged.csv", mime="text/csv")
else:
    st.info("上の2つの入力欄に とどラン記事のURL を貼ってから「相関を計算・表示する」を押してください。")
