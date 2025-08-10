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

def make_unique(seq: list) -> list:
    """重複列名を強制的にユニーク化（同名が来たら __2, __3 を付与）"""
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

# ---- とどランURL → (pref, value) DataFrame -----------------------------------
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, version: int = 5) -> pd.DataFrame:
    """とどラン記事URLから、『都道府県』と『偏差値でない実数値の列』を自動抽出して返す。"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    # 候補テーブルから「都道府県」列＋「偏差値でない数値列」を選ぶ
    def pick_value_dataframe(df: pd.DataFrame):
        df = df.copy()
        df.columns = make_unique(flatten_columns(df.columns))  # ← 列名を必ずユニーク化
        # 列名重複をさらに安全側で除去（理屈上は不要だが保険）
        df = df.loc[:, ~df.columns.duplicated()]

        cols = list(df.columns)
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名", "道府県", "府県"))]
        if not pref_cols:
            return None

        value_candidates = [c for c in cols if ("偏差値" not in c) an]()_
