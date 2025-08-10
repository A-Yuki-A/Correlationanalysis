# streamlit_app.py
# とどランURL×2 → 都道府県の「総数」データを抽出し、
# 外れ値あり／外れ値除外の散布図（横並び・回帰直線つき）を表示。
# 各散布図の直下に n・相関係数r・決定係数r2 を表示。
# 最下部に外れ値の定義（IQR法）を記載。

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

# === フォント設定：同梱フォント最優先（なければシステム → japanize相当） ===
fp = Path("fonts/SourceHanCodeJP-Regular.otf")  # 置いたフォント
if fp.exists():
    fm.fontManager.addfont(str(fp))
    plt.rcParams["font.family"] = "Source Han Code JP"  # 家族名
else:
    for name in ["Noto Sans JP","IPAexGothic","Yu Gothic","Hiragino Sans","Meiryo"]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の豆腐回避

st.set_page_config(page_title="都道府県データ 相関ツール（URL版）", layout="wide")
st.title("都道府県データ 相関ツール（URL版）")
st.write(
    "とどランの **各ランキング記事のURL** を2つ貼り付けてください。"
    "表の「偏差値」「順位」は使わず、**総数（件数・人数・金額などの実数値）**を自動抽出し、"
    "ページ内の **表タイトル** をグラフのラベルに反映します。"
)

# -------------------- 表示サイズ --------------------
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
EXPORT_DPI = 200
SCATTER_WIDTH_PX = 480   # 横並び2枚で収まる幅

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
RATE_WORDS = ["率","割合","比率","％","パーセント","人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり"]
EXCLUDE_WORDS = ["順位","偏差値"]

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
    """散布図＋回帰直線（日本語）を描き、直下に n, r, r2 を表示"""
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
            r = float(np.corrcoef(x, y)[0, 1])
            r2 = r**2

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
def load_todoran_table(url: str, version: int = 25):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")

    page
