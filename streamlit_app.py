# streamlit_app.py
# とどランのランキング記事URLを2つ貼り付けて、
# 「都道府県 × 実数値（偏差値や順位は除外）」を自動抽出。
# 表タイトル（<caption>/<h1> 等）をラベルに反映し、
# 相関係数・決定係数・散布図・箱ひげ図・5数要約を表示する。
# さらに、散布図・箱ひげ図を“実寸で50%”に確実に縮小して表示（st.imageでwidth指定）、
# 箱ひげ図は上下に、右側に要約表（四分位数・中央値など）を配置。

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

st.set_page_config(page_title="都道府県データ 相関ツール（URL版）", layout="wide")
st.title("都道府県データ 相関ツール（URL版）")
st.write(
    "とどランの **各ランキング記事のURL** を2つ貼り付けてください。"
    "表の「偏差値」「順位」は使わず、**総数（件数・人数・金額などの実数値）**を自動抽出し、"
    "ページ内の **表タイトル** をグラフや表のラベルに反映します。"
)

# -------------------- 図サイズ制御（50%で確実に表示） --------------------
# Matplotlibの“論理サイズ”は既定 6.4×4.8 inch, DPI=100 → 640×480px 相当。
# ここではPNGにして st.image(width=...) でピクセル幅を強制指定する。
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
EXPORT_DPI = 200                    # 保存時のDPI（高めにしてクッキリ）
DISPLAY_SCALE = 0.50                # 画面表示を50%に
DISPLAY_WIDTH_PX = int(BASE_W_INCH * EXPORT_DPI * DISPLAY_SCALE)

def show_fig(fig):
    """figをPNGにして、width指定で確実に縮小表示。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=DISPLAY_WIDTH_PX)
    plt.close(fig)

# -------------------- 日本語フォント設定（自動検出） --------------------
def set_japanese_font():
    candidates = [
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic",
        "Yu Gothic", "Hiragino Sans", "Meiryo", "TakaoGothic", "VL PGothic"
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

set_japanese_font()

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
    m = re.search(r"-?\d+(?:\.\d+
