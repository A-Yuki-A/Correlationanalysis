# streamlit_app.py
# とどランURL×2 → 都道府県データ抽出、相関分析（外れ値あり／なし散布図）
# ・割合列も許可（オプション）
# ・「偏差値」や「順位」列は除外
# ・外れ値は「X軸で外れ値」「Y軸で外れ値」のみ横並び2カラム表示
# ・グレースケールデザイン／中央寄せ／アクセシビリティ配慮／タイトル余白修正
# ・AI分析（計算結果をSessionに保存→ボタン外置き）
# ・「クリア」ボタンで2つのURLと計算結果をリセット（on_click方式）

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

# -------------------- セッション初期化 --------------------
if "url_a" not in st.session_state:
    st.session_state["url_a"] = ""
if "url_b" not in st.session_state:
    st.session_state["url_b"] = ""

# -------------------- クリア関数 --------------------
def clear_urls():
    st.session_state["url_a"] = ""
    st.session_state["url_b"] = ""
    st.session_state.pop("calc", None)
    st.rerun()

# -------------------- ユーティリティ関数（略） --------------------
# ※ ここは元の to_number, iqr_mask, draw_scatter_reg_with_metrics, load_todoran_table 等そのまま残す
# 長くなるため省略せずに元のコードを全てここに入れてください（前回提示の通り）

# -------------------- UI --------------------
url_a = st.text_input("X軸（説明変数）URL",
                      placeholder="https://todo-ran.com/t/kiji/XXXXX",
                      key="url_a")
url_b = st.text_input("Y軸（目的変数）URL",
                      placeholder="https://todo-ran.com/t/kiji/YYYYY",
                      key="url_b")
allow_rate = st.checkbox("割合（率・％・当たり）も対象にする", value=True)

col_calc, col_clear = st.columns([2, 1])
with col_calc:
    do_calc = st.button("相関を計算・表示する", type="primary")
with col_clear:
    st.button("クリア", help="入力中の2つのURLを消去します", on_click=clear_urls)

# -------------------- メイン処理（計算実行ボタン） --------------------
# ※ ここも元の if do_calc: 以下の相関計算処理を全てそのまま入れてください（前回提示の通り）

# -------------------- AI分析ボタン --------------------
# ※ 元の AI分析 の処理をそのまま入れてください（前回提示の通り）

# -------------------- 外れ値の定義表示 --------------------
# ※ 元の st.markdown(...) の部分をそのまま入れてください（前回提示の通り）
