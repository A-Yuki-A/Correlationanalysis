# streamlit_app.py
# CorrGraph（都道府県データ相関分析ツール）
# - とどランURL×2 → 都道府県データ抽出と相関分析
# - 外れ値は箱ひげ図（IQR, whis=1.5）基準（X or Y のどちらか外れで除外）
# - 散布図＋周辺箱ひげ図（外れ値含む／外れ値除外）
#   * 中央は Matplotlib：散布図＋周辺箱ひげ図（軸タイトルは箱ひげ側に表示）
#   * 散布図内に回帰直線＆ r / r² を表示（※軸範囲を線も含むよう調整し、確実に表示）
#   * 下に Plotly 散布図（ホバーで都道府県名と値を表示、箱ひげなし）
#   * Plotlyも外れ値「含む」「除外」を両方表示
# - 外れ値一覧（表＋CSV）
# - 高校生向けIQR説明
# - デバッグ表示あり

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
import plotly.express as px
import plotly.graph_objects as go

# ===== フォント設定 =====
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

# ===== スタイル =====
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
EXPORT_DPI = 200
WHIS = 1.5  # 箱ひげIQR倍率
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8

# ===== タイトル／CSS =====
st.title("都道府県データの相関分析ツール")
st.write("とどランの記事URLを2つ入力すると、相関分析ができます。")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { color:#111; background:#f5f5f5; }

/* ★ 上の余白をしっかり確保（Safe Area対応） */
.block-container {
  max-width:980px;
  padding-top: calc(2.8rem + env(safe-area-inset-top, 0px));
  padding-bottom:3rem;
}

/* 見た目はそのまま */
h1,h2,h3 { color:#111; letter-spacing:.01em; }
h1 { font-weight:800; } h2,h3 { font-weight:700; }
button[kind="primary"], .stButton>button { background:#222; color:#fff; border:1.5px solid #000; }
[data-testid="stDataFrame"] thead tr th { background:#e8e8e8; color:#111; font-weight:700; }
.badge { display:inline-block; padding:.2rem .55rem; border-radius:9999px;
         border:1px solid #111; background:#fff; color:#111; margin-right:.35rem; }
.badge.blue { background:#e7f0ff; border-color:#2b67f6; color:#1a3daa; }
.badge.gray { background:#eee; color:#222; }
.small { font-size:.9rem; color:#333; }

/* 見出しへスクロールしたときの欠け防止（内部リンク用） */
h1, h2, h3 { scroll-margin-top: 96px; }
</style>
""", unsafe_allow_html=True)

# ===== 定数 =====
PREFS = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"
]
PREF_SET = set(PREFS)
TOTAL_KEYWORDS = ["総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数","金額","額","費用","支出","収入","販売額","生産額","生産量","面積","延","数"]
RATE_WORDS = ["率","割合","比率","％","パーセント","人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり","戸建て率"]
EXCLUDE_WORDS = ["順位","偏差値"]

# ===== ユーティリティ =====
def _clean_label(s: str) -> str:
    txt = str(s or "")
    # 「昇順」「降順」やその括弧つき表記を除去
    txt = re.sub(r"[（(]?(昇順|降順)[)）]?", "", txt)
    # 連続空白を1つに
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def show_fig(fig, width_px):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)

def to_number(x):
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

def fmt(v):
    try:
        vf = float(v)
        if not np.isfinite(vf):
            return "-"
        return f"{vf:.3f}"
    except Exception:
        return "-"

def boxplot_inlier_mask(arr, whis=WHIS):
    """箱ひげ（whis×IQR）基準でinlier判定と境界値を返す"""
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=bool), (np.nan,)*5
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    low, high = q1 - whis * iqr, q3 + whis * iqr
    inlier = (arr >= low) & (arr <= high)
    inlier[~valid] = False
    return inlier, (q1, q3, iqr, low, high)

# ===== 散布図＋箱ひげ図（Matplotlib：軸タイトルは箱ひげ側に表示） =====
def draw_scatter_with_marginal_boxplots(
    x, y, la, lb, title, width_px, outs_x=None, outs_y=None, pref_all=None
):
    import matplotlib.gridspec as gridspec

    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]
    y = np.asarray(y)[ok]
    if len(x) == 0:
        st.warning("描画できるデータがありません。")
        return

    fig = plt.figure(figsize=(BASE_W_INCH * 1.2, BASE_H_INCH * 1.2))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, 4],
        height_ratios=[4, 1],
        wspace=0.03,
        hspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[0, 1])
    ax_box_y = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_box_x = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_empty = fig.add_subplot(gs[1, 0]); ax_empty.axis("off")

    # --- 散布図（外れ値は青） ---
    if outs_x is not None and outs_y is not None and pref_all is not None:
        pref_all = np.asarray(pref_all)[ok]
        out_set_x, out_set_y = set(map(str, outs_x)), set(map(str, outs_y))
        out_mask = np.array([(p in out_set_x) or (p in out_set_y) for p in pref_all])
        in_mask = ~out_mask
        ax_main.scatter(x[in_mask], y[in_mask], color="gray", s=DEFAULT_MARKER_SIZE, label="通常データ", zorder=2)
        ax_main.scatter(x[out_mask], y[out_mask], color="blue", s=DEFAULT_MARKER_SIZE, label="外れ値", zorder=3)
        ax_main.legend(frameon=False, loc="best")
    else:
        ax_main.scatter(x, y, color="gray", s=DEFAULT_MARKER_SIZE, zorder=2)

    # --- ラベル類 ---
    la_clean = _clean_label(la)
    lb_clean = _clean_label(lb)
    ax_main.set_title(_clean_label(title))
    ax_main.xaxis.offsetText.set_visible(False)
    ax_main.yaxis.offsetText.set_visible(False)

    # --- 周辺箱ひげ（whis=1.5） ---
    ax_box_x.boxplot(x, vert=False, widths=0.6, whis=WHIS, showfliers=True)
    ax_box_x.set_xlabel(la_clean, labelpad=6)
    ax_box_x.set_yticks([])

    ax_box_y.boxplot(y, vert=True, widths=0.6, whis=WHIS, showfliers=True)
    ax_box_y.set_ylabel(lb_clean, labelpad=6)
    ax_box_y.set_xticks([])

    # --- 回帰直線＆ r / r²（最後に描く＋軸範囲を線も含むよう調整）---
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        r2 = r ** 2

        xs = np.linspace(float(x.min()), float(x.max()), 200)
        ys = slope * xs + intercept

        ax_main.plot(xs, ys, color="black", linewidth=DEFAULT_LINE_WIDTH, zorder=10)
        ax_main.text(
            0.02, 0.95, f"r={r:.3f}\nr²={r2:.3f}",
            transform=ax_main.transAxes, ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=11,
        )

        # ★重要：点の範囲＋回帰直線の範囲を両方含む
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin_pts, ymax_pts = float(np.min(y)), float(np.max(y))
        ymin_line, ymax_line = float(np.min(ys)), float(np.max(ys))
        ymin = min(ymin_pts, ymin_line)
        ymax = max(ymax_pts, ymax_line)

        pad_x = (xmax - xmin) * 0.03 if xmax > xmin else 1.0
        pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 1.0

        ax_main.set_xlim(xmin - pad_x, xmax + pad_x)
        ax_main.set_ylim(ymin - pad_y, ymax + pad_y)
    else:
        ax_main.text(
            0.02, 0.95, "回帰直線なし\n(点数不足/ばらつき不足)",
            transform=ax_main.transAxes, ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=11,
        )

    show_fig(fig, width_px)

# ===== インタラクティブ散布図（Plotly：箱ひげなし、散布図だけホバー） =====
def draw_interactive_scatter(x, y, la, lb, pref_all, title, width_px=720):
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]
    y = np.asarray(y)[ok]
    if len(x) == 0:
        st.warning("描画できるデータがありません。")
        return

    if pref_all is not None:
        pref_all = np.asarray(pref_all)[ok].astype(str)
    else:
        pref_all = np.array([""] * len(x))

    df_plot = pd.DataFrame({"x": x, "y": y, "pref": pref_all})

    la_clean = _clean_label(la)
    lb_clean = _clean_label(lb)
    title_clean = _clean_label(title)

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        hover_data={"pref": True, "x": ":.3f", "y": ":.3f"},
        labels={"x": la_clean, "y": lb_clean, "pref": "都道府県"},
        title=title_clean,
    )

    # 回帰直線（※この散布図の x,y から毎回計算）
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs_line = np.linspace(float(x.min()), float(x.max()), 200)
        ys_line = slope * xs_line + intercept
        r = float(np.corrcoef(x, y)[0, 1]); r2 = r ** 2

        fig.add_trace(
            go.Scatter(
                x=xs_line,
                y=ys_line,
                mode="lines",
                name="回帰直線",
                line=dict(color="black", width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"r={r:.3f}<br>r²={r2:.3f}",
            showarrow=False,
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=0,
            font=dict(size=12),
        )

    fig.update_layout(
        width=width_px,
        height=int(width_px * 0.7),
        margin=dict(l=60, r=20, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

# ===== URL入力 =====
url_a = st.text_input("X軸URL（説明変数）", placeholder="https://todo-ran.com/t/kiji/XXXXX", key="url_a")
url_b = st.text_input("Y軸URL（目的変数）", placeholder="https://todo-ran.com/t/kiji/YYYYY", key="url_b")
allow_rate = st.checkbox("割合（％・〜当たり）を含める", value=True)
debug_mode = st.checkbox("デバッグ表示（表の抽出状況を表示）", value=False)

# ===== URL読込（抽出強化＋デバッグ対応） =====
@st.cache_data(show_spinner=False)
def load_todoran_table(url, allow_rate=True, debug=False):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "us-ascii"):
        try:
            r.encoding = r.apparent_encoding
        except Exception:
            pass
    r.raise_for_status()
    html = r.text

    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html, flavor="bs4")
        except Exception:
            tables = []

    if debug:
        st.info(f"URL: {url} / 検出テーブル数: {len(tables)}")
        for i, t in enumerate(tables):
            st.write(f"表 {i+1} の列名: {list(t.columns)}")
            st.dataframe(t.head(5), use_container_width=True)

    PREF_PAT = re.compile("(" + "|".join(map(re.escape, PREFS)) + ")")

    def extract_pref(df: pd.DataFrame):
        for c in df.columns:
            s = df[c].astype(str).str.replace(r"\s+", "", regex=True)
            pref = s.str.extract(PREF_PAT, expand=False)
            if pref.isin(PREFS).sum() >= 30:
                return pref
        return None

    def is_rank_like(nums: pd.Series) -> bool:
        s = pd.to_numeric(nums, errors="coerce").dropna()
        if s.empty:
            return False
        ints = (np.abs(s - np.round(s)) < 1e-9)
        share_int = float(ints.mean())
        in_range = float(((s >= 1) & (s <= 60)).mean())
        unique_close = (s.nunique() >= min(30, len(s)))
        return (share_int >= 0.8) and (in_range >= 0.9) and unique_close

    def pick_value_dataframe(df: pd.DataFrame):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        pref_series = extract_pref(df)
        if pref_series is None:
            return None, None

        def bad_name(name: str) -> bool:
            return any(w in str(name) for w in EXCLUDE_WORDS)

        value_candidates = [
            c for c in df.columns
            if (c not in ("順位","都道府県","都道府県名","県名","道府県","府県","自治体","地域"))
            and (not bad_name(c))
        ]

        prior = [c for c in value_candidates if any(k in c for k in TOTAL_KEYWORDS)]
        if allow_rate:
            fallback = value_candidates[:]
        else:
            fallback = [c for c in value_candidates if not any(rw in c for rw in RATE_WORDS)]

        def score_and_build(pref_series, candidate_cols):
            best_score, best_df, best_vc = -1, None, None
            mask_pref = pref_series.isin(PREF_SET).to_numpy()
            for vc in candidate_cols:
                col = pd.to_numeric(df[vc].map(to_number), errors="coerce")
                col = col[mask_pref]
                if is_rank_like(col):
                    continue
                base = int(col.notna().sum())
                bonus = 15 if any(k in vc for k in TOTAL_KEYWORDS) else 0
                score = base + bonus
                if score > best_score and base >= 30:
                    tmp = pd.DataFrame({"pref": pref_series[mask_pref].values, "value": col.values})
                    tmp = tmp.dropna().drop_duplicates(subset=["pref"])
                    best_score, best_df, best_vc = score, tmp, vc
            return best_df, best_vc

        got, val_col = score_and_build(pref_series, prior)
        if got is not None:
            got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
            return got.sort_values("pref").reset_index(drop=True), val_col

        got, val_col = score_and_build(pref_series, fallback)
        if got is not None:
            got["pref"] = pd.Categorical(got["pref"], categories=PREFS, ordered=True)
            return got.sort_values("pref").reset_index(drop=True), val_col

        return None, None

    for raw in tables:
        got, label = pick_value_dataframe(raw)
        if got is not None:
            return got, label

    if debug:
        st.warning("該当テーブルを特定できませんでした（都道府県列や数値列の候補が不足）。")
    return pd.DataFrame(columns=["pref","value"]), "データ"

# ===== クリア＆ボタン =====
def clear_all():
    for k in ["url_a","url_b","display_df","calc"]:
        st.session_state[k] = None
    st.rerun()

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    do_calc = st.button("相関を計算・表示する", type="primary")
with col_btn2:
    st.button("クリア", on_click=clear_all)

# ===== メイン処理 =====
if do_calc:
    if not url_a or not url_b:
        st.error("2つのURLを入力してください。")
        st.stop()

    try:
        df_a, label_a = load_todoran_table(url_a, allow_rate, debug_mode)
        df_b, label_b = load_todoran_table(url_b, allow_rate, debug_mode)
    except requests.RequestException as e:
        st.error(f"ページの取得に失敗しました：{e}")
        st.stop()

    if df_a.empty or df_b.empty:
        st.error("表の抽出に失敗しました。")
        st.stop()

    df = pd.merge(
        df_a.rename(columns={"value": "value_a"}),
        df_b.rename(columns={"value": "value_b"}),
        on="pref", how="inner"
    )
    if df.empty:
        st.error("共通の都道府県データが見つかりません。")
        st.stop()

    st.session_state["display_df"] = df.rename(columns={"value_a": label_a, "value_b": label_b})

    x0 = pd.to_numeric(df["value_a"], errors="coerce")
    y0 = pd.to_numeric(df["value_b"], errors="coerce")
    mask0 = x0.notna() & y0.notna()

    x_all = x0[mask0].to_numpy()
    y_all = y0[mask0].to_numpy()
    pref_all = df.loc[mask0, "pref"].astype(str).to_numpy()

    mask_x_in, (q1x, q3x, ix, lox, hix) = boxplot_inlier_mask(x_all, WHIS)
    mask_y_in, (q1y, q3y, iy, loy, hiy) = boxplot_inlier_mask(y_all, WHIS)
    mask_in = mask_x_in & mask_y_in

    outs_x = pref_all[~mask_x_in]
    outs_y = pref_all[~mask_y_in]

    x_in = x_all[mask_in]
    y_in = y_all[mask_in]
    pref_in = pref_all[mask_in]

    st.session_state["calc"] = {
        "x_all": x_all, "y_all": y_all, "pref_all": pref_all,
        "x_in": x_in, "y_in": y_in, "pref_in": pref_in,
        "label_a": label_a, "label_b": label_b,
        "outs_x": outs_x, "outs_y": outs_y,
        "iqr_info": {
            "x": {"Q1": q1x, "Q3": q3x, "IQR": ix, "LOW": lox, "HIGH": hix},
            "y": {"Q1": q1y, "Q3": q3y, "IQR": iy, "LOW": loy, "HIGH": hiy},
        }
    }

# ===== 表示 =====
if st.session_state.get("display_df") is not None:
    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(st.session_state["display_df"], use_container_width=True, hide_index=True)
    st.download_button(
        "結合後のデータをCSVで保存",
        st.session_state["display_df"].to_csv(index=False).encode("utf-8-sig"),
        file_name="merged_pref_data.csv", mime="text/csv"
    )

    if st.session_state.get("calc") is not None:
        c = st.session_state["calc"]

        # Matplotlib：外れ値を含む（回帰直線あり）
        draw_scatter_with_marginal_boxplots(
            c["x_all"], c["y_all"], c["label_a"], c["label_b"],
            "散布図＋箱ひげ図（外れ値を含む）", width_px=720,
            outs_x=c["outs_x"], outs_y=c["outs_y"], pref_all=c["pref_all"]
        )

        # Matplotlib：外れ値除外（回帰直線あり）
        draw_scatter_with_marginal_boxplots(
            c["x_in"], c["y_in"], c["label_a"], c["label_b"],
            "散布図＋箱ひげ図（外れ値除外）", width_px=720
        )

        # Plotly：外れ値を含む（回帰直線あり）
        st.markdown("### 散布図（マウスオーバーで都道府県名と値を確認：外れ値を含む）")
        draw_interactive_scatter(
            c["x_all"], c["y_all"],
            c["label_a"], c["label_b"],
            c["pref_all"],
            "インタラクティブ散布図（外れ値を含む）",
            width_px=720
        )

        # Plotly：外れ値除外（回帰直線あり）
        st.markdown("### 散布図（マウスオーバーで都道府県名と値を確認：外れ値除外）")
        draw_interactive_scatter(
            c["x_in"], c["y_in"],
            c["label_a"], c["label_b"],
            c["pref_in"],
            "インタラクティブ散布図（外れ値除外）",
            width_px=720
        )

        # 外れ値一覧
        st.markdown("---")
        st.subheader("外れ値として処理した都道府県（一覧）")
        label_a, label_b = c["label_a"], c["label_b"]
        pref_all = np.asarray(c["pref_all"], dtype=str)
        x_all = np.asarray(c["x_all"], dtype=float)
        y_all = np.asarray(c["y_all"], dtype=float)
        outs_x_set = set(map(str, c["outs_x"]))
        outs_y_set = set(map(str, c["outs_y"]))
        is_out_x = np.array([p in outs_x_set for p in pref_all])
        is_out_y = np.array([p in outs_y_set for p in pref_all])
        is_out_any = is_out_x | is_out_y

        out_df = pd.DataFrame({
            "都道府県": pref_all[is_out_any],
            "外れ軸": np.where(is_out_x[is_out_any] & is_out_y[is_out_any], "両方",
                               np.where(is_out_x[is_out_any], "X", "Y")),
            f"X値（{label_a}）": x_all[is_out_any],
            f"Y値（{label_b}）": y_all[is_out_any],
        })
        order_key = out_df["外れ軸"].map({"両方": 0, "X": 1, "Y": 2})
        out_df = (
            out_df.assign(_k=order_key)
            .sort_values(["_k", "都道府県"])
            .drop(columns="_k")
            .reset_index(drop=True)
        )

        n_x = int(is_out_x.sum()); n_y = int(is_out_y.sum())
        n_both = int((is_out_x & is_out_y).sum()); n_any = int(is_out_any.sum())
        st.markdown(
            f'<span class="badge gray">総数 {n_any} 件</span>'
            f'<span class="badge blue">X軸 {n_x} 件</span>'
            f'<span class="badge blue">Y軸 {n_y} 件</span>'
            f'<span class="badge gray">両方 {n_both} 件</span>',
            unsafe_allow_html=True
        )
        st.dataframe(
            out_df[["都道府県", "外れ軸", f"X値（{label_a}）", f"Y値（{label_b}）"]],
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "外れ値一覧をCSVで保存",
            out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="outliers_list.csv", mime="text/csv"
        )

        # 高校生向け：箱ひげの基準
        st.markdown("### 箱ひげ図で使われている基準とは？")
        st.markdown(
            """
箱ひげ図は、データのばらつきを **四分位数** で表します。  
**IQR（四分位範囲）＝Q3 − Q1** は、「まん中の50％」の広がりを表します。

- **Q1（第1四分位数）** … 下から25％の位置  
- **Q3（第3四分位数）** … 下から75％の位置  
- **外れ値のきまり** … **Q1 − 1.5×IQR** より小さい、または **Q3 + 1.5×IQR** より大きい値

このアプリでは、X軸とY軸の両方についてこの基準で外れ値を判定し、  
**どちらか一方でも外れ値にあてはまる都道府県** を「外れ値」として除外しています。
            """
        )
