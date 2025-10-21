# streamlit_app.py
# CorrGraph（高校生にも分かりやすい相関分析ツール）
# - とどランURL×2 → 都道府県データ抽出・結合・相関分析
# - 箱ひげ図の外れ値（IQR, 1.5倍ルール）を「除外外れ値」として統一
# - 散布図＋周辺箱ひげ図（外れ値含む／除外の2種類）
#   * 外れ値含む散布図では外れ値だけ青
#   * 回帰直線と相関係数 r / r² を表示
# - 外れ値一覧を見やすく表示（バッジ＋表＋CSV保存）
# - 高校生向けの「箱ひげ図の基準」説明つき

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

# ===== UIスタイル =====
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
WHIS = 1.5  # 箱ひげのひげ長（IQR倍率）

st.title("CorrGraph：都道府県データの相関分析ツール")
st.write("とどランの記事URLを2つ入力すると、共通の都道府県データで相関関係を分析できます。")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { color:#111; background:#f5f5f5; }
.block-container { max-width:980px; padding-top:1.2rem; padding-bottom:3rem; }
h1, h2, h3 { color:#111; letter-spacing:.01em; }
h1 { font-weight:800; } h2,h3 { font-weight:700; }
button[kind="primary"], .stButton>button { background:#222; color:#fff; border:1.5px solid #000; }
[data-testid="stDataFrame"] thead tr th { background:#e8e8e8; color:#111; font-weight:700; }
.badge { display:inline-block; padding:.2rem .55rem; border-radius:9999px;
         border:1px solid #111; background:#fff; color:#111; margin-right:.35rem; }
.badge.blue { background:#e7f0ff; border-color:#2b67f6; color:#1a3daa; }
.badge.gray { background:#eee; color:#222; }
.small { font-size:.9rem; color:#333; }
</style>
""", unsafe_allow_html=True)

# ===== 定数 =====
BASE_W_INCH, BASE_H_INCH = 6.4, 4.8
PREFS = ["北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
          "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
          "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
          "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
          "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"]
PREF_SET = set(PREFS)
TOTAL_KEYWORDS = ["総数","合計","件数","人数","人口","世帯","戸数","台数","店舗数","病床数","施設数","金額","額","費用","支出","収入","販売額","生産額","生産量","面積","延","数"]
RATE_WORDS = ["率","割合","比率","％","パーセント","人当たり","一人当たり","人口当たり","千人当たり","10万人当たり","当たり","戸建て率"]
EXCLUDE_WORDS = ["順位","偏差値"]

# ===== セッション初期化 =====
for key in ["url_a", "url_b", "display_df", "calc"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "url" in key else None

# ===== ユーティリティ =====
def show_fig(fig, width_px: int):
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

def fmt(v) -> str:
    try:
        vf = float(v)
        if not np.isfinite(vf):
            return "-"
        return f"{vf:.3f}"
    except Exception:
        return "-"

# 箱ひげ（whis=1.5）に一致する外れ値判定
def boxplot_inlier_mask(arr: np.ndarray, whis: float = WHIS):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=bool), (np.nan, np.nan, np.nan, np.nan, np.nan)
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    iqr = q3 - q1
    low = q1 - whis * iqr
    high = q3 + whis * iqr
    inlier = (arr >= low) & (arr <= high)
    inlier[~valid] = False
    return inlier, (q1, q3, iqr, low, high)

# 散布図＋周辺箱ひげ図（外れ値は青、回帰直線＆r/r² 表示）
def draw_scatter_with_marginal_boxplots(
    x, y, la, lb, title, width_px, outs_x=None, outs_y=None, pref_all=None
):
    import matplotlib.gridspec as gridspec
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[ok]; y = np.asarray(y)[ok]
    if len(x) == 0:
        st.warning("描画できるデータがありません。")
        return

    fig = plt.figure(figsize=(BASE_W_INCH*1.2, BASE_H_INCH*1.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[4,1], wspace=0.05, hspace=0.05)
    ax_main  = fig.add_subplot(gs[0,1])
    ax_box_y = fig.add_subplot(gs[0,0], sharey=ax_main)
    ax_box_x = fig.add_subplot(gs[1,1])
    ax_empty = fig.add_subplot(gs[1,0]); ax_empty.axis("off")

    # 散布図（外れ値だけ青）
    handles, labels = [], []
    if outs_x is not None and outs_y is not None and pref_all is not None:
        pref_all = np.asarray(pref_all)[ok]
        out_set_x = set(map(str, outs_x))
        out_set_y = set(map(str, outs_y))
        out_mask = np.array([(p in out_set_x) or (p in out_set_y) for p in pref_all])
        in_mask = ~out_mask
        h1 = ax_main.scatter(x[in_mask], y[in_mask], color="gray", s=DEFAULT_MARKER_SIZE, label="通常データ")
        h2 = ax_main.scatter(x[out_mask], y[out_mask], color="blue", s=DEFAULT_MARKER_SIZE, label="外れ値")
        handles.extend([h1, h2]); labels.extend(["通常データ", "外れ値"])
    else:
        h = ax_main.scatter(x, y, color="gray", s=DEFAULT_MARKER_SIZE, label="データ")
        handles.append(h); labels.append("データ")

    # 回帰直線＆相関係数
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 200)
        line = ax_main.plot(xs, slope*xs + intercept, color="black", linewidth=DEFAULT_LINE_WIDTH, label="回帰直線")
        r = float(np.corrcoef(x, y)[0, 1]); r2 = r**2
        ax_main.legend(handles + line, labels + [f"r={r:.3f}, r²={r2:.3f}"], frameon=False, loc="best")
    else:
        ax_main.legend(handles, labels, frameon=False, loc="best")

    # 軸（Yラベルは箱ひげ側のみ）
    ax_main.set_xlabel(la)
    ax_main.set_ylabel("")
    ax_main.set_title(title)
    ax_main.tick_params(axis="y", which="both", left=False, labelleft=False)

    # 周辺箱ひげ（whis=1.5）
    ax_box_x.boxplot(x, vert=False, widths=0.6, whis=WHIS, showfliers=True)
    ax_box_x.set_xlabel(la)
    ax_box_x.yaxis.set_visible(False)

    ax_box_y.boxplot(y, vert=True, widths=0.6, whis=WHIS, showfliers=True)
    ax_box_y.set_ylabel(lb)
    ax_box_y.xaxis.set_visible(False)

    show_fig(fig, width_px)

# ===== URL読み込み（とどラン表抽出） =====
@st.cache_data(show_spinner=False)
def load_todoran_table(url: str, allow_rate: bool = True):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit/URL-extractor)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "lxml")
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html, flavor="bs4")
        except Exception:
            tables = []
    bs_tables = soup.find_all("table")

    def flatten_columns(cols):
        def _n(c): return re.sub(r"\s+", "", str(c).strip())
        if isinstance(cols, pd.MultiIndex):
            flat = []
            for tup in cols:
                parts = [str(x) for x in tup if pd.notna(x)]
                parts = [p for p in parts if not str(p).startswith("Unnamed")]
                nm = " ".join(parts).strip()
                flat.append(nm if nm else "col")
            return [_n(c) for c in flat]
        return [_n(c) for c in cols]

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
            if s and str(s).strip(): return str(s).strip()
        return "データ"

    def pick_value_dataframe(df):
        df = df.copy()
        df.columns = make_unique(flatten_columns(df.columns))
        df = df.loc[:, ~df.columns.duplicated()]
        cols = list(df.columns)
        pref_cols = [c for c in cols if ("都道府県" in c) or (c in ("県名","道府県","府県"))]
        if not pref_cols:
            return None, None

        def bad_name(name: str): return any(w in str(name) for w in EXCLUDE_WORDS)
        raw_value_candidates = [c for c in cols if (c not in ("順位","都道府県","県名","府県")) and not bad_name(c)]
        total_name_candidates = [c for c in raw_value_candidates if any(k in c for k in TOTAL_KEYWORDS)]
        fallback_candidates = raw_value_candidates if allow_rate else [c for c in raw_value_candidates if not any(rw in c for rw in RATE_WORDS)]

        def score_and_build(pref_col, candidate_cols):
            best_score, best_df, best_vc = -1, None, None
            pref_series = df[pref_col]
            if isinstance(pref_series, pd.DataFrame): pref_series = pref_series.iloc[:, 0]
            pref_series = pref_series.map(lambda x: str(x).strip())
            mask = pref_series.isin(PREF_SET).to_numpy()
            if not mask.any(): return None, None
            for vc in candidate_cols:
                if vc not in df.columns: continue
                col = df[vc]
                if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
                col_num = pd.to_numeric(col.map(to_number), errors="coerce").loc[mask]
                if is_rank_like(col_num): continue
                base = int(col_num.notna().sum()); bonus = 15 if any(k in vc for k in TOTAL_KEYWORDS) else 0
                score = base + bonus
                if score > best_score and base >= 30:
                    tmp = pd.DataFrame({"pref": pref_series.loc[mask].values, "value": col_num.values})
                    tmp = tmp.dropna(subset=["value"]).drop_duplicates(subset=["pref"])
                    best_score, best_df, best_vc = score, tmp, vc
            return best_df, best_vc

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

# ===== UI（入力とボタン） =====
url_a = st.text_input("X軸URL（説明変数）", placeholder="https://todo-ran.com/t/kiji/XXXXX", key="url_a")
url_b = st.text_input("Y軸URL（目的変数）", placeholder="https://todo-ran.com/t/kiji/YYYYY", key="url_b")
allow_rate = st.checkbox("割合（％・〜当たり）を含める", value=True)

def clear_all():
    st.session_state["url_a"] = ""
    st.session_state["url_b"] = ""
    st.session_state["display_df"] = None
    st.session_state["calc"] = None
    st.rerun()

col_btn1, col_btn2 = st.columns([2,1])
with col_btn1:
    do_calc = st.button("相関を計算・表示する", type="primary")
with col_btn2:
    st.button("クリア", on_click=clear_all)

# ===== メイン計算 =====
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

    # 数値化と外れ値判定
    x0 = pd.to_numeric(df["value_a"], errors="coerce")
    y0 = pd.to_numeric(df["value_b"], errors="coerce")
    mask0 = x0.notna() & y0.notna()
    x_all = x0[mask0].to_numpy()
    y_all = y0[mask0].to_numpy()
    pref_all = df.loc[mask0, "pref"].astype(str).to_numpy()

    mask_x_in, (q1x, q3x, ix, lox, hix) = boxplot_inlier_mask(x_all, WHIS)
    mask_y_in, (q1y, q3y, iy, loy, hiy) = boxplot_inlier_mask(y_all, WHIS)
    mask_in = mask_x_in & mask_y_in  # どちらか一方でも外れなら除外
    outs_x = pref_all[~mask_x_in]
    outs_y = pref_all[~mask_y_in]

    st.session_state["calc"] = {
        "x_all": x_all, "y_all": y_all,
        "x_in": x_all[mask_in], "y_in": y_all[mask_in],
        "label_a": label_a, "label_b": label_b,
        "outs_x": outs_x, "outs_y": outs_y, "pref_all": pref_all,
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
        file_name="merged_pref_data.csv",
        mime="text/csv"
    )

    if st.session_state.get("calc") is not None:
        c = st.session_state["calc"]

        # 1) 外れ値を含む散布図（外れ値は青）
        draw_scatter_with_marginal_boxplots(
            c["x_all"], c["y_all"], c["label_a"], c["label_b"],
            "散布図＋箱ひげ図（外れ値を含む）", width_px=720,
            outs_x=c["outs_x"], outs_y=c["outs_y"], pref_all=c["pref_all"]
        )

        # 2) 外れ値除外の散布図（通常配色）
        draw_scatter_with_marginal_boxplots(
            c["x_in"], c["y_in"], c["label_a"], c["label_b"],
            "散布図＋箱ひげ図（外れ値除外）", width_px=720
        )

        # ---- 外れ値一覧（見やすく）＋ IQR説明 ----
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
        order_key = out_df["外れ軸"].map({"両方":0, "X":1, "Y":2})
        out_df = out_df.assign(_k=order_key).sort_values(["_k","都道府県"]).drop(columns="_k").reset_index(drop=True)

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
            out_df[["都道府県","外れ軸", f"X値（{label_a}）", f"Y値（{label_b}）"]],
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "外れ値一覧をCSVで保存",
            out_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="outliers_list.csv", mime="text/csv"
        )

        # 高校生向け：箱ひげ図の基準とは？
        st.markdown("### 箱ひげ図で使われている基準とは？")
        st.markdown(
            """
箱ひげ図は、データのばらつきを **四分位数** で表します。  
**IQR（Interquartile Range）＝Q3 − Q1** は、「まん中の50％」の広がりです。

- **Q1（第1四分位数）** … 下から25％の位置  
- **Q3（第3四分位数）** … 下から75％の位置  
- **外れ値のきまり** … **Q1 − 1.5×IQR** より小さいか、**Q3 + 1.5×IQR** より大きい値

このアプリでは、X軸とY軸でそれぞれ外れ値を計算し、**どちらか一方でも外れ値に当てはまる都道府県**は
「外れ値として除外」して、下の「外れ値除外」のグラフを作っています。
            """
        )
