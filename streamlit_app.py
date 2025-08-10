import re
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="都道府県データ 相関ツール", layout="wide")

st.title("都道府県データ 相関ツール（貼り付け用）")
st.write("todo-ran.com などの表から **都道府県名と値の2列** をコピー＆ペーストしてください。")

with st.expander("使い方（かんたん）", expanded=False):
    st.markdown(
        "- それぞれのランキングページで、表の **都道府県名の列と値の列** をドラッグ選択 → コピーします。\n"
        "- 下の **データA・データB** に貼り付けます。\n"
        "- 単位（%, 人, 円 など）が付いていてもOK。自動で数値化します。\n"
        "- 2つのデータは **都道府県名で結合** します（共通の都道府県のみを使います）。"
    )

# 入力UI
c1, c2 = st.columns(2)
with c1:
    a_text = st.text_area(
        "データA（都道府県名と値の2列を貼り付け）",
        height=220,
        placeholder="例）\n北海道\t123.4\n青森県\t98.7\n…",
    )
with c2:
    b_text = st.text_area(
        "データB（都道府県名と値の2列を貼り付け）",
        height=220,
        placeholder="例）\n北海道\t56.7%\n青森県\t43.2%\n…",
    )

pref_list = [
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県",
    "埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県",
    "岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県",
    "鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県",
    "佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"
]
pref_set = set(pref_list)

def to_number(x: str):
    """文字列から数値を抜き出して float 化。% は取り除くだけ（そのままの数値として扱う）。"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(",", "").replace("　", " ").replace("％", "%")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return np.nan
    return np.nan

def parse_pasted(text: str) -> pd.DataFrame:
    """貼り付けテキストから (pref, value) の DataFrame を推定して作る。"""
    if not text or not text.strip():
        return pd.DataFrame(columns=["pref","value"])

    rows = []
    for line in text.strip().splitlines():
        parts = re.split(r"[\t,\s]+", line.strip())
        if not parts or len(parts) < 2:
            continue
        pref_idx = None
        for i, tok in enumerate(parts):
            if tok in pref_set:
                pref_idx = i
                break
        if pref_idx is None:
            continue
        val = None
        if pref_idx + 1 < len(parts):
            val = to_number(parts[pref_idx + 1])
        if (val is None or pd.isna(val)) and len(parts) > 2:
            for j in range(len(parts)):
                if j == pref_idx:
                    continue
                v = to_number(parts[j])
                if not pd.isna(v):
                    val = v
                    break
        if not pd.isna(val):
            rows.append((parts[pref_idx], float(val)))

    df = pd.DataFrame(rows, columns=["pref","value"]).drop_duplicates(subset=["pref"], keep="first")
    df["pref"] = pd.Categorical(df["pref"], categories=pref_list, ordered=True)
    df = df.sort_values("pref").reset_index(drop=True)
    return df

def five_number_summary(series: pd.Series):
    """最小値, 第1四分位(Q1), 中央値(Q2), 第3四分位(Q3), 最大値"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return dict(最小値=np.nan, 第1四分位=np.nan, 中央値=np.nan, 第3四分位=np.nan, 最大値=np.nan)
    q1 = float(s.quantile(0.25))
    q2 = float(s.median())
    q3 = float(s.quantile(0.75))
    return dict(最小値=float(s.min()), 第1四分位=q1, 中央値=q2, 第3四分位=q3, 最大値=float(s.max()))

def draw_scatter(df_merged: pd.DataFrame, label_a: str, label_b: str):
    fig, ax = plt.subplots()
    ax.scatter(df_merged["value_a"], df_merged["value_b"])
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title("散布図")
    st.pyplot(fig)

def draw_boxplot(series: pd.Series, label: str):
    fig, ax = plt.subplots()
    ax.boxplot(series.dropna(), vert=True)
    ax.set_title(f"箱ひげ図：{label}")
    st.pyplot(fig)

if st.button("相関を計算・表示する", type="primary"):
    df_a = parse_pasted(a_text)
    df_b = parse_pasted(b_text)

    if df_a.empty or df_b.empty:
        st.error("データAまたはデータBが読み取れませんでした。都道府県名と値の2列を貼り付けてください。")
        st.stop()

    df = pd.merge(df_a.rename(columns={"value":"value_a"}),
                  df_b.rename(columns={"value":"value_b"}),
                  on="pref", how="inner")

    st.subheader("結合後のデータ（共通の都道府県のみ）")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if len(df) < 3:
        st.warning("共通データが少ないため、相関係数の計算が不安定です。")
        st.stop()

    r = float(pd.Series(df["value_a"]).corr(pd.Series(df["value_b"])))
    r2 = r ** 2

    st.subheader("相関の結果")
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
        st.markdown("**データAの要約（5数要約）**")
        a_summary = five_number_summary(df["value_a"])
        st.table(pd.DataFrame(a_summary, index=["値"]))

    with c6:
        draw_boxplot(df["value_b"], "データB")
        st.markdown("**データBの要約（5数要約）**")
        b_summary = five_number_summary(df["value_b"])
        st.table(pd.DataFrame(b_summary, index=["値"]))

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("結合データをCSVで保存", data=csv, file_name="todoran_merged.csv", mime="text/csv")

else:
    st.info("上の2つの入力欄にデータを貼り付けてから「相関を計算・表示する」を押してください。")
