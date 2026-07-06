# app.py
# 运行方式：
#   streamlit run app.py
#
# 功能：
#   科学图像 .npy 数据集多功能工作台
#   - 模块一：数据规范化与增强
#   - 模块二：数据集融合
#   - 模块三：数据清洗与预览
#   - 模块四：复杂天区生成（待接入）

from __future__ import annotations

import streamlit as st

from workbench.pages.cleaning_page import render_cleaning_page
from workbench.pages.complex_sky_page import render_complex_sky_page
from workbench.pages.fusion_page import render_fusion_page
from workbench.pages.normalization_page import render_normalization_page


st.set_page_config(
    page_title="科学图像数据集多功能工作台",
    page_icon="🧪",
    layout="wide",
)


PAGES = {
    "模块一：数据规范化": render_normalization_page,
    "模块二：数据集融合": render_fusion_page,
    "模块三：数据清洗与预览": render_cleaning_page,
    "模块四：复杂天区生成": render_complex_sky_page,
}


with st.sidebar:
    st.title("🧪 数据集工作台")
    selected_page = st.radio(
        "侧边栏导航",
        options=list(PAGES.keys()),
        index=2,
    )
    st.divider()


PAGES[selected_page]()
