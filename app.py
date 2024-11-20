import streamlit as st
import numpy as np
import pandas as pd
import os  # 导入 os 模块
# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加一张图片
st.image("images/示意图.png", caption="示意图", use_column_width=True)

# 计算两条直线之间的最小距离
def line_distance(point1, point2, direction1, direction2):
    # 计算单位向量
    v1 = direction1 / np.linalg.norm(direction1)
    v2 = direction2 / np.linalg.norm(direction2)
    w0 = point1 - point2
    a = np.dot(v1, v1)  # v1·v1
    b = np.dot(v1, v2)  # v1·v2
    c = np.dot(v2, v2)  # v2·v2
    d = np.dot(v1, w0)  # v1·w0
    e = np.dot(v2, w0)  # v2·w0

    # 计算分母和分子
    denominator = a * c - b * b
    if denominator == 0:
        # 直线平行，返回两条线之间的距离
        return np.linalg.norm(w0 - (d / a) * v1)

    # 计算参数
    s = (b * e - c * d) / denominator
    t = (a * e - b * d) / denominator
    # 计算最近点
    closest_point1 = point1 + s * v1
    closest_point2 = point2 + t * v2
    return np.linalg.norm(closest_point1 - closest_point2)


# Streamlit 页面布局
st.title("进线档线间距计算")

# 用户输入
d = st.number_input("构架导线挂点间距 d (单位: m)", min_value=0.0, value=1.0)
L = st.number_input("进线档档距 L (单位: m)", min_value=0.0, value=1.0)
l1 = st.number_input("终端塔上横担长度 l1 (单位: m)", min_value=0.0, value=1.0)
l2 = st.number_input("终端塔中横担长度 l2 (单位: m)", min_value=0.0, value=1.0)
l3 = st.number_input("终端塔下横担长度 l3 (单位: m)", min_value=0.0, value=1.0)
#构架地线与导线水平间距
d1 = st.number_input("构架地线与导线水平间距 d1 (单位: m)", min_value=0.0, value=1.0)

# 高度输入
h = st.number_input("构架高度 h (单位: m)", min_value=0.0, value=10.0)
H = st.number_input("终端塔呼高 H (单位: m)", min_value=0.0, value=10.0)

# 导线距输入
H1 = st.number_input("中上导线层间距 H1 (单位: m)", min_value=0.0, value=1.0)
H2 = st.number_input("中下导线层间距 H2 (单位: m)", min_value=0.0, value=1.0)

# 选择构架相
选择A = st.selectbox("构架边相（铁塔侧）至", ["上", "中", "下"])

# 根据选择计算直线坐标
if 选择A == "上":
    A = np.array([d1, L, h])
    D = np.array([l1, 0, H+H1+H2])
    # 定义方向向量
    direction1 = D - A
elif 选择A == "中":
    A = np.array([d1, L, h])
    D = np.array([l2, 0, H+H2])
    # 定义方向向量
    direction1 = D - A
else:  # 下
    A = np.array([d1, L, h])
    D = np.array([l3, 0, H])
    # 定义方向向量
    direction1 = D - A
# 选择构架相
选择B = st.selectbox("构架中相至", ["上", "中", "下"])

# 根据选择计算直线坐标
if 选择B == "上":
    B = np.array([d1+d, L, h])
    E = np.array([l1, 0, H+H1+H2])
    # 定义方向向量
    direction2 = E - B
elif 选择B == "中":
    B = np.array([d1+d, L, h])
    E = np.array([l2, 0, H+H2])
    # 定义方向向量
    direction2 = E - B
else:  # 下
    B = np.array([d1+d, L, h])
    E = np.array([l3, 0, H])
    # 定义方向向量
    direction2 = E - B
# 选择构架相
选择C = st.selectbox("构架边相至", ["上", "中", "下"])
# 根据选择计算直线坐标
if 选择C == "上":
    C = np.array([d1+d+d, L, h])
    F = np.array([l1, 0, H+H1+H2])
    # 定义方向向量
    direction3 = F - C
elif 选择C == "中":
    C = np.array([d1+d+d, L, h])
    F = np.array([l2, 0, H+H2])
    # 定义方向向量
    direction3 = F - C
else:  # 下
    C = np.array([d1+d+d, L, h])
    F = np.array([l3, 0, H])
    # 定义方向向量
    direction3 = F - C

# 计算最小距离
distance1 = line_distance(A, B, direction1, direction2)
distance2 = line_distance(B, C, direction2, direction3)
distance3 = line_distance(C, A, direction3, direction1)
# 显示结果
st.write("铁塔侧边向与中相距离: {:.4f} m".format(distance1))
st.write("中相与另一侧边相: {:.4f} m".format(distance2))
st.write("两边相: {:.4f} m".format(distance3))


# 导出数据的部分
if st.button("导出数据为 Excel"):
    # 创建一个字典保存输入数据和计算结果
    data = {
        "构架导线挂点间距 d (单位: m)": [d],
        "进线档档距 L (单位: m)": [L],
        "终端塔上横担长度 l1 (单位: m)": [l1],
        "终端塔中横担长度 l2 (单位: m)": [l2],
        "终端塔下横担长度 l3 (单位: m)": [l3],
        "构架地线与导线水平间距 d1 (单位: m)": [d1],
        "构架高度 h (单位: m)": [h],
        "终端塔呼高 H (单位: m)": [H],
        "中上导线层间距 H1 (单位: m)": [H1],
        "中下导线层间距 H2 (单位: m)": [H2],
        "铁塔侧边相与中相距离(单位: m)": [distance1],
        "中相与外侧边相距离(单位: m)": [distance2],
        "两边相距离": [distance3],
    }

    # 将字典转换为 DataFrame
    df = pd.DataFrame(data)

    # 导出为 Excel 文件
    excel_file = "../计算结果.xlsx"
    df.to_excel(excel_file, index=False)

    # 提供下载链接
    st.success("数据已成功导出为 Excel 文件！")
    with open(excel_file, "rb") as f:
        st.download_button("点击下载 Excel 文件", f, file_name=excel_file)
