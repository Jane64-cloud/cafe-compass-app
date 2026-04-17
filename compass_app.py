import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import random

# ---------- 页面配置 ----------
st.set_page_config(page_title="咖啡店罗盘", layout="wide")
st.title("☕ 咖啡店罗盘")
st.markdown("输入门店信息，系统将预测商区日均订单量。")

# ---------- 加载模型 ----------
@st.cache_resource
def load_models():
    adt_model = joblib.load('adt_model.pkl')
    net_model = joblib.load('net_model.pkl')
    return adt_model, net_model

adt_model, net_model = load_models()

# ---------- 加载城市数据 ----------
@st.cache_data
def load_city_data():
    df = pd.read_csv('city_data.csv')  # 需要包含 province, city, Tier
    return df

city_df = load_city_data()
all_provinces = city_df['province'].unique().tolist()

# ---------- 加载城市目标编码映射 ----------
@st.cache_data
def load_city_encodings():
    try:
        with open('city_spc_mean.pkl', 'rb') as f:
            city_spc_mean = pickle.load(f)
        global_spc_mean = np.mean(list(city_spc_mean.values()))
    except FileNotFoundError:
        st.warning("未找到 city_spc_mean.pkl，将使用全局平均 SPC（可能影响预测精度）")
        city_spc_mean = {}
        global_spc_mean = 793816.53

    try:
        with open('city_at_mean.pkl', 'rb') as f:
            city_at_mean = pickle.load(f)
        global_at_mean = np.mean(list(city_at_mean.values()))
    except FileNotFoundError:
        st.warning("未找到 city_at_mean.pkl，将使用全局平均 AT（可能影响预测精度）")
        city_at_mean = {}
        global_at_mean = 49.43

    return city_spc_mean, global_spc_mean, city_at_mean, global_at_mean

city_spc_mean, global_spc_mean, city_at_mean, global_at_mean = load_city_encodings()

# ---------- 固定选项 ----------
CHANNELS = [
    'keycity 大型综合商圈', '市级商业中心区', '区域级商业区', '社区型商业',
    '办公商圈（写字楼、园区）', '交通枢纽', '住宅', '旅游',
    '特殊商圈（学校、医院、博物馆）'
]
CHANNEL_SUBS = [
    '奢侈品商场', '购物中心', '百货', '超市/大卖场', '社区-商业中心', '奥莱',
    '服务社区的商区', '办公-餐饮街', '写字楼门店', '各类办公园区', '企业总部',
    '企业总部-内部', '机场', '火车站', '高速公路服务区', '地铁',
    '独立性较强的景点，一般需购票进入', '旅游特色商业街', '大学', '医院',
    '酒店', '街铺', '商业街', '专业市场', '餐饮/酒吧街', '书店/图书馆等文化场所',
    '电影院', '剧院/音乐厅', '其他商业类型', '其他办公类型', '其他类型社区店', '其他旅游类型', '其他类型交通枢纽', '其他特殊类型'
]

# 瑞幸店型（前端显示）
RUIXING_TYPES = ['快取店', '悠享店', '旗舰店', '外卖厨房']

# ---------- 侧边栏输入 ----------
with st.sidebar:
    st.header("🏪 门店基本信息")
    province = st.selectbox("省份", all_provinces)
    cities_in_province = city_df[city_df['province'] == province]['city'].tolist()
    city = st.selectbox("城市", cities_in_province)
    Tier = city_df.loc[city_df['city'] == city, 'Tier'].values[0]
    st.text_input("城市等级", value=Tier, disabled=True)

    channel = st.selectbox("商圈", CHANNELS)
    channel_sub = st.selectbox("商区", CHANNEL_SUBS)
    ruixing_type = st.selectbox("门店类型", RUIXING_TYPES, index=0)
    area = st.number_input("面积（平方米）", min_value=10.0, value=100.0, step=10.0)

    st.divider()
    st.header("📜 租金")
    first_year_rent = st.number_input("租金（元/年）", min_value=0, value=0, step=10000)
    lease_term = 1
    rent_escalation = 0.0

# ---------- 瑞幸店型映射到模型店型 ----------
def map_to_starbucks_type(ruixing_type):
    mapping = {
        '快取店': '标准店',
        '悠享店': '高级标准店',
        '旗舰店': '旗舰店',
        '外卖厨房': '标准店'
    }
    return mapping.get(ruixing_type, '标准店')

# ---------- 生成未来年份和租金 ----------
start_year = 2026
years = [start_year]
rents = [first_year_rent]

# ---------- 辅助函数：计算Hurdle倍数 ----------
def get_hurdle_multiplier(Tier, ruixing_type):
    base = {'T1': 1.15, 'T2': 1.10, 'T3': 1.08, 'T4': 1.08, 'T5': 1.08}.get(Tier, 1.08)
    if ruixing_type in ['悠享店', '旗舰店']:  # 悠享店和旗舰店要求更高业绩
        base += 0.10
    return base

# ---------- 预测函数 ----------
def predict_year(year, Rent, area, Tier, channel, channel_sub, ruixing_type, province, city):
    # 获取城市编码
    city_spc = city_spc_mean.get(city, global_spc_mean)
    city_at = city_at_mean.get(city, global_at_mean) * 0.4  

    # 映射店型
    starbucks_type = map_to_starbucks_type(ruixing_type)

    # 构造输入特征
    input_df = pd.DataFrame([{
        'year': year,
        'Rent': Rent,
        'area': area,
        'city_at_mean': city_at,
        'city_spc_mean': city_spc,
        'Tier': Tier,
        'channel': channel,
        'channel_sub': channel_sub,
        'design_type': starbucks_type,
        'province': province,
        'city': city
    }])

    # 确保分类特征为 category 类型
    cat_cols = ['province', 'city', 'Tier', 'channel', 'channel_sub', 'design_type']
    for col in cat_cols:
        input_df[col] = input_df[col].astype('category')

    # 预测 ADT（瑞幸预估：模型输出 * 1.6，经验系数）
    adt = adt_model.predict(input_df)[0] * 1.6
    net = net_model.predict(input_df)[0] * 0.8   # 年收入
    
    # 固定成本率（瑞幸典型值）
    material_rate = 0.30
    labor_rate = 0.22
    utilities = 0.06
    depreciation = 0.06
    total_cost_rate = material_rate + labor_rate + utilities + depreciation

    # 盈亏平衡ADT
    if (1 - total_cost_rate) > 0 and adt > 0:
        required_net = Rent / (1 - total_cost_rate)
        avg_revenue_per_trans = net / adt
        break_even_adt = int((required_net / avg_revenue_per_trans).round(0))
    else:
        break_even_adt = np.nan

    # Hurdle ADT
    multiplier = get_hurdle_multiplier(Tier, ruixing_type)
    hurdle_adt = int(break_even_adt * multiplier) if not np.isnan(break_even_adt) else np.nan

    return adt, break_even_adt, hurdle_adt

# ---------- 主按钮和结果展示 ----------
if st.button("🔮 开始预测", type="primary"):
    results = []
    for year, Rent in zip(years, rents):
        adt, break_even_adt, hurdle_adt = predict_year(
            year, Rent, area, Tier, channel, channel_sub, ruixing_type, province, city
        )
        results.append([year, adt, break_even_adt, hurdle_adt])

    result_df = pd.DataFrame(results, columns=['年份', 'ADT', '盈亏平衡ADT', 'HurdleADT'])
    result_df = result_df.round(0).astype(int)

    # st.subheader("📊 逐年预测结果")
    # st.dataframe(result_df, width='stretch', hide_index=True)

    # 基于首年预测给出评估
    first = result_df.iloc[0]
    adt = first['ADT']
    break_even = first['盈亏平衡ADT']
    hurdle = first['HurdleADT']

    st.subheader("📊 商区评估结果")
    col1, col2, col3 = st.columns(3)
    col1.metric("预估日均订单量", f"{adt}")
    col2.metric("盈亏平衡订单量", f"{break_even}")
    col3.metric("建议最低订单量 (Hurdle)", f"{hurdle}")

    # 简单建议
    if adt >= hurdle:
        st.success("✅ 建议开店")
    elif adt >= break_even:
        st.warning("⚠️ 需谨慎评估")
    else:
        st.error("❌ 不建议开店")

    # 导出 CSV
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载预测结果 (csv)",
        data=csv,
        file_name="cafe_forecast.csv",
        mime="text/csv"
    )
