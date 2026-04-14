import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import random

# ---------- 页面配置 ----------
st.set_page_config(page_title="咖啡店罗盘", layout="wide")
st.title("☕ 咖啡店罗盘")
st.markdown("输入门店信息和租赁条款，系统将预测未来多年的经营指标。")

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
    # 城市平均 SPC 映射
    try:
        with open('city_spc_mean.pkl', 'rb') as f:
            city_spc_mean = pickle.load(f)
        global_spc_mean = np.mean(list(city_spc_mean.values())) 
    except FileNotFoundError:
        st.warning("未找到 city_spc_mean.pkl，将使用全局平均 SPC（可能影响预测精度）")
        city_spc_mean = {}
        global_spc_mean = 793816.53

    # 城市平均 AT 映射
    try:
        with open('city_at_mean.pkl', 'rb') as f:
            city_at_mean = pickle.load(f)
        global_at_mean = np.mean(list(city_at_mean.values()))
    except FileNotFoundError:
        st.warning("未找到 city_at_mean.pkl，将使用全局平均 AT（可能影响预测精度）")
        city_at_mean = {}
        global_at_mean = 49.43*0.4
        
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
DESIGN_TYPES = ['标准店', '高级标准店', '臻选店', '旗舰店']

# ---------- 侧边栏输入 ----------
with st.sidebar:
    st.header("🏪 门店基本信息")

    province = st.selectbox("省份", all_provinces)

    cities_in_province = city_df[city_df['province'] == province]['city'].tolist()
    city = st.selectbox("城市", cities_in_province)

    # 自动填充城市等级
    Tier = city_df.loc[city_df['city'] == city, 'Tier'].values[0]
    st.text_input("城市等级", value=Tier, disabled=True)

    channel = st.selectbox("商圈", CHANNELS)
    channel_sub = st.selectbox("子商圈", CHANNEL_SUBS)
    design_type = st.selectbox("门店类型", DESIGN_TYPES, index=0)
    area = st.number_input("面积（平方米）", min_value=10.0, value=100.0, step=10.0)

    st.divider()
    st.header("📜 租金条件")

    lease_term = st.number_input("租期（年）", min_value=1, max_value=20, value=10, step=1)
    first_year_rent = st.number_input("首年租金（元/年）", min_value=0, value=0, step=10000)
    rent_escalation = st.number_input("年租金递增比例（%）", min_value=0.0, value=0.0, step=0.5) / 100.0

    # 投资成本估算（根据设计类型）
    cost_per_sqm = {
        '标准店': (7000, 9000),
        '高级标准店': (9000, 12000),
        '臻选店': (12000, 14000),
        '旗舰店': (14000, 16000)
    }
    low, high = cost_per_sqm.get(design_type, (8000, 10000))
    investment = area * random.uniform(low, high)
    st.info(f"预估投资成本：{investment:,.0f} 元")

# ---------- 生成未来年份和租金 ----------
start_year = 2026
years = [start_year]
rents = [first_year_rent]

# ---------- 预测函数 ----------
def predict_year(year, Rent, area, Tier, channel, channel_sub, design_type, province, city):
    """
    对单个年份预测 ADT、NetRevenue
    返回：adt, net, spc, break_even_adt, cost_rates
    """
    # 获取城市编码
    city_spc = city_spc_mean.get(city, global_spc_mean)
    city_at = city_at_mean.get(city, global_at_mean)*0.4

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
        'design_type': design_type,
        'province': province,
        'city': city
    }])

    # 确保分类特征为字符串类型（LightGBM 自动处理）
    cat_cols = ['province', 'city', 'Tier', 'channel', 'channel_sub', 'design_type']
    for col in cat_cols:
        input_df[col] = input_df[col].astype('category')

    # 预测 ADT 和 NetRevenue
    adt = adt_model.predict(input_df)[0]*1.6
    net = net_model.predict(input_df)[0]

    # ---------- 盈亏平衡点 ADT 计算 ----------
    # 瑞幸固定成本
    material_rate = 0.42
    labor_rate = 0.15
    utilities = 0.1
    depreciation = 0.05
    
    total_cost_rate = material_rate + labor_rate + utilities + depreciation
    
    spc = net - Rent - net * total_cost_rate
        
    if (1 - total_cost_rate) > 0 and adt > 0:
        required_net = Rent / (1 - total_cost_rate)
        avg_revenue_per_trans = net / adt
        break_even_adt = (required_net / avg_revenue_per_trans).round(0)
    else:
        break_even_adt = np.nan
     # ---------- 新增：计算Hurdle ADT ----------
    def get_hurdle_multiplier(Tier, design_type):
        base = {'T1': 1.15, 'T2': 1.10, 'T3': 1.08, 'T4': 1.08, 'T5': 1.08}.get(Tier, 1.08)
        if design_type in ['高级标准店', '臻选店', '旗舰店']:
            base += 0.10
        return base

    multiplier = get_hurdle_multiplier(Tier, design_type)
    hurdle_adt = int(break_even_adt * multiplier) if not np.isnan(break_even_adt) else np.nan

    # 成本率
    cost_rates = {
        '材料成本率': material_rate,
        '人工成本率': labor_rate,
        '水电杂费率': utilities,
        '折旧率':depreciation
    }
    
    # 修改返回值，增加 hurdle_adt
    return adt, net, spc, break_even_adt, hurdle_adt, cost_rates

# 主按钮和结果展示 
if st.button("🔮 开始预测", type="primary"):
    results = []
    cost_rates_list = [] #储存每年的成本率
    for year, Rent in zip(years, rents):
        adt, net, spc, break_even_adt, hurdle_adt, cost_rates = predict_year(
            year, Rent, area, Tier, channel, channel_sub, design_type, province, city
        )
        results.append([year, Rent, adt, net, spc, break_even_adt, hurdle_adt])
        cost_rates_list.append(cost_rates)

    result_df = pd.DataFrame(results, columns=[
        '年份', '年租金', 'ADT', '年收入', '年利润', '盈亏平衡ADT', 'HurdleADT'
    ])

    # 格式化数值
    result_df = result_df.round({
        'ADT': 0,
        '年收入': 0,
        '年利润': 0,
        '盈亏平衡ADT': 0
    })
    
    def highlight_negative_column(col):
        if col.name == '年利润':
            return ['color: red' if v < 0 else 'color: black' for v in col]
        else:
            return ['color:black'] * len(col)
    
    # 使用 st.dataframe 并设置列格式（手机自适应）
    styled_df = result_df.style.format({
            '年租金': '{:,.0f}',
            'ADT': '{:,.0f}',
            '年收入': '{:,.0f}',
            '年利润': '{:,.0f}',
            '盈亏平衡ADT': '{:,.0f}'
        }).apply(highlight_negative_column)
    
    st.subheader("📊 开业第一年预估ADT")
    st.dataframe(styled_df, width='stretch', hide_index=True)

    #新增：添加开店建议
    first_year = result_df.iloc[0]
    adt = first_year['ADT']
    break_even = first_year['盈亏平衡ADT']
    hurdle = first_year['HurdleADT']

    if adt >= hurdle:
        suggestion = "✅ 建议开店"
        color = "green"
    elif adt >= break_even:
        suggestion = "⚠️ 需谨慎评估"
        color = "orange"
    else:
        suggestion = "❌ 不建议开店"
        color = "red"
        
    st.subheader("📊 开店决策建议（基于首年预测）")
    st.markdown(f"<h3 style='color:{color}'>{suggestion}</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("预估日均杯数", f"{adt:.0f}")
    col2.metric("盈亏平衡杯数", f"{break_even:.0f}")
    col3.metric("建议最低杯数 (Hurdle)", f"{hurdle:.0f}")

    # st.subheader("💰 运营成本构成明细（范围内每年随机估算）")
    # cost_df = pd.DataFrame(cost_rates_list)
    # cost_df.insert(0, '年份', result_df['年份'])
    # cost_amount_df = cost_df.copy()
    # for col in ['材料成本率', '人工成本率', '水电杂费率', '折旧率']:
    #     cost_amount_df[col] = (cost_amount_df[col] * result_df['年收入']).round(0)
    # cost_amount_df = cost_amount_df.rename(columns={
    #     '材料成本率': '材料成本',
    #     '人工成本率': '人工成本',
    #     '水电杂费率': '水电杂费',
    #     '折旧率': '折旧'
    # })
    # st.dataframe(cost_amount_df.style.format({
    #     '材料成本': '{:,.0f}',
    #     '人工成本': '{:,.0f}',
    #     '水电杂费': '{:,.0f}',
    #     '折旧': '{:,.0f}'
    # }), width='stretch', hide_index=True)
    # st.caption("注：成本率每年随机生成，反映成本估算的不确定性。实际成本请参考企业财务数据。")
    

    # # 投资回收期（累计利润回本）
    # cumulative = 0
    # payback_year = None
    # for _, row in result_df.iterrows():
    #     cumulative += row['年利润']
    #     if cumulative >= investment and payback_year is None:
    #         payback_year = row['年份']
    #         break
    # if payback_year:
    #     st.success(f"💰 预计在 {payback_year} 年收回投资（累计利润 {cumulative:,.0f} 元）")
    # else:
    #     st.warning("⚠️ 租期内累计利润未能覆盖投资成本")


    # 绘制趋势图（手机友好）
    # st.subheader("📈 趋势图")
    # chart_data = result_df.set_index('年份')[['年收入', '年利润']]
    # st.line_chart(chart_data)
    
    # 导出 CSV
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载预测结果 (csv)",
        data=csv,
        file_name="cafe_forecast.csv",
        mime="text/csv"
    )
