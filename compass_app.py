import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import joblib

#标题
st.title("咖啡店罗盘")
st.write("输入新店基本信息，系统将预测开业稳定后的日均交易单量（ADT）和年利润（SPC）.")

#load model
@st.cache_resource
def load_models():
    adt_model = joblib.load('adt_predictor_rf.pkl')
    spc_model = joblib.load('spc_predictor_rf.pkl')
    return adt_model, spc_model

#加载平均AT
@st.cache_data
def load_avg_at():
    return pd.read_csv('avg_at.csv')

adt_model, spc_model = load_models()
avg_at_df = load_avg_at()

#使用cat 避免产生无效选项
categories = {
    'province': ['安徽省','北京市','重庆市','福建省','甘肃省','广东省','广西自治区','贵州省','海南省','河北省','河南省','黑龙江省','湖北省','湖南省',
                 '吉林省','江苏省','江西省','辽宁省','内蒙古自治','宁夏自治区','青海省','山东省','山西省','陕西省','上海市','四川省',
                 '天津市','云南省','浙江省'],
    'Tier': ['T1', 'T2', 'T3', 'T4', 'T5'],
    'channel': ['keycity 大型综合商圈','市级商业中心区', '区域级商业区','社区型商业', '办公商圈（写字楼、园区）', '交通枢纽', '住宅', '旅游',
 '特殊商圈（学校、医院、博物馆）'],
    'channel_sub': ['奢侈品商场', '购物中心', '百货', '超市/大卖场', '社区-商业中心', '奥莱', '服务社区的商区', '办公-餐饮街', '写字楼门店', 
                     '各类办公园区', '企业总部', '企业总部-内部', '机场', '火车站', '高速公路服务区', '地铁', '独立性较强的景点，一般需购票进入', 
                     '旅游特色商业街', '大学', '医院', '酒店', '街铺', '商业街','专业市场', '餐饮/酒吧街', '书店/图书馆等文化场所', '电影院', '剧院/音乐厅'],
    'design_type': ['标准店', '高级标准店', '臻选店', '旗舰店']
}
with open('categories.pkl', 'rb') as f:
    categories = pickle.load(f)

#用户输入
province = st.selectbox("省份/直辖市（按首字母排序）", categories['province'])
tier = st.selectbox("城市等级", categories['Tier'])
channel = st.selectbox("请选择商圈", categories['channel'])
channel_sub = st.selectbox("请选择商区", categories['channel_sub'])
design_type = st.selectbox("请选择门店类型", categories['design_type'])
area = st.number_input("面积（平方米）", min_value=0.0, value=0.0)
#租金输入方式，二选一
rent_input_method = st.radio('租金输入方式', ['月租总金额', '每平米月租金额'])

if rent_input_method == "月租总金额":
    rent = st.number_input("月租总金额（元）", min_value=0.0, value=0.0, step=10000.0)
    if area > 0:
        rent_per_square = rent / area
        st.info(f"计算后每平米月租金额：{rent_per_square:.2f}元/平米/月")
    else:
        st.warning("请先输入面积")
        rent_per_square = 0.0
else:
    rent_per_square = st.number_input("每平米租金（元/平米/月）", min_value=0.0, value=0.0, step=100.0)
    rent = area * rent_per_square
    st.info(f"计算后月租金总额：{rent:.2f}元/月")

#新增匹配平均AT
avg_at_row = avg_at_df[(avg_at_df['channel'] == channel) & (avg_at_df['Tier'] == tier) & (avg_at_df['design_type'] == design_type) & 
                        (avg_at_df['channel_sub'] == channel_sub) & (avg_at_df['province'] == province)]
if not avg_at_row.empty:
    avg_at = avg_at_row['avg_at'].values[0]
else:
    st.warning(f"未找到对应渠道{channel} 和城市等级{tier}的平均客单价，将默认使用40元。")
    avg_at=40.0

    
#将输入转为 Dataframe
input_data = pd.DataFrame({
    'province':[province],
    'Tier':[tier],
    'channel':[channel],
    'channel_sub':[channel_sub],
    'design_type':[design_type],
    'area':[area],
    'FY25_P12_rent':[rent],
    'rent_per_square':[rent_per_square],
    'avg_at':[avg_at]
})

if st.button("预测"):
    adt_pred = adt_model.predict(input_data)[0]
    monthly_spc_pred = spc_model.predict(input_data)[0]
    annual_spc_pred = monthly_spc_pred * 12
    
    st.success(f"预测日均交易单量(ADT)为:{adt_pred:.0f}笔")
    st.success(f"预测年利润（SPC）为：{annual_spc_pred:.0f} 元")

# 新增，投资回收期
# 根据设计类型估计投资成本（元/平米）

cost_per_sqm = {
    '标准店':(7000, 9000),
    '高级标准店':(9000,12000),
    '臻选店':(12000,14000),
    '旗舰店':(14000,16000)
}
if design_type in cost_per_sqm:
    low, high = cost_per_sqm[design_type]
    investment = area * np.random.uniform(low, high)
else:
    investment = area * 8000

#预估投资成本+投资回收期

annual_spc_pred = spc_model.predict(input_data)[0] * 12

if annual_spc_pred > 0:
    payback_years = investment /annual_spc_pred
    st.info(f"预估投资成本：{investment:,.0f}元")
    st.info(f"预估投资回收期:{payback_years:.1f}年")
else:
    st.error("预测利润为负，无法计算回收期。")