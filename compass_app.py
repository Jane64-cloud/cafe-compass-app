import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load model
model = joblib.load(r'F:\python\PythonProject\adt_predictor_rf.pkl')

#标题
st.title("咖啡店罗盘")
st.write("输入新店基本信息，系统将预测开业稳定后的日均交易单量（ADT）.")

#用户输入
channel = st.selectbox("请选择商圈",['keycity 大型综合商圈', '市级商业中心区', '区域级商业区', '社区型商业', '办公商圈（写字楼、园区）', '住宅',
                                 '旅游', '交通枢纽', '特殊商圈（学校、医院、博物馆）'])
sub_channel = st.selectbox("请选择商区", ['机场','各类办公园区','电影院','商业街','书店/图书馆等文化场所','百货','企业总部','企业总部-内部',
                                     '高速公路服务区','医院','酒店','奢侈品商场','购物中心','超市/大卖场','地铁','社区-商业中心','写字楼门店',
                                     '办公-餐饮街','奥莱','街铺','火车站','餐饮/酒吧街','服务社区的商区','专业市场','独立性较强的景点，一般需购票进入',
                                     '剧院/音乐厅','旅游特色商业街','大学','其他商业类型','其他办公类型','其他类型社区店','其他旅游类型','其他类型交通枢纽'
                                     ,'其他特殊类型'])

tier = st.selectbox("城市等级", ['T1', 'T2', 'T3', 'T4', 'T5'])
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
    
#将输入转为 Dataframe
input_data = pd.DataFrame({
    'channel':[channel],
    'channel_sub':[sub_channel],
    'Tier':[tier],
    'area':[area],
    'FY25_P12_rent':[rent],
    'rent_per_square':[rent_per_square]
})

if st.button("预测ADT"):
    pred = model.predict(input_data)[0]
    st.success(f"预测日均交易单量(ADT)为:{pred:.0f}笔")