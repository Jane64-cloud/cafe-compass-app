import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st

#load model
model = joblib.load('adt_predictor_rf.pkl')
with open('categories.pkl', 'rb') as f:
    categories = pickle.load(f)

preprocessor = model.named_steps['preprocessor']

#标题
st.title("咖啡店罗盘")
st.write("输入新店基本信息，系统将预测开业稳定后的日均交易单量（ADT）.")

#用户输入
province = st.selectbox("省份/直辖市（按首字母排序）", categories['province'])
tier = st.selectbox("城市等级", categories['Tier'])
channel = st.selectbox("请选择商圈", categories['channel'])
sub_channel = st.selectbox("请选择商区", categories['channel_sub'])

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
    'province':[province],
    'Tier':[tier],
    'channel':[channel],
    'channel_sub':[sub_channel],
    'area':[area],
    'FY25_P12_rent':[rent],
    'rent_per_square':[rent_per_square]
})

if st.button("预测ADT"):
    pred = model.predict(input_data)[0]
    st.success(f"预测日均交易单量(ADT)为:{pred:.0f}笔")