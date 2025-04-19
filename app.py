import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import matplotlib as mpl
import platform
import os
import streamlit as st
from datetime import datetime

# ==================== 字体配置部分 ====================
def configure_noto_font():
    """跨平台字体配置"""
    try:
        # 优先尝试使用系统内置中文字体
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        elif platform.system() == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        else:  # Linux/Streamlit Cloud环境
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 回退到支持中文的字体
        
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        st.warning(f"字体配置失败: {str(e)}")

# 初始化字体配置（必须在所有绘图操作前调用）
configure_noto_font()

# ==================== 页面设置 ====================
st.set_page_config(page_title="3D打印制剂实验数据分析系统", layout="wide")
st.title("3D打印制剂实验数据分析与预测系统")

# ==================== 数据加载 ====================
if os.path.exists('drug_data.csv'):
    df = pd.read_csv('drug_data.csv')
else:
    st.error("未找到数据文件 drug_data.csv")
    st.stop()

# ==================== 数据预处理 ====================
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ==================== 数据分析部分 ====================
# 计算相关性矩阵
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

# 可视化相关性（使用思源黑体）
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('参数相关性矩阵')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# ==================== 模型训练部分 ====================
targets = ['Hardness (kg)', 'Friability (%)', 'T50 (h)', 'T80(h)']
feature_importance = {}
feature_cols = ['Drug Loading (%)', 'Printing Temperature(℃)', 'Layer Height (mm)', 
               'SA/V Ratio (mm-1)', 'Path Spacing (mm)', 'Content of Plasticizer (%)',
               'Matrix Content (%)', 'Shell Number', 'Printing Speed (mm/s)', 
               'Extrusion Speed (mm/s)']

for target in targets:
    temp_df = df.dropna(subset=[target])
    X = temp_df[feature_cols]
    y = temp_df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance[target] = importance.sort_values(ascending=False)
    
    with open(f'model_{target}.pkl', 'wb') as f:
        pickle.dump(model, f)

with open('feature_importance.pkl', 'wb') as f:
    pickle.dump(feature_importance, f)

# ==================== Streamlit 界面部分 ====================
# 数据录入系统
st.header("3D打印制剂实验数据录入系统")
with st.expander("点击填写新实验数据"):
    with st.form("data_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            drug_name = st.text_input("Drug English Name")
            drug_loading = st.number_input("Drug Loading (%)", min_value=0.0, max_value=100.0, step=0.1)
            matrix_type = st.text_input("Matrix Type")
            printing_temp = st.number_input("Printing Temperature(℃)", min_value=0.0, max_value=300.0, step=0.1)
            layer_height = st.number_input("Layer Height (mm)", min_value=0.0, max_value=1.0, step=0.01)
            geometric_shape = st.text_input("Geometric Shape")
            sav_ratio = st.number_input("SA/V Ratio (mm-1)", min_value=0.0, step=0.01)
            
        with col2:
            path_spacing = st.number_input("Path Spacing (mm)", min_value=0.0, step=0.01)
            plasticizer_content = st.number_input("Content of Plasticizer (%)", min_value=0.0, max_value=100.0, step=0.1)
            matrix_content = st.number_input("Matrix Content (%)", min_value=0.0, max_value=100.0, step=0.1)
            shell_number = st.number_input("Shell Number", min_value=0, step=1)
            printing_speed = st.number_input("Printing Speed (mm/s)", min_value=0.0, step=0.1)
            extrusion_speed = st.number_input("Extrusion Speed (mm/s)", min_value=0.0, step=0.01)
            hardness = st.number_input("Hardness (kg)", min_value=0.0, step=0.1)
            
        col3, col4 = st.columns(2)
        with col3:
            friability = st.number_input("Friability (%)", min_value=0.0, step=0.01)
            t50 = st.number_input("T50 (h)", min_value=0.0, step=0.01)
        with col4:
            t80 = st.number_input("T80(h)", min_value=0.0, step=0.01)
        
        submitted = st.form_submit_button("提交数据")
        
        if submitted:
            new_data = {
                'Drug English Name': drug_name,
                'Drug Loading (%)': drug_loading,
                'Matrix Type': matrix_type,
                'Printing Temperature(℃)': printing_temp,
                'Layer Height (mm)': layer_height,
                'Geometric Shape': geometric_shape,
                'SA/V Ratio (mm-1)': sav_ratio,
                'Path Spacing (mm)': path_spacing,
                'Content of Plasticizer (%)': plasticizer_content,
                'Matrix Content (%)': matrix_content,
                'Shell Number': shell_number,
                'Printing Speed (mm/s)': printing_speed,
                'Extrusion Speed (mm/s)': extrusion_speed,
                'Hardness (kg)': hardness,
                'Friability (%)': friability,
                'T50 (h)': t50,
                'T80(h)': t80
            }
            
            new_df = pd.DataFrame([new_data])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv('drug_data.csv', index=False)
            st.success("数据已成功保存！")
            st.balloons()

# 数据分析与可视化
st.header("数据分析与可视化")
st.subheader("参数相关性分析")
st.image('correlation_matrix.png')

# 特征重要性分析
st.subheader("特征重要性分析")
try:
    with open('feature_importance.pkl', 'rb') as f:
        feature_importance = pickle.load(f)
        
    target = st.selectbox("选择目标变量", ['Hardness (kg)', 'Friability (%)', 'T50 (h)', 'T80(h)'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance[target][:10].plot(kind='barh', ax=ax)
    ax.set_title(f'影响{target}的最重要因素')
    ax.set_xlabel('重要性分数')
    st.pyplot(fig)
    
except FileNotFoundError:
    st.warning("未找到特征重要性分析结果，请先运行分析脚本。")

# 预测模型
st.header("3D打印药物质量预测")
try:
    models = {}
    for target in ['Hardness (kg)', 'Friability (%)', 'T50 (h)', 'T80(h)']:
        with open(f'model_{target}.pkl', 'rb') as f:
            models[target] = pickle.load(f)
    
    with st.form("prediction_form"):
        st.write("请输入以下参数进行预测：")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_drug_loading = st.number_input("Drug Loading (%)", min_value=0.0, max_value=100.0, step=0.1, key='pred_drug')
            pred_printing_temp = st.number_input("Printing Temperature(℃)", min_value=0.0, max_value=300.0, step=0.1, key='pred_temp')
            pred_layer_height = st.number_input("Layer Height (mm)", min_value=0.0, max_value=1.0, step=0.01, key='pred_layer')
            pred_sav_ratio = st.number_input("SA/V Ratio (mm-1)", min_value=0.0, step=0.01, key='pred_sav')
            pred_path_spacing = st.number_input("Path Spacing (mm)", min_value=0.0, step=0.01, key='pred_path')
            
        with col2:
            pred_plasticizer = st.number_input("Content of Plasticizer (%)", min_value=0.0, max_value=100.0, step=0.1, key='pred_plastic')
            pred_matrix_content = st.number_input("Matrix Content (%)", min_value=0.0, max_value=100.0, step=0.1, key='pred_matrix')
            pred_shell_number = st.number_input("Shell Number", min_value=0, step=1, key='pred_shell')
            pred_printing_speed = st.number_input("Printing Speed (mm/s)", min_value=0.0, step=0.1, key='pred_print')
            pred_extrusion_speed = st.number_input("Extrusion Speed (mm/s)", min_value=0.0, step=0.01, key='pred_extrusion')
        
        predict_button = st.form_submit_button("进行预测")
        
        if predict_button:
            try:
                input_data = pd.DataFrame({
                    'Drug Loading (%)': [pred_drug_loading],
                    'Printing Temperature(℃)': [pred_printing_temp],
                    'Layer Height (mm)': [pred_layer_height],
                    'SA/V Ratio (mm-1)': [pred_sav_ratio],
                    'Path Spacing (mm)': [pred_path_spacing],
                    'Content of Plasticizer (%)': [pred_plasticizer],
                    'Matrix Content (%)': [pred_matrix_content],
                    'Shell Number': [pred_shell_number],
                    'Printing Speed (mm/s)': [pred_printing_speed],
                    'Extrusion Speed (mm/s)': [pred_extrusion_speed]
                })
                
                feature_cols = list(models['Hardness (kg)'].feature_names_in_)
                input_data = input_data[feature_cols]
                
                predictions = {}
                for target, model in models.items():
                    predictions[target] = model.predict(input_data)[0]
                
                st.subheader("预测结果")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hardness (kg)", f"{predictions['Hardness (kg)']:.2f}")
                with col2:
                    st.metric("Friability (%)", f"{predictions['Friability (%)']:.2f}")
                with col3:
                    st.metric("T50 (h)", f"{predictions['T50 (h)']:.2f}")
                with col4:
                    st.metric("T80(h)", f"{predictions['T80(h)']:.2f}")
                    
            except Exception as e:
                st.error(f"预测过程中出错: {str(e)}")

except FileNotFoundError:
    st.warning("未找到预测模型，请先训练模型。")
except Exception as e:
    st.error(f"加载模型时出错: {str(e)}")

# 数据浏览
st.header("实验数据浏览")
if not df.empty:
    df_with_selections = df.copy()
    df_with_selections.insert(0, "选择", False)
    
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"选择": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        width=1500,
        height=500
    )
    
    selected_rows = edited_df[edited_df["选择"]]
    
    if not selected_rows.empty:
        st.warning(f"已选择 {len(selected_rows)} 行数据准备删除")
        if st.button("确认删除选中行", type="primary"):
            remaining_rows = edited_df[~edited_df["选择"]]
            remaining_rows = remaining_rows.drop(columns=["选择"])
            df = remaining_rows.copy()
            df.to_csv('drug_data.csv', index=False)
            st.success("已成功删除选中的数据行！")
            st.rerun()
else:
    st.warning("当前没有数据可显示")

st.dataframe(df)
