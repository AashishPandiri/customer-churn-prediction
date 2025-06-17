import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from model_analytics import ModelAnalytics
from src.models.predict_churn import ChurnPredictor

st.set_page_config(
    page_title="E-commerce Customer Churn Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"

def main():
    st.title("🛒 E-commerce Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard Overview", "Individual Prediction", "Batch Prediction", "Model Analytics"]
    )
    
    if page == "Dashboard Overview":
        dashboard_overview()
    elif page == "Individual Prediction":
        individual_prediction()
    elif page == "Batch Prediction":
        batch_prediction()
    elif page == "Model Analytics":
        model_analytics()

def dashboard_overview():
    st.header("📊 E-commerce Churn Analytics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value="12,543",
            delta="324 from last month"
        )
    
    with col2:
        st.metric(
            label="Churned Customers",
            value="2,156",
            delta="-45 from last month",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Churn Rate",
            value="17.2%",
            delta="-0.8% from last month",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Avg Customer Satisfaction",
            value="3.8/5",
            delta="+0.2 from last month"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        city_tier_data = pd.DataFrame({
            'City_Tier': ['Tier 1', 'Tier 2', 'Tier 3'],
            'Churn_Rate': [15.2, 18.7, 19.8],
            'Total_Customers': [4521, 4312, 3710]
        })
        
        fig = px.bar(
            city_tier_data, 
            x='City_Tier', 
            y='Churn_Rate',
            title='Churn Rate by City Tier',
            color='Churn_Rate',
            color_continuous_scale='Reds',
            text='Churn_Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        payment_data = pd.DataFrame({
            'Payment_Mode': ['Credit Card', 'Debit Card', 'UPI', 'COD', 'E wallet'],
            'Churn_Rate': [14.5, 16.2, 12.8, 22.3, 18.9]
        })
        
        fig = px.pie(
            payment_data, 
            values='Churn_Rate', 
            names='Payment_Mode',
            title='Churn Distribution by Payment Mode'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction_data = pd.DataFrame({
            'Satisfaction_Score': [1, 2, 3, 4, 5],
            'Churn_Rate': [45.2, 32.1, 18.7, 8.9, 3.2],
            'Customer_Count': [856, 1234, 4521, 3892, 2040]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=satisfaction_data['Satisfaction_Score'],
            y=satisfaction_data['Churn_Rate'],
            name='Churn Rate (%)',
            yaxis='y',
            marker_color='red',
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=satisfaction_data['Satisfaction_Score'],
            y=satisfaction_data['Customer_Count'],
            mode='lines+markers',
            name='Customer Count',
            yaxis='y2',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title='Customer Satisfaction vs Churn Rate',
            xaxis_title='Satisfaction Score',
            yaxis=dict(title='Churn Rate (%)', side='left'),
            yaxis2=dict(title='Customer Count', side='right', overlaying='y'),
            legend=dict(x=0.7, y=0.9)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        days_data = pd.DataFrame({
            'Days_Range': ['0-7', '8-15', '16-30', '31-60', '60+'],
            'Churn_Rate': [5.2, 12.8, 25.4, 42.1, 68.9],
            'Customers': [3245, 2876, 2654, 2198, 1570]
        })
        
        fig = px.bar(
            days_data,
            x='Days_Range',
            y='Churn_Rate',
            title='Churn Rate by Days Since Last Order',
            color='Churn_Rate',
            color_continuous_scale='Reds',
            text='Churn_Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_title='Days Since Last Order')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**High Risk Segments**\n- COD payment users (22.3% churn)\n- Low satisfaction scores (1-2)\n- 60+ days since last order")
    
    with col2:
        st.success("**Retention Opportunities**\n- Target Tier 3 cities\n- Improve satisfaction scores\n- Re-engagement campaigns")
    
    with col3:
        st.warning("**Revenue Impact**\n- Avg revenue per churned customer: ₹2,845\n- Monthly revenue at risk: ₹61.4L\n- Potential savings with 20% reduction: ₹12.3L")

def individual_prediction():
    churn_predictor = ChurnPredictor()
    st.header("🎯 Individual Customer Churn Prediction")
    
    with st.form("customer_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Demographics")
            customer_id = st.text_input("Customer ID", value="C001")
            tenure = st.slider("Tenure (months)", 0, 60, 12)
            preferred_login_device = st.selectbox("Preferred Login Device", 
                                                ["Mobile Phone", "Phone", "Computer"])
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            warehouse_to_home = st.slider("Warehouse to Home Distance (km)", 5, 127, 15)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            number_of_address = st.slider("Number of Addresses", 1, 22, 2)
            
        with col2:
            st.subheader("Shopping Behavior")
            preferred_payment_mode = st.selectbox("Preferred Payment Mode", 
                                                ["Debit Card", "UPI", "Credit Card", "Cash on Delivery", "E wallet"])
            hour_spend_on_app = st.slider("Hours Spent on App", 0, 5, 2)
            number_of_device_registered = st.slider("Number of Devices Registered", 1, 6, 3)
            preferred_order_cat = st.selectbox("Preferred Order Category", 
                                             ["Laptop & Accessory", "Mobile Phone", "Others", "Fashion", "Grocery"])
            satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
            complain = st.selectbox("Complain", [0, 1])
            order_amount_hike_from_last_year = st.slider("Order Amount Hike from Last Year (%)", 11, 26, 15)
            coupon_used = st.slider("Coupons Used", 0, 16, 5)
            order_count = st.slider("Order Count", 1, 16, 5)
            days_since_last_order = st.slider("Days Since Last Order", 0, 46, 10)
            cashback_amount = st.slider("Cashback Amount", 0, 324, 150)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            customer_data = {
                "customerID": customer_id,
                "Tenure": tenure,
                "PreferredLoginDevice": preferred_login_device,
                "CityTier": city_tier,
                "WarehouseToHome": warehouse_to_home,
                "PreferredPaymentMode": preferred_payment_mode,
                "Gender": gender,
                "HourSpendOnApp": hour_spend_on_app,
                "NumberOfDeviceRegistered": number_of_device_registered,
                "PreferedOrderCat": preferred_order_cat,
                "SatisfactionScore": satisfaction_score,
                "MaritalStatus": marital_status,
                "NumberOfAddress": number_of_address,
                "Complain": complain,
                "OrderAmountHikeFromlastYear": order_amount_hike_from_last_year,
                "CouponUsed": coupon_used,
                "OrderCount": order_count,
                "DaySinceLastOrder": days_since_last_order,
                "CashbackAmount": cashback_amount
            }
            
            try:
                churn_probability, churn_prediction = churn_predictor.predict_individual(customer_data)

                if churn_probability < 0.3:
                    risk_level = "Low"
                elif churn_probability < 0.7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                st.success("Prediction completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Churn Probability", f"{churn_probability:.2%}")
                
                with col2:
                    prediction_text = "Will Churn" if churn_prediction == 1 else "Will Stay"
                    st.metric("Prediction", prediction_text)
                
                with col3:
                    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_level]
                    st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                              unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = churn_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Risk Factors Analysis")
                risk_breakdown = []
                
                if satisfaction_score <= 2:
                    risk_breakdown.append(f"Low Satisfaction Score ({satisfaction_score}/5)")
                if days_since_last_order > 30:
                    risk_breakdown.append(f"Long time since last order ({days_since_last_order} days)")
                if complain == 1:
                    risk_breakdown.append("Customer has complaints")
                if preferred_payment_mode == "Cash on Delivery":
                    risk_breakdown.append("Uses Cash on Delivery payment")
                if order_count < 3:
                    risk_breakdown.append(f"Low order frequency ({order_count} orders)")
                if tenure < 6:
                    risk_breakdown.append(f"New customer ({tenure} months tenure)")
                
                if risk_breakdown:
                    for factor in risk_breakdown:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ No major risk factors identified")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def batch_prediction():
    churn_predictor = ChurnPredictor()
    st.header("📊 Batch Customer Prediction")
    
    if st.button("Download Sample Template"):
        sample_data = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'Tenure': [12, 8, 24],
            'PreferredLoginDevice': ['Mobile Phone', 'Computer', 'Phone'],
            'CityTier': [1, 2, 3],
            'WarehouseToHome': [15, 20, 10],
            'PreferredPaymentMode': ['Debit Card', 'Credit Card', 'UPI'],
            'Gender': ['Male', 'Female', 'Male'],
            'HourSpendOnApp': [3, 2, 4],
            'NumberOfDeviceRegistered': [4, 3, 2],
            'PreferedOrderCat': ['Laptop & Accessory', 'Fashion', 'Mobile Phone'],
            'SatisfactionScore': [4, 2, 5],
            'MaritalStatus': ['Single', 'Married', 'Single'],
            'NumberOfAddress': [2, 3, 1],
            'Complain': [0, 1, 0],
            'OrderAmountHikeFromlastYear': [15, 20, 12],
            'CouponUsed': [5, 8, 3],
            'OrderCount': [8, 3, 12],
            'DaySinceLastOrder': [5, 25, 2],
            'CashbackAmount': [200, 100, 300]
        })
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv,
            file_name="customer_data_template.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Run Batch Prediction"):
                probabilities, predictions = churn_predictor.predict_batch(uploaded_file)
                
                results_df = pd.DataFrame(predictions)
                results_df['churn_probability'] = probabilities
                
                st.write("Prediction Results:")
                st.dataframe(results_df)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_risk_count = len([p for p in predictions if p['risk_level'] == 'High'])
                    st.metric("High Risk Customers", high_risk_count)
                
                with col2:
                    medium_risk_count = len([p for p in predictions if p['risk_level'] == 'Medium'])
                    st.metric("Medium Risk Customers", medium_risk_count)
                
                with col3:
                    predicted_churners = sum(p['churn_prediction'] for p in predictions)
                    st.metric("Predicted Churners", predicted_churners)
                
                with col4:
                    churn_rate = (predicted_churners / len(predictions)) * 100
                    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                
                risk_dist = pd.DataFrame(results_df['risk_level'].value_counts()).reset_index()
                risk_dist.columns = ['Risk Level', 'Count']
                
                fig = px.pie(risk_dist, values='Count', names='Risk Level', 
                           title='Risk Level Distribution',
                           color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                st.plotly_chart(fig, use_container_width=True)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def model_analytics():
    st.header("🤖 E-commerce Churn Model Analytics")

    model_analytics = ModelAnalytics()
    rf_metrics, xgb_metrics, ensemble_metrics, correlation_data = model_analytics.get_model_metrics()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        def get_metric(metric_dict, metric):
            return metric_dict['classification_report']['weighted avg'].get(metric, None)
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Random Forest': [rf_metrics['accuracy'], get_metric(rf_metrics, 'precision'), get_metric(rf_metrics, 'recall'), get_metric(rf_metrics, 'f1-score'), rf_metrics['auc']],
            'XGBoost': [xgb_metrics['accuracy'], get_metric(xgb_metrics, 'precision'), get_metric(xgb_metrics, 'recall'), get_metric(xgb_metrics, 'f1-score'), xgb_metrics['auc']],
            'Ensemble': [ensemble_metrics['accuracy'], get_metric(ensemble_metrics, 'precision'), get_metric(ensemble_metrics, 'recall'), get_metric(ensemble_metrics, 'f1-score'), ensemble_metrics['auc']]
        })
        st.dataframe(metrics_df, use_container_width=True)
        fig = px.bar(
            metrics_df.melt(id_vars='Metric', var_name='Model', value_name='Score'),
            x='Metric',
            y='Score',
            color='Model',
            barmode='group',
            title='Model Performance Comparison'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Feature Importance")
        
        feature_importance = model_analytics.get_feature_importances()
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Correlation Analysis")
    
    if isinstance(correlation_data, dict):
        correlation_data = pd.DataFrame(correlation_data)
    if isinstance(correlation_data, pd.DataFrame):
        arr = correlation_data.values
    else:
        arr = correlation_data
    np.fill_diagonal(arr, 1)
    if isinstance(correlation_data, pd.DataFrame):
        correlation_data = pd.DataFrame(arr, columns=correlation_data.columns, index=correlation_data.index)

    fig = px.imshow(correlation_data,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()