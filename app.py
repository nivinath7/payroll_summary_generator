# app.py

import streamlit as st
import pandas as pd
import os
import glob
import openai
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="Optum Payroll Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main background and container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        text-align: center;
        opacity: 0.9;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #FF6B35;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 0.5rem;
    }
    
    /* Filter section */
    .filter-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #dee2e6;
    }
    
    /* Data quality indicators */
    .data-quality {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Alert styling */
    .alert-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 107, 53, 0.3);
    }
    
    /* Dataframe styling */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
<div class="main-header">
    <h1>🏥 Optum India Payroll Dashboard</h1>
    <p>Comprehensive Payroll Analytics & Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# --- Set OpenAI API Key ---
api_key = st.secrets["OPENAI_API_KEY"]
if api_key == "YOUR_API_KEY_HERE":
    st.error("⚠️ Please replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key in the code.")
else:
    openai.api_key = api_key

# --- Data Loading Function with Enhanced Error Handling ---
@st.cache_data
def load_data(path):
    try:
        all_files = glob.glob(os.path.join(path, "*.csv"))
        if not all_files:
            st.error(f"📂 No CSV files found in the '{path}' directory. Please add your data files.")
            return pd.DataFrame(), 0
            
        df_list = []
        total_records = 0
        
        for file in all_files:
            df_temp = pd.read_csv(file)
            df_list.append(df_temp)
            total_records += len(df_temp)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df['Pay_Period_Date'] = pd.to_datetime(combined_df['Pay_Period_Date'])
        combined_df = combined_df.sort_values(by='Pay_Period_Date', ascending=False)
        
        return combined_df, total_records
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(), 0

# --- Load the data ---
DATA_PATH = "optum_payroll_files"
df, total_records = load_data(DATA_PATH)

# --- Data Quality Summary ---
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", f"{total_records:,}")
    with col2:
        st.metric("👥 Unique Employees", f"{df['Employee_ID'].nunique():,}")
    with col3:
        st.metric("🏢 Departments", f"{df['Department'].nunique()}")
    with col4:
        st.metric("📅 Pay Periods", f"{df['Pay_Period_Date'].nunique()}")
    
    st.markdown('<div class="data-quality">✅ Data loaded successfully and ready for analysis</div>', unsafe_allow_html=True)
else:
    st.stop()

# --- Enhanced Sidebar for Filters ---
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    st.markdown("---")
    
    # Department filter with search
    department_options = ['All'] + sorted(list(df['Department'].unique()))
    department = st.selectbox("🏢 Department", options=department_options, key="dept_filter")
    
    # Designation filter
    designation_options = ['All'] + sorted(list(df['Designation'].unique()))
    designation = st.selectbox("💼 Designation", options=designation_options, key="desig_filter")
    
    # Date range filter
    st.markdown("📅 **Date Range**")
    date_range = st.date_input(
        "Select Pay Period Range",
        value=(df['Pay_Period_Date'].min().date(), df['Pay_Period_Date'].max().date()),
        min_value=df['Pay_Period_Date'].min().date(),
        max_value=df['Pay_Period_Date'].max().date()
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    # Show filtered data summary in sidebar
    df_temp = df.copy()
    if department != "All":
        df_temp = df_temp[df_temp['Department'] == department]
    if designation != "All":
        df_temp = df_temp[df_temp['Designation'] == designation]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_temp = df_temp[
            (df_temp['Pay_Period_Date'].dt.date >= start_date) &
            (df_temp['Pay_Period_Date'].dt.date <= end_date)
        ]
    
    st.metric("Filtered Records", f"{len(df_temp):,}")
    st.metric("Avg Net Pay", f"₹{df_temp['Net_Pay'].mean():,.0f}" if not df_temp.empty else "₹0")

# --- Filter the dataframe based on selection ---
df_selection = df.copy()
if department != "All":
    df_selection = df_selection[df_selection['Department'] == department]
if designation != "All":
    df_selection = df_selection[df_selection['Designation'] == designation]

if len(date_range) == 2:
    start_date, end_date = date_range
    df_selection = df_selection[
        (df_selection['Pay_Period_Date'].dt.date >= start_date) &
        (df_selection['Pay_Period_Date'].dt.date <= end_date)
    ]

# --- Main Dashboard Content ---
filter_text = f"**{department}** Department" if department != "All" else "**All Departments**"
filter_text += f" | **{designation}** Role" if designation != "All" else " | **All Roles**"

st.markdown(f'<h2 class="section-header">📊 Analytics Overview - {filter_text}</h2>', unsafe_allow_html=True)

if not df_selection.empty and len(df_selection['Pay_Period_Date'].unique()) > 1:
    latest_date = df_selection['Pay_Period_Date'].unique()[0]
    previous_date = df_selection['Pay_Period_Date'].unique()[1]
    
    df_current = df_selection[df_selection['Pay_Period_Date'] == latest_date]
    df_previous = df_selection[df_selection['Pay_Period_Date'] == previous_date]

    # --- Enhanced KPI Section ---
    employees_current = df_current['Employee_ID'].nunique()
    employees_previous = df_previous['Employee_ID'].nunique()
    total_earnings_current = df_current['Total_Earnings'].sum()
    total_earnings_previous = df_previous['Total_Earnings'].sum()
    total_deductions_current = df_current['Total_Deductions'].sum()
    total_deductions_previous = df_previous['Total_Deductions'].sum()
    net_pay_current = df_current['Net_Pay'].sum()
    net_pay_previous = df_previous['Net_Pay'].sum()
    
    # KPI Cards with better formatting
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="👥 Employees Paid", 
            value=f"{employees_current:,}",
            delta=f"{employees_current - employees_previous:+,} employees"
        )
    with col2:
        delta_earnings = ((total_earnings_current - total_earnings_previous) / total_earnings_previous) * 100 if total_earnings_previous != 0 else 0
        st.metric(
            label="💰 Total Earnings", 
            value=f"₹{total_earnings_current/1_000_000:.1f}M",
            delta=f"{delta_earnings:+.1f}%"
        )
    with col3:
        delta_deductions = ((total_deductions_current - total_deductions_previous) / total_deductions_previous) * 100 if total_deductions_previous != 0 else 0
        st.metric(
            label="📉 Total Deductions", 
            value=f"₹{total_deductions_current/1_000_000:.1f}M",
            delta=f"{delta_deductions:+.1f}%"
        )
    with col4:
        delta_net_pay = ((net_pay_current - net_pay_previous) / net_pay_previous) * 100 if net_pay_previous != 0 else 0
        st.metric(
            label="💵 Net Pay", 
            value=f"₹{net_pay_current/1_000_000:.1f}M",
            delta=f"{delta_net_pay:+.1f}%"
        )

    # --- Enhanced Trend Analysis Section ---
    st.markdown('<h2 class="section-header">📈 6-Month Payroll Trends</h2>', unsafe_allow_html=True)
    
    monthly_summary = df_selection.groupby(pd.Grouper(key='Pay_Period_Date', freq='MS'))[
        ['Total_Earnings', 'Total_Deductions', 'Net_Pay']
    ].sum().reset_index()
    monthly_summary = monthly_summary.sort_values(by='Pay_Period_Date')
    
    # Create interactive Plotly chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Earnings vs Net Pay Trend', 'Monthly Deductions'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        vertical_spacing=0.15
    )
    
    # Top chart - Earnings vs Net Pay
    fig.add_trace(
        go.Scatter(
            x=monthly_summary['Pay_Period_Date'],
            y=monthly_summary['Total_Earnings'],
            name='Total Earnings',
            line=dict(color='#FF6B35', width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_summary['Pay_Period_Date'],
            y=monthly_summary['Net_Pay'],
            name='Net Pay',
            line=dict(color='#28a745', width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Bottom chart - Deductions
    fig.add_trace(
        go.Bar(
            x=monthly_summary['Pay_Period_Date'],
            y=monthly_summary['Total_Deductions'],
            name='Total Deductions',
            marker_color='#dc3545',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="",
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Enhanced Component Breakdown Section ---
    st.markdown(f'<h2 class="section-header">📊 Detailed Breakdown - {latest_date.strftime("%B %Y")}</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 💰 Earnings Components")
        earnings_data = {
            'Basic Salary': df_current['Basic_Salary'].sum(),
            'HRA': df_current['HRA'].sum(),
            'Performance Bonus': df_current['Performance_Bonus'].sum()
        }
        
        fig_earnings = go.Figure(data=[
            go.Bar(
                x=list(earnings_data.keys()),
                y=list(earnings_data.values()),
                marker_color=['#FF6B35', '#F7931E', '#FFB84D'],
                text=[f'₹{v/1_000_000:.1f}M' for v in earnings_data.values()],
                textposition='auto'
            )
        ])
        fig_earnings.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif")
        )
        fig_earnings.update_xaxes(showgrid=False)
        fig_earnings.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig_earnings, use_container_width=True)
    
    with col2:
        st.markdown("##### 📉 Deductions Breakdown")
        deductions_data = {
            'Provident Fund': df_current['Provident_Fund'].sum(),
            'Professional Tax': df_current['Professional_Tax'].sum(),
            'Income Tax': df_current['Income_Tax'].sum()
        }
        
        fig_deductions = go.Figure(data=[
            go.Bar(
                x=list(deductions_data.keys()),
                y=list(deductions_data.values()),
                marker_color=['#dc3545', '#e74c3c', '#c0392b'],
                text=[f'₹{v/1_000_000:.1f}M' for v in deductions_data.values()],
                textposition='auto'
            )
        ])
        fig_deductions.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif")
        )
        fig_deductions.update_xaxes(showgrid=False)
        fig_deductions.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig_deductions, use_container_width=True)

    # --- Enhanced Outlier Detection Section ---
    st.markdown('<h2 class="section-header">🔍 AI-Powered Outlier Detection</h2>', unsafe_allow_html=True)

    def find_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    outliers, lower_bound, upper_bound = find_outliers(df_current, 'Performance_Bonus')

    # Display outlier statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Outliers Detected", f"{len(outliers)}")
    with col2:
        st.metric("📊 Analysis Range", f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}")
    with col3:
        outlier_percentage = (len(outliers) / len(df_current)) * 100 if len(df_current) > 0 else 0
        st.metric("📈 Outlier Rate", f"{outlier_percentage:.1f}%")

    if not outliers.empty:
        st.markdown(f"""
        <div class="alert-warning">
            ⚠️ <strong>Attention Required:</strong> Found {len(outliers)} potential outlier(s) in Performance Bonuses 
            for {latest_date.strftime('%B %Y')} that fall outside the expected range.
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced outlier table
        outlier_display = outliers[['Employee_Name', 'Department', 'Designation', 'Performance_Bonus', 'Total_Earnings']].copy()
        outlier_display['Performance_Bonus'] = outlier_display['Performance_Bonus'].apply(lambda x: f"₹{x:,.0f}")
        outlier_display['Total_Earnings'] = outlier_display['Total_Earnings'].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(
            outlier_display,
            use_container_width=True,
            hide_index=True
        )
        
        # AI Analysis Button with enhanced styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🤖 Generate AI Analysis Report", key="ai_analysis", use_container_width=True):
                if api_key == "YOUR_API_KEY_HERE":
                    st.error("⚠️ Cannot generate summary. Please add your OpenAI API key to the code.")
                else:
                    with st.spinner("🤖 AI Analyst is analyzing outliers..."):
                        try:
                            outliers_str = outliers[['Employee_Name', 'Designation', 'Department', 'Performance_Bonus']].to_string(index=False)
                            
                            prompt = f"""
                            You are a senior payroll analyst for Optum India. Analyze the following payroll outliers for the leadership team.
                            
                            OUTLIER DATA FOR {latest_date.strftime('%B %Y')}:
                            {outliers_str}
                            
                            CONTEXT:
                            - Normal bonus range: ₹{lower_bound:,.0f} to ₹{upper_bound:,.0f}
                            - Total employees analyzed: {len(df_current)}
                            - Department: {department}
                            - Designation filter: {designation}
                            
                            Please provide:
                            1. Executive Summary (2-3 sentences)
                            2. Analysis by employee with potential business reasons
                            3. Risk assessment (High/Medium/Low for each)
                            4. Recommended actions for HR/Finance teams
                            
                            Format as professional bullet points. Be specific and actionable.
                            """
                            
                            response = openai.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=1000
                            )
                            ai_summary = response.choices[0].message.content
                            
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                                <h4 style="margin: 0; color: white;">🤖 AI Analyst Report</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(ai_summary)
                            
                            # Add download button for the report
                            st.download_button(
                                label="📄 Download Analysis Report",
                                data=ai_summary,
                                file_name=f"outlier_analysis_{latest_date.strftime('%Y_%m')}.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating AI analysis: {str(e)}")
    else:
        st.markdown("""
        <div class="alert-success">
            ✅ <strong>All Good:</strong> No significant outliers detected in Performance Bonuses 
            for the selected filters. All values fall within expected statistical ranges.
        </div>
        """, unsafe_allow_html=True)

    # --- Department Performance Summary ---
    if department == "All":
        st.markdown('<h2 class="section-header">🏢 Department Performance Overview</h2>', unsafe_allow_html=True)
        
        dept_summary = df_current.groupby('Department').agg({
            'Employee_ID': 'nunique',
            'Total_Earnings': ['sum', 'mean'],
            'Net_Pay': ['sum', 'mean'],
            'Performance_Bonus': 'mean'
        }).round(0)
        
        dept_summary.columns = ['Employees', 'Total Earnings', 'Avg Earnings', 'Total Net Pay', 'Avg Net Pay', 'Avg Bonus']
        dept_summary = dept_summary.reset_index()
        
        # Format currency columns
        for col in ['Total Earnings', 'Avg Earnings', 'Total Net Pay', 'Avg Net Pay', 'Avg Bonus']:
            dept_summary[col] = dept_summary[col].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(dept_summary, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div class="alert-warning">
        ⚠️ <strong>Insufficient Data:</strong> No data available for the selected filters to perform comprehensive analysis. 
        Please adjust your filter criteria or check data availability.
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Optum India Payroll Dashboard</strong> | Powered by Streamlit & OpenAI | 
    Data Analysis & Business Intelligence Platform</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">
        For technical support or feature requests, contact the Optum Analytics Team
    </p>
</div>
""", unsafe_allow_html=True)
