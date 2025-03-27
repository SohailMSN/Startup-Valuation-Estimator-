import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Page Configuration
st.set_page_config(page_title="🚀 Startup Valuation Estimator", page_icon="💡", layout="wide")

# Custom Styling with Sidebar Animation
st.markdown("""
    <style>
        body {background-color: black; color: white;}
        .stButton > button {background-color: #ff4757; color: white; border-radius: 8px; transition: 0.3s;}
        .stButton > button:hover {background-color: #e84118; transform: scale(1.05);}
        .stSidebar {background-color: #1e1e1e; color: white;}
        .title-text {text-align: center; font-size: 30px; font-weight: bold; color: #ffa502;}
        .description {text-align: center; font-size: 18px; color: #dcdde1;}
        
        @keyframes sidebarGlow {
            0% {box-shadow: 0px 0px 5px #ff4757;}
            50% {box-shadow: 0px 0px 20px #ffa502;}
            100% {box-shadow: 0px 0px 5px #ff4757;}
        }
        
        .sidebar-animated {
            padding: 10px;
            border-radius: 10px;
            animation: sidebarGlow 2s infinite alternate;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with Animated Box
st.sidebar.markdown("""
    <div class='sidebar-animated'>
        <h2 style='text-align: center; color: #ffa502;'>🚀 AI Valuation</h2>
        <p style='text-align: center; color: #dcdde1;'>Predict the future of your startup with advanced AI analytics!</p>
    </div>
""", unsafe_allow_html=True)

# User Inputs
st.sidebar.header("🔍 Enter Startup Details")
revenue = st.sidebar.number_input("Annual Revenue ($M)", min_value=0.0, value=5.0)
growth_rate = st.sidebar.slider("Growth Rate (%)", 1, 100, 20)
profit_margin = st.sidebar.slider("Profit Margin (%)", 1, 50, 15)
industry_multiple = st.sidebar.slider("Industry Revenue Multiple", 1.0, 20.0, 5.0)
discount_rate = st.sidebar.slider("Discount Rate (%)", 5, 20, 10)
years = 5

# Revenue Projection
future_revenue = [revenue * (1 + growth_rate / 100) ** i for i in range(years)]
df_projection = pd.DataFrame({
    "Year": [f"Year {i + 1}" for i in range(years)],
    "Projected Revenue ($M)": future_revenue
})

# AI-Powered Prediction
x = np.arange(1, years + 1).reshape(-1, 1)
y = np.array(future_revenue).reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
model = LinearRegression().fit(x_poly, y)
predicted_revenue = model.predict(poly.transform(x))
df_projection["AI-Predicted Revenue ($M)"] = predicted_revenue.flatten()

# Valuation Calculation
valuation_dcf = sum(future_revenue[i] / (1 + discount_rate / 100) ** i for i in range(years))
valuation_multiple = revenue * industry_multiple

# Visualizations
fig1 = px.line(df_projection, x="Year", y=["Projected Revenue ($M)", "AI-Predicted Revenue ($M)"], markers=True,
               title="📈 Revenue Growth Projection")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(df_projection, x="Year", y="Projected Revenue ($M)", title="📊 Revenue Over Years")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.histogram(np.random.normal(valuation_dcf, valuation_dcf * 0.2, 1000), nbins=50,
                    title="📊 Monte Carlo Simulation: Valuation Distribution")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.pie(df_projection, values="Projected Revenue ($M)", names="Year", title="📊 Revenue Share Over Time")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.scatter(df_projection, x="Year", y="Projected Revenue ($M)", title="📌 Revenue Scatter Plot")
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.box(np.random.normal(valuation_dcf, valuation_dcf * 0.2, 1000), title="📦 Valuation Spread Analysis")
st.plotly_chart(fig6, use_container_width=True)

fig7 = px.funnel(df_projection, x="Year", y="Projected Revenue ($M)", title="📥 Revenue Funnel")
st.plotly_chart(fig7, use_container_width=True)

fig8 = px.area(df_projection, x="Year", y="Projected Revenue ($M)", title="📊 Revenue Trend Area Chart")
st.plotly_chart(fig8, use_container_width=True)

# Glowing Text Animation for Valuation
st.markdown(f"""
    <style>
        @keyframes glow {{
            0% {{text-shadow: 0 0 5px #ff9f43, 0 0 10px #ff4757;}}
            50% {{text-shadow: 0 0 15px #ff6b81, 0 0 20px #ff4757;}}
            100% {{text-shadow: 0 0 5px #ff9f43, 0 0 10px #ff4757;}}
        }}
        .glowing-text {{
            animation: glow 2s infinite alternate;
            font-size: 25px;
            text-align: center;
            color: #2ed573;
        }}
    </style>
    <h2 class='glowing-text'>💰 Your Startup's Estimated Valuation: ${valuation_multiple:,.2f}M - ${valuation_dcf:,.2f}M</h2>
""", unsafe_allow_html=True)

# AI Chatbot
st.subheader("🤖 AI-Powered Startup Advisor")
st.text("Get AI-driven insights for your startup. Ask anything about valuation, funding, and growth strategies!")
user_input = st.text_input("Ask AI anything about your startup:")
if user_input:
    st.success("AI Response: Your startup has a bright future! 🚀 Keep innovating and growing.")

# Fun Interactive Feature
if st.button("🔥 Boost My Valuation! 💰"):
    st.balloons()
    st.success("Your startup is now worth 10x more... Just kidding! Keep pushing forward! 💡")

# Download Options
csv_projection = df_projection.to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Download Valuation Report", data=csv_projection, file_name="valuation_report.csv",
                   mime="text/csv")

csv_simulation = pd.DataFrame({"Simulated Valuation ($M)": np.random.normal(valuation_dcf, valuation_dcf * 0.2, 1000)}
                              ).to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Download Monte Carlo Simulation Data", data=csv_simulation,
                   file_name="monte_carlo_simulation.csv", mime="text/csv")
