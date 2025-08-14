import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Load compiled CSV data
@st.cache_data
def load_data():
    df = pd.read_csv("compiled_data.csv")
    df["Date of collection"] = pd.to_datetime(df["Date of collection"], dayfirst=True, format = 'mixed', errors = 'coerce')
    df["log Copy Number"] = df["Copy Number"].apply(lambda x: np.log10(x) if x > 0 else 0)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")
selected_vrdls = st.sidebar.multiselect("Select VRDL(s)", options=df["VRDL Name"].unique(), default=df["VRDL Name"].unique())
selected_dates = st.sidebar.multiselect("Select Dates", options=df["Date of collection"].dt.strftime("%d-%m-%Y").unique(), default=df["Date of collection"].dt.strftime("%d-%m-%Y").unique())

# Filter data based on selections
filtered_df = df[(df["VRDL Name"].isin(selected_vrdls)) &
                 (df["Date of collection"].dt.strftime("%d-%m-%Y").isin(selected_dates))]

# Plot type selection
plot_type = st.selectbox("Select Plot Type", ["Heatmap", "Bar Plot", "Line Plot"])
value_type = st.radio("Select Value to Plot", ["Avg Cq", "Copy Number", "log Copy Number"])

# Plotting
if plot_type == "Heatmap":
    filtered_df["VRDL+Date"] = filtered_df["VRDL Name"] + " (" + filtered_df["Date of collection"].dt.strftime("%d-%m-%Y") + ")"
    pivot = filtered_df.pivot_table(index="Target", columns="VRDL+Date", values=value_type, aggfunc="mean", fill_value=0)
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Bar Plot":
    bar_df = filtered_df.copy()
    bar_df["Date"] = bar_df["Date of collection"].dt.strftime("%d-%m-%Y")
    fig = px.bar(bar_df, x="Target", y=value_type, color="VRDL Name", facet_col="Date", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Line Plot":
    line_df = filtered_df.groupby(["Date of collection", "Target"])[value_type].mean().reset_index()
    fig = px.line(line_df, x="Date of collection", y=value_type, color="Target", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# Optionally show data table
show_table = st.checkbox("Show Filtered Data Table")
if show_table:
    st.write("### Filtered Data", filtered_df)
