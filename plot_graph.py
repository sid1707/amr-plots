import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pathlib
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list

# Apply page configuration
st.set_page_config(layout="wide")

# Inject custom CSS from external stylesheet
css_path = pathlib.Path("styles.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Target renaming dictionary
TARGET_RENAME = {
"cmy": "CMY", "ctxm": "CTX-M", "ent": "IC_en", "ic": "IC_ex", "imp": "IMP",
"kpc": "KPC", "mcr126": "MCR 1/2/6", "mcr3": "MCR 3", "mcr4": "MCR 4", "mcr5910": "MCR 5/9/10",
"mcr7": "MCR 7", "mcr8": "MCR 8", "meca": "mecA", "mtb": "MTB", "ndm": "NDM",
"nuc": "nuc", "oxa": "OXA", "pvl": "pvl", "shv": "SHV", "vana": "van A",
"vanb": "van B", "vanm": "van M", "vim": "VIM"
}

# Header section with banner-style layout
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
        st.sidebar.image("images/icmr_logo.png", width=120)
with col2:
    st.markdown("<div class='custom-header'><h2>AMR Surveillance Dashboard</h2></div>", unsafe_allow_html=True)
with col3:
    st.sidebar.image("images/tigs_logo.png", width=120)


# Load compiled CSV data
@st.cache_data
def load_data():
    df = pd.read_csv("compiled_data.csv")
    df["Date of collection"] = pd.to_datetime(df["Date of collection"], dayfirst=True, format='mixed', errors='coerce')
    df["Avg Cq"] = pd.to_numeric(df["Avg Cq"], errors='coerce').fillna(40)
    df["log Copy Number"] = df["Copy Number"].apply(lambda x: np.log10(x) if x > 0 else 0)
    df["Target"] = df["Target"].str.lower().map(TARGET_RENAME).fillna(df["Target"]) # Rename targets
    return df

df = load_data()

# Sidebar filters

# --- VRDL Select with dynamic "Select All" logic ---
vrdl_values = sorted(df["VRDL Name"].dropna().unique().tolist())
selected_vrdls = st.sidebar.multiselect("Select VRDL(s)", options = ["Select All"] + vrdl_values, default=["Select All"])

if "Select All" in selected_vrdls and len(selected_vrdls) > 1:
    selected_vrdls = [opt for opt in selected_vrdls if opt != "Select All"]
elif "Select All" in selected_vrdls:
    selected_vrdls = vrdl_values

# --- Date Select with same logic ---
date_values = sorted(df["Date of collection"].dt.strftime("%d-%m-%Y").dropna().unique().tolist())
date_options = ["Select All"] + date_values
selected_dates = st.sidebar.multiselect("Select Dates", options=date_options, default=["Select All"])

if "Select All" in selected_dates and len(selected_dates) > 1:
    selected_dates = [opt for opt in selected_dates if opt != "Select All"]
elif "Select All" in selected_dates:
    selected_dates = date_values

# --- Target Select ---
target_values = sorted(df["Target"].dropna().unique().tolist())
target_options = ["Select All"] + target_values
selected_targets = st.sidebar.multiselect("Select Target(s)", options=target_options, default=["Select All"])

if "Select All" in selected_targets and len(selected_targets) > 1:
    selected_targets = [opt for opt in selected_targets if opt != "Select All"]
elif "Select All" in selected_targets:
    selected_targets = target_values


# Sidebar logic
# Detect whether all options are selected
all_selected = (
    set(selected_vrdls) == set(vrdl_values) and
    set(selected_dates) == set(date_values) and
    set(selected_targets) == set(target_values)
)

# Define dynamic options
plot_options = ["Heatmap"] if all_selected else ["Bar Plot"]
value_options = ["Avg Cq", "log Copy Number"] if all_selected else ["Avg Cq", "Copy Number", "log Copy Number"]

# Use shared keys to prevent duplication
plot_type = st.sidebar.selectbox("Plot Type", plot_options, key="plot_type")
value_type = st.sidebar.selectbox("Value", value_options, key="value_type")


# Filter data based on selections
filtered_df = df[(df["VRDL Name"].isin(selected_vrdls)) &
                 (df["Date of collection"].dt.strftime("%d-%m-%Y").isin(selected_dates)) &
                 (df["Target"].isin(selected_targets))]

# --- Visualization Section ---
st.markdown("<div class='section-title'>📊 Visualizations</div>", unsafe_allow_html=True)


if plot_type == "Heatmap":
    filtered_df["VRDL+Date"] = filtered_df["VRDL Name"] + " (" + filtered_df["Date of collection"].dt.strftime("%d-%m-%Y") + ")"
    pivot = filtered_df.pivot_table(index="Target", columns="VRDL+Date", values=value_type, aggfunc="mean", fill_value=0)

    # Scale data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot)

    # Cluster rows
    row_linkage = linkage(scaled_data, method='ward')
    row_leaves = leaves_list(row_linkage)
    ordered_rows = pivot.index[row_leaves]

    # Cluster columns
    col_linkage = linkage(scaler.fit_transform(pivot.T), method='ward')
    col_leaves = leaves_list(col_linkage)
    ordered_cols = pivot.columns[col_leaves]

    # Reorder the data
    clustered_data = pivot.loc[ordered_rows, ordered_cols]
    # Prepare text labels as strings (optionally format the numbers)
    text_values = np.round(clustered_data.values, 1).astype(str)

    # Plot with Plotly
    heatmap = go.Heatmap(
        z=clustered_data.values,
        x=clustered_data.columns,
        y=clustered_data.index,
        colorscale="Viridis",
        colorbar=dict(title=value_type),
        zmin=0,
        text=text_values,
        texttemplate="%{text}",
        showscale=True   
    )

    layout = go.Layout(
        width=1400,
        height=800,
        title="Heatmap",
        plot_bgcolor="#F3F3F4",
        paper_bgcolor="#F3F3F4"
    )

    clustered_fig = go.Figure(data=[heatmap], layout=layout)
    st.plotly_chart(clustered_fig, use_container_width=True)


elif plot_type == "Bar Plot":
    bar_df = filtered_df.copy()
    bar_df["Date"] = bar_df["Date of collection"].dt.strftime("%d-%m-%Y")
    error_y_col = "Cq SD" if value_type == "Avg Cq" else "Copy SD" if value_type == "Copy Number" else None

    fig = px.bar(
        bar_df,
        x = "Target",
        y = value_type,
        color = "Date",
        facet_col = "VRDL Name",
        barmode = "group",
        error_y = error_y_col,
        height = 800,
        width = 1400
    )

    fig.update_layout(plot_bgcolor="#F3F3F4", paper_bgcolor="#F3F3F4")
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Line Plot":
    line_df = filtered_df.groupby(["Date of collection", "Target"])[value_type].mean().reset_index()
    fig = px.line(line_df, x="Date of collection", y=value_type, color="Target", markers=True, height=800, width=1400)
    fig.update_layout(plot_bgcolor="#F3F3F4", paper_bgcolor="#F3F3F4")
    st.plotly_chart(fig, use_container_width=True)

# --- Data Table Section ---
show_table = st.checkbox("Show Filtered Data Table")
if show_table:
    st.markdown("<div class='section-title'> Filtered Data Table</div>", unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)
