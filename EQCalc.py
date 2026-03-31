import streamlit as st

# Import your two separate modules
import module_2016
import module_2025

# Page config must be the very first Streamlit command in the main file
st.set_page_config(page_title="Seismic Base Shear Calculator", layout="wide")

# ==========================================
# GLOBAL CODE VERSION SWITCH (The Router)
# ==========================================
st.sidebar.header("⚙️ Code Version")
code_version = st.sidebar.radio(
    "Select IS 1893 Standard:", 
    ["IS 1893 (Part 1) : 2016", "IS 1893 (Part 1) : 2025"]
)

st.sidebar.divider()

# Route the app to the correct module based on the toggle!
if code_version == "IS 1893 (Part 1) : 2016":
    module_2016.run_2016()
elif code_version == "IS 1893 (Part 1) : 2025":
    module_2025.run_2025()
