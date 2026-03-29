import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import json
import base64
from groq import Groq

# --- IS 1893 Response Spectrum Function ---
def get_sa_g(T, soil_type):
    if T == 0: return 1.0
    if soil_type == "Hard (Type I)":
        if T < 0.1: return 1 + 15 * T
        elif T <= 0.4: return 2.5
        else: return 1.0 / T
    elif soil_type == "Medium (Type II)":
        if T < 0.1: return 1 + 15 * T
        elif T <= 0.55: return 2.5
        else: return 1.36 / T
    else: 
        if T < 0.1: return 1 + 15 * T
        elif T <= 0.67: return 2.5
        else: return 1.67 / T

# --- Initialize Session State ---
if 'h_val' not in st.session_state: st.session_state.h_val = 12.0
if 'dx_val' not in st.session_state: st.session_state.dx_val = 15.0
if 'dy_val' not in st.session_state: st.session_state.dy_val = 8.0
if 'stories_val' not in st.session_state: st.session_state.stories_val = 4

if 'z_val' not in st.session_state: st.session_state.z_val = 0.36
if 'i_val' not in st.session_state: st.session_state.i_val = 1.2
if 'r_val' not in st.session_state: st.session_state.r_val = 5.0
if 'soil_val' not in st.session_state: st.session_state.soil_val = "Medium (Type II)"
if 'struct_val' not in st.session_state: st.session_state.struct_val = "RC Bare Frame"

# We create a specific state variable to hold the AI's success message
if 'success_msg' not in st.session_state: st.session_state.success_msg = ""

valid_soil_types = ["Hard (Type I)", "Medium (Type II)", "Soft (Type III)"]
valid_struct_types = ["RC Bare Frame", "RC Frame with Masonry Infill", "Steel Frame"]
valid_zones = [0.10, 0.16, 0.24, 0.36]

st.set_page_config(page_title="Seismic Base Shear Calculator", layout="wide")
st.title("IS 1893 Seismic Base Shear Calculator (ESM & RSM)")

# ==========================================
# 🤖 REAL AI VISION EXTRACTION
# ==========================================
st.sidebar.header("🤖 AI Drawing Reader")
st.sidebar.markdown("Upload a Plan/Elevation to auto-extract dimensions and seismic notes.")

# Enable multiple files
uploaded_files = st.sidebar.file_uploader("Upload Structural Drawings (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if uploaded_files: # This is now a list!
    # Display all uploaded images in the sidebar
    for file in uploaded_files:
        st.sidebar.image(file, caption=file.name, use_container_width=True)
    
    # Display the success message if it exists in the session state
    if st.session_state.success_msg:
        st.sidebar.success(st.session_state.success_msg)
    
    if st.sidebar.button("Extract Data with Llama 4 Scout") and api_key:
        with st.spinner(f"Scanning {len(uploaded_files)} drawings and notes..."):
            
            # Slightly tweaked prompt to mention multiple drawings
            prompt = """
            You are an expert structural engineer analyzing the provided architectural/structural drawings and notes.
            Extract the parameters into the exact JSON structure below.

            CRITICAL RULES:
            1. Exercise engineering judgment: Look across all provided images. Base dimensions are usually in plans, height/stories in elevations. Calculate total dimensions by summing grid lines, derive total height from story heights, and convert distributed area loads (kN/m²) to lumped floor loads (kN) using the plan area (base_x * base_y).
            2. If a value is missing across ALL images and cannot be logically derived, return `null`. Do not hallucinate.
            3. Values of dimensions are required in meters. If the images have dimension values in 1000s (i.e. mm), convert them to meter.
            4. `floor_data` must contain exactly `num_stories` objects, ordered from Bottom (Story 1) to Top.
            5. Respond ONLY with raw JSON. No markdown blocks, no explanations.

            {
              "height": null, 
              "base_x": null, 
              "base_y": null, 
              "num_stories": null, 
              "z": null, 
              "i": null, 
              "r": null, 
              "soil_type": null, 
              "structure_type": null, 
              "floor_data": [
                {
                  "dl": null, 
                  "ll": null, 
                  "k": null, 
                  "h": null  
                }
              ]
            }
            """
            
            # Build the dynamic payload
            content_payload = [{"type": "text", "text": prompt}]
            
            # Loop through all uploaded files, convert them, and add them to the payload
            for file in uploaded_files:
                base64_image = base64.b64encode(file.getvalue()).decode('utf-8')
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": content_payload # Pass the dynamic list containing text + ALL images
                        }
                    ],
                    temperature=0.1
                )
                
                raw_text = response.choices[0].message.content
                clean_text = raw_text.replace('```json', '').replace('```', '').strip()
                extracted_data = json.loads(clean_text)

                
                # Update Session State
                if extracted_data.get("height"): st.session_state.h_val = float(extracted_data["height"])
                if extracted_data.get("base_x"): st.session_state.dx_val = float(extracted_data["base_x"])
                if extracted_data.get("base_y"): st.session_state.dy_val = float(extracted_data["base_y"])
                if extracted_data.get("num_stories"): st.session_state.stories_val = int(extracted_data["num_stories"])
                if extracted_data.get("z") is not None: st.session_state.z_val = float(extracted_data["z"])
                if extracted_data.get("i"): st.session_state.i_val = float(extracted_data["i"])
                if extracted_data.get("r"): st.session_state.r_val = float(extracted_data["r"])
                if extracted_data.get("soil_type") in valid_soil_types: st.session_state.soil_val = extracted_data["soil_type"]
                if extracted_data.get("structure_type") in valid_struct_types: st.session_state.struct_val = extracted_data["structure_type"]
                
                if extracted_data.get("floor_data") and isinstance(extracted_data["floor_data"], list):
                    # Save the raw AI matrix to the session state
                    st.session_state.ai_matrix = extracted_data["floor_data"]
                    
                    # If the AI didn't find overall height, let's calculate it from the sum of storey heights
                    if not extracted_data.get("height"):
                        calc_height = sum([float(f.get("h", 0) or 0) for f in extracted_data["floor_data"]])
                        if calc_height > 0:
                            st.session_state.h_val = calc_height
                
                found_items = {k: v for k, v in extracted_data.items() if v is not None}
                
                # Save the message to state, then RERUN to perfectly sync the UI!
                st.session_state.success_msg = f"✅ Found overall params and structural matrix for {len(extracted_data.get('floor_data', []))} floors!"
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

# ==========================================
# SEISMIC PARAMETERS & UI (Bound by Keys)
# ==========================================
st.sidebar.header("Seismic Parameters")

# Notice how we removed 'value' and 'index' and replaced them with 'key'. 
# This ties the input directly to the st.session_state dictionary!
zone_factor = st.sidebar.number_input("Zone Factor (Z)", min_value=0.01, step=0.01, format="%.2f", key="z_val")
importance_factor = st.sidebar.number_input("Importance Factor (I)", min_value=1.0, step=0.1, key="i_val")
response_reduction = st.sidebar.number_input("Response Reduction Factor (R)", min_value=1.0, step=0.5, key="r_val")
soil_type = st.sidebar.selectbox("Soil Type", valid_soil_types, key="soil_val")
structure_type = st.sidebar.selectbox("Structure Type for Ta", valid_struct_types, key="struct_val")

# MAIN PAGE
st.header("Step 1: Building Geometry & Direction")
col_geo1, col_geo2, col_geo3 = st.columns(3)
h = col_geo1.number_input("Building Height (h) in meters", min_value=3.0, key="h_val")
dx = col_geo2.number_input("Base X-Dimension (dx)", min_value=3.0, key="dx_val")
dy = col_geo3.number_input("Base Y-Dimension (dy)", min_value=3.0, key="dy_val")

eq_direction = st.radio("Select Earthquake Direction for Analysis:", ["X-Direction", "Y-Direction"], horizontal=True)
d = dx if eq_direction == "X-Direction" else dy
st.info(f"Using effective base dimension **d = {d}m** for Time Period (Ta) calculation.")

num_stories = st.number_input("Number of Stories", min_value=1, max_value=50, key="stories_val")

# ==========================================
# CALCULATION LOGIC
# ==========================================
# ==========================================
# DYNAMIC MATRIX GENERATION
# ==========================================
# Initialize lists to hold table data
dl_list, ll_list, k_list, height_from_base_list = [], [], [], []
cumulative_height = 0.0

# Check if AI extracted matrix data AND if the number of stories matches
if 'ai_matrix' in st.session_state and len(st.session_state.ai_matrix) == num_stories:
    # Read from Bottom (Story 1) to Top (Story N)
    for floor in st.session_state.ai_matrix:
        dl = float(floor.get("dl")) if floor.get("dl") is not None else 1000.0
        ll = float(floor.get("ll")) if floor.get("ll") is not None else 300.0
        k = float(floor.get("k")) if floor.get("k") is not None else 50000.0
        sh = float(floor.get("h")) if floor.get("h") is not None else (h / num_stories)
        
        cumulative_height += sh
        dl_list.append(dl)
        ll_list.append(ll)
        k_list.append(k)
        height_from_base_list.append(cumulative_height)
        
    # The UI Table displays Top-Down (Story N down to 1), so we must reverse the lists
    dl_list.reverse()
    ll_list.reverse()
    k_list.reverse()
    height_from_base_list.reverse()

else:
    # Fallback: If no AI data, or user manually changed the number of stories, use standard defaults
    dl_list = [1000.0] * num_stories
    ll_list = [300.0] * num_stories
    k_list = [50000.0] * num_stories
    height_from_base_list = [h - (i * (h/num_stories)) for i in range(num_stories)]

# Assemble the final dataframe
default_data = {
    "Story": [f"Story {num_stories - i}" for i in range(num_stories)],
    "Dead Load (kN)": dl_list,
    "Live Load (kN)": ll_list,
    "Stiffness (kN/m)": k_list,
    "Height from Base (m)": height_from_base_list
}

df_input = pd.DataFrame(default_data)
st.write("Edit the structural properties per floor. For the roof, set Live Load to 0 as per IS 1893.")
df_edited = st.data_editor(df_input, use_container_width=True)
if st.button("Run Seismic Analysis"):
    df_calc = df_edited.iloc[::-1].reset_index(drop=True)
    W_array = df_calc["Dead Load (kN)"].values + df_calc["Live Load (kN)"].values
    mass_array = W_array / 9.81  
    K_array = df_calc["Stiffness (kN/m)"].values
    H_array = df_calc["Height from Base (m)"].values
    total_W = np.sum(W_array)
    
    st.subheader(f"Total Seismic Weight (W): {total_W:.2f} kN")

    # ESM
    st.header("Step 2: Equivalent Static Method (ESM)")
    if structure_type == "RC Bare Frame": Ta = 0.075 * (h ** 0.75)
    elif structure_type == "RC Frame with Masonry Infill": Ta = 0.09 * h / np.sqrt(d)
    else: Ta = 0.085 * (h ** 0.75)

    sa_g_stat = get_sa_g(Ta, soil_type)
    Ah_stat = (zone_factor / 2) * (importance_factor / response_reduction) * sa_g_stat
    VB_stat = Ah_stat * total_W
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Time Period (Ta)", f"{Ta:.3f} s")
    col2.metric("Sa/g", f"{sa_g_stat:.3f}")
    col3.metric("Ah", f"{Ah_stat:.4f}")
    col4.metric("Static Base Shear (VB)", f"{VB_stat:.2f} kN")

    # RSM
    st.header("Step 3: Dynamic Analysis (RSM)")
    M_mat = np.diag(mass_array)
    K_mat = np.zeros((num_stories, num_stories))
    
    for i in range(num_stories):
        K_mat[i, i] = K_array[i]
        if i < num_stories - 1:
            K_mat[i, i] += K_array[i+1]
            K_mat[i, i+1] = -K_array[i+1]
            K_mat[i+1, i] = -K_array[i+1]
            
    eigenvalues, eigenvectors = eigh(K_mat, M_mat)
    omega = np.sqrt(eigenvalues)
    time_periods = 2 * np.pi / omega
    
    modal_mass = np.zeros(num_stories)
    modal_participation = np.zeros(num_stories)
    modal_base_shear = np.zeros(num_stories)
    r_vec = np.ones(num_stories) 
    
    for i in range(num_stories):
        phi_i = eigenvectors[:, i]
        Mn = np.dot(phi_i.T, np.dot(M_mat, phi_i))
        Ln = np.dot(phi_i.T, np.dot(M_mat, r_vec))
        Pk = Ln / Mn
        Mk = (Ln ** 2) / Mn
        modal_participation[i] = Pk
        modal_mass[i] = Mk
        
        sa_g_dyn = get_sa_g(time_periods[i], soil_type)
        Ah_dyn = (zone_factor / 2) * (importance_factor / response_reduction) * sa_g_dyn
        modal_base_shear[i] = Ah_dyn * Mk * 9.81
    
    df_modes = pd.DataFrame({
        "Mode": [f"Mode {i+1}" for i in range(num_stories)],
        "Time Period (s)": time_periods,
        "Modal Mass (tonnes)": modal_mass,
        "Mass Participation (%)": (modal_mass / np.sum(mass_array)) * 100,
        "Modal Base Shear (kN)": np.abs(modal_base_shear)
    })
    st.table(df_modes.style.format({"Time Period (s)": "{:.3f}", "Mass Participation (%)": "{:.2f}%", "Modal Base Shear (kN)": "{:.2f}"}))
    
    VB_dyn = np.sqrt(np.sum(modal_base_shear**2))
    st.markdown(f"**Dynamic Base Shear (SRSS combination):** `{VB_dyn:.2f} kN`")

    # BASE SHEAR SCALING
    st.header("Step 4: Base Shear Scaling")
    scale_factor = VB_stat / VB_dyn if VB_dyn < VB_stat else 1.0
    
    if scale_factor > 1.0:
        st.warning(f"VB_dyn ({VB_dyn:.2f} kN) < VB_stat ({VB_stat:.2f} kN). Scaling is required!")
        st.metric("Scale Factor (VB_stat / VB_dyn)", f"{scale_factor:.4f}")
        st.success(f"Design Base Shear (Scaled): {VB_dyn * scale_factor:.2f} kN")
    else:
        st.success(f"VB_dyn ({VB_dyn:.2f} kN) >= VB_stat ({VB_stat:.2f} kN). No scaling required.")
        st.metric("Scale Factor", "1.000")
        st.success(f"Design Base Shear: {VB_dyn:.2f} kN")
        
    st.subheader("Mode Shapes")
    fig, ax = plt.subplots(figsize=(6, 8))
    y_coords = np.concatenate(([0], H_array))
    for i in range(min(3, num_stories)): 
        x_coords = np.concatenate(([0], eigenvectors[:, i] * modal_participation[i]))
        ax.plot(x_coords, y_coords, marker='o', label=f'Mode {i+1} (T={time_periods[i]:.2f}s)')
        
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title("First 3 Mode Shapes")
    ax.set_ylabel("Height from Base (m)")
    ax.set_xlabel("Modal Displacement")
    ax.legend()
    st.pyplot(fig)
