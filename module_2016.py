import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# --- IMPORT OUR CUSTOM MODULE ---
from ai_extractor import elements_popup, process_drawings

def run_2016():
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

    # --- Constants for Physics Engine ---
    E_concrete = 25000000.0  # kN/m^2 (M25 approx)
    density_rc = 25.0        # kN/m^3

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
    if 'success_msg' not in st.session_state: st.session_state.success_msg = ""
    if 'floors' not in st.session_state: st.session_state.floors = []

    valid_soil_types = ["Hard (Type I)", "Medium (Type II)", "Soft (Type III)"]
    valid_struct_types = ["RC Bare Frame", "RC Frame with Masonry Infill", "Steel Frame"]

    st.set_page_config(page_title="Seismic Base Shear Calculator", layout="wide")
    st.title("IS 1893 Seismic Base Shear Calculator (ESM & RSM)")

    # ==========================================
    # SIDEBAR: AI EXTRACTION TRIGGER
    # ==========================================
    st.sidebar.header("AI Drawing Reader")
    uploaded_files = st.sidebar.file_uploader("Upload Structural Drawings (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Use st.secrets or user input for API key depending on your deployment
    api_key = st.secrets["GROQ_API_KEY"] 

    if uploaded_files:
        for file in uploaded_files: 
            st.sidebar.image(file, caption=file.name, width="stretch")
        if st.session_state.success_msg: 
            st.sidebar.success(st.session_state.success_msg)
        
        if st.sidebar.button("Extract Data") and api_key:
            with st.spinner(f"Scanning {len(uploaded_files)} drawings and schedules..."):
                # Call the imported function!
                process_drawings(uploaded_files, api_key)

    # ==========================================
    # SEISMIC PARAMETERS & UI
    # ==========================================
    st.sidebar.header("Seismic Parameters")

    # --- 1. Helper Functions for Clean UI ---
    def info_number_input(label, min_val, step, key, info_title, info_body, format_str=None, images=None):
        """Creates a number input with a perfectly aligned info popover and optional images."""
        col1, col2 = st.sidebar.columns([0.85, 0.15])
        with col1:
            result = st.number_input(label, min_value=min_val, step=step, format=format_str, key=key)
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            with st.popover("ℹ️"):
                st.markdown(f"**{info_title}**")
                st.markdown(info_body)
                # Safely check if a list of images was provided
                if images:
                    for img in images:
                        st.image(img, use_container_width=True)
        return result

    def info_selectbox(label, options, key, info_title, info_body, images=None):
        """Creates a selectbox with a perfectly aligned info popover and optional images."""
        col1, col2 = st.sidebar.columns([0.85, 0.15])
        with col1:
            result = st.selectbox(label, options, key=key)
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            with st.popover("ℹ️"):
                st.markdown(f"**{info_title}**")
                st.markdown(info_body)
                # Safely check if a list of images was provided
                if images:
                    for img in images:
                        st.image(img, use_container_width=True)
        return result

    # --- 2. Render the Inputs ---
    # Notice we pass the full file names in a list!
    zone_factor = info_number_input(
        label="Zone Factor (Z)", min_val=0.01, step=0.01, format_str="%.2f", key="z_val",
        info_title="", 
        info_body="",
        images=["Zone_factor.png"] # You can pass 1, 2, or 100 images here!
    )

    importance_factor = info_number_input(
        label="Importance Factor (I)", min_val=1.0, step=0.1, key="i_val",
        info_title="", 
        info_body="",
        images = ["impFactor.png", "impFactor_notes.png"]
    )

    response_reduction = info_number_input(
        label="Response Reduction Factor (R)", min_val=1.0, step=0.5, key="r_val",
        info_title="", 
        info_body="",
        images=["Resp_Red1.png", "Resp_Red2.png", "Resp_Red3.png", "Resp_Red4.png"]
    )

    soil_type = info_selectbox(
        label="Soil Type", options=valid_soil_types, key="soil_val",
        info_title="Soil Classification", 
        info_body="Determines the spectral acceleration (Sa/g) curve.",
        images = ["soil_type.png"]
    )

    structure_type = info_selectbox(
        label="Structure Type for Ta", options=valid_struct_types, key="struct_val",
        info_title="Fundamental Time Period", 
        info_body="Dictates the empirical formula used to calculate Ta."
    )

    st.header("Step 1: Building Geometry & Direction")
    col_geo1, col_geo2, col_geo3 = st.columns(3)
    h = col_geo1.number_input("Building Height (h)", min_value=3.0, key="h_val")
    dx = col_geo2.number_input("Base X-Dimension (dx)", min_value=3.0, key="dx_val")
    dy = col_geo3.number_input("Base Y-Dimension (dy)", min_value=3.0, key="dy_val")

    eq_direction = st.radio("Select Earthquake Direction for Analysis:", ["X-Direction", "Y-Direction"], horizontal=True)
    d = dx if eq_direction == "X-Direction" else dy
    num_stories = st.number_input("Number of Stories", min_value=1, max_value=50, key="stories_val")

    # ==========================================
    # DYNAMIC MATRIX GENERATION (Physics Engine Integration)
    # ==========================================
    st.divider()
    st.subheader("Structural Element Editor")
    st.write("Click below to adjust member sizes. The table below will update automatically based on structural dynamics principles.")

    if st.button("🔍 Open Element Viewer & Editor", type="primary"):
        # Call the imported UI popup!
        elements_popup()

    dl_list, ll_list, k_list, height_from_base_list = [], [], [], []
    cumulative_height = 0.0

    if 'floors' in st.session_state and len(st.session_state.floors) == num_stories:
        floors = st.session_state.floors
        for i, floor in enumerate(floors):
            sh = float(floor.get("story_height", h / num_stories))
            ll = float(floor.get("live_load", 300.0))
            cumulative_height += sh
            
            # 1. Floor Mass
            slab_vol = sum([float(s.get("thickness", 0)) * float(s.get("total_area", 0)) for s in floor.get("slabs", [])])
            beam_vol = sum([float(b.get("b", 0)) * float(b.get("d", 0)) * float(b.get("total_length", 0)) for b in floor.get("beams", [])])
            floor_weight = (slab_vol + beam_vol) * density_rc

            # 2. Vertical Elements (This story)
            col_vol = sum([float(c.get("b", 0)) * float(c.get("d", 0)) * sh * float(c.get("count", 0)) for c in floor.get("columns", [])])
            sw_vol = sum([float(sw.get("length", 0)) * float(sw.get("thickness", 0)) * sh * float(sw.get("count", 0)) for sw in floor.get("shear_walls", [])])
            vert_wt_current = (col_vol + sw_vol) * density_rc
            
            # 3. Vertical Elements (Story Above)
            vert_wt_above = 0.0
            if i < num_stories - 1:
                floor_above = floors[i+1]
                sh_above = float(floor_above.get("story_height", 3.0))
                col_vol_above = sum([float(c.get("b", 0)) * float(c.get("d", 0)) * sh_above * float(c.get("count", 0)) for c in floor_above.get("columns", [])])
                vert_wt_above = (col_vol_above) * density_rc

            lumped_dl = floor_weight + (0.5 * vert_wt_current) + (0.5 * vert_wt_above)
            
            # 4. Stiffness
            total_k = 0.0
            for c in floor.get("columns", []):
                b_val = float(c.get("b", 0))
                d_val = float(c.get("d", 0))
                eff_d = b_val if eq_direction == "X-Direction" else d_val 
                eff_b = d_val if eq_direction == "X-Direction" else b_val
                
                if sh > 0:
                    I = (eff_b * (eff_d**3)) / 12.0
                    total_k += ((12 * E_concrete * I) / (sh**3)) * float(c.get("count", 0))

            dl_list.append(lumped_dl)
            ll_list.append(ll)
            k_list.append(total_k)
            height_from_base_list.append(cumulative_height)
            
        dl_list.reverse()
        ll_list.reverse()
        k_list.reverse()
        height_from_base_list.reverse()
    else:
        # Fallback Data
        dl_list = [1000.0] * num_stories
        ll_list = [300.0] * num_stories
        k_list = [50000.0] * num_stories
        height_from_base_list = [h - (i * (h/num_stories)) for i in range(num_stories)]

    df_input = pd.DataFrame({
        "Story": [f"Story {num_stories - i}" for i in range(num_stories)],
        "Dead Load (kN)": dl_list,
        "Live Load (kN)": ll_list,
        "Stiffness (kN/m)": k_list,
        "Height from Base (m)": height_from_base_list
    })

    st.write("Calculated Global Matrix (Editable):")
    df_edited = st.data_editor(df_input, width="stretch")

    if st.button("Run Seismic Analysis"):
        df_calc = df_edited.iloc[::-1].reset_index(drop=True)
        W_array = df_calc["Dead Load (kN)"].values + df_calc["Live Load (kN)"].values
        mass_array = W_array / 9.81  
        K_array = df_calc["Stiffness (kN/m)"].values
        H_array = df_calc["Height from Base (m)"].values
        total_W = np.sum(W_array)
        
        st.subheader(f"Total Seismic Weight (W): {total_W:.2f} kN")

        # --- ESM ---
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

        # --- RSM ---
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

        # --- BASE SHEAR SCALING ---
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
            
        # ==========================================
        # STEP 5: VERTICAL SHEAR DISTRIBUTION
        # ==========================================
        st.header("Step 5: Vertical Shear Distribution (ESM)")
        
        # 1. Calculate the math required for the graphs
        W_h2 = W_array * (H_array ** 2)
        sum_Wh2 = np.sum(W_h2)
        Q_i_stat = VB_stat * (W_h2 / sum_Wh2)
        V_i_stat = np.cumsum(Q_i_stat[::-1])[::-1]
        
        # 2. Build DataFrame
        df_shear = pd.DataFrame({
            "Story": [("Ground Floor" if j == 0 else f"Floor {j}") for j in range(num_stories)][::-1],
            "Height h_i (m)": H_array[::-1],
            "Weight W_i (kN)": W_array[::-1],
            "W_i * h_i^2": W_h2[::-1],
            "Lateral Force Q_i (kN)": Q_i_stat[::-1],
            "Storey Shear V_i (kN)": V_i_stat[::-1]
        })
        
        st.table(df_shear.style.format({
            "Height h_i (m)": "{:.2f}",
            "Weight W_i (kN)": "{:.2f}",
            "W_i * h_i^2": "{:.2f}",
            "Lateral Force Q_i (kN)": "{:.2f}",
            "Storey Shear V_i (kN)": "{:.2f}"
        }))

        # ==========================================
        # STEP 6: VISUALIZATIONS: MODE SHAPES & LUMPED MASS
        # ==========================================
        st.divider()
        st.header("Step 6: Visualizations")
        col_plot1, col_plot2 = st.columns(2)

        # --- LEFT COLUMN: MODE SHAPES ---
        with col_plot1:
            st.subheader("First 3 Mode Shapes")
            fig_modes, ax_modes = plt.subplots(figsize=(5, max(6, int(num_stories * 1.2))))
            y_coords = np.concatenate(([0], H_array))
            
            for i in range(min(3, num_stories)): 
                x_coords = np.concatenate(([0], eigenvectors[:, i] * modal_participation[i]))
                ax_modes.plot(x_coords, y_coords, marker='o', label=f'Mode {i+1} (T={time_periods[i]:.2f}s)')
                
            ax_modes.axvline(0, color='black', linestyle='--')
            ax_modes.set_ylabel("Height from Base (m)")
            ax_modes.set_xlabel("Modal Displacement")
            ax_modes.legend()
            st.pyplot(fig_modes)

        # --- RIGHT COLUMN: LUMPED MASS MODEL ---
        with col_plot2:
            st.subheader("Vertical Shear Distribution")
            fig_lumped, ax_lumped = plt.subplots(figsize=(5, max(6, int(num_stories * 1.2))))
            
            # Ground and Stick
            ax_lumped.plot([-2, 2], [0, 0], color='black', linewidth=4, zorder=1)
            ax_lumped.plot([0, 0], [0, H_array[-1]], color='gray', linewidth=3, zorder=2)
            
            max_Q = np.max(Q_i_stat) if np.max(Q_i_stat) > 0 else 1
            max_W = np.max(W_array) if np.max(W_array) > 0 else 1
            
            for i in range(num_stories):
                # Draw Mass
                m_size = 10 + (W_array[i] / max_W) * 20
                ax_lumped.plot(0, H_array[i], marker='o', markersize=m_size, color='#0068c9', zorder=4)
                
                # Draw Force Arrow
                arrow_len = (Q_i_stat[i] / max_Q) * 2.5
                if arrow_len > 0:
                    ax_lumped.arrow(0, H_array[i], arrow_len, 0, 
                                    head_width=H_array[-1]/40, head_length=0.2, 
                                    fc='red', ec='red', zorder=3, length_includes_head=True)
                
                # Labels
                ax_lumped.text(-0.3, H_array[i], f"{W_array[i]:.0f} kN", va='center', ha='right', fontsize=10)
                ax_lumped.text(arrow_len + 0.1, H_array[i], f"Q = {Q_i_stat[i]:.1f} kN", va='center', ha='left', color='red', fontsize=10, weight='bold')

            ax_lumped.set_xlim(-3, 4)
            ax_lumped.set_ylim(-H_array[-1]*0.05, H_array[-1]*1.1)
            ax_lumped.axis('off')
            st.pyplot(fig_lumped)
