import streamlit as st
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from ai_extractor import elements_popup, process_drawings

# --- IS 1893:2025 Response Spectrum Function (A_NH) ---
def get_sa_g_2025(T, site_class, method="ESM"):
    """2025 Code explicitly separates ESM and RSM spectral curves at low periods"""
    if method == "ESM":
        if site_class == "Site Classes A & B (Hard)":
            if T <= 0.4: return 2.5
            elif T <= 6.0: return 1.0 / T
            else: return 6.0 / (T**2)
        elif site_class == "Site Class C (Medium)":
            if T <= 0.6: return 2.5
            elif T <= 6.0: return 1.5 / T
            else: return 9.0 / (T**2)
        else: # Site Class D (Soft)
            if T <= 0.8: return 2.5
            elif T <= 6.0: return 2.0 / T
            else: return 12.0 / (T**2)
    else: # RSM
        if site_class == "Site Classes A & B (Hard)":
            if T <= 0.1: return 1.0 + 15*T
            elif T <= 0.4: return 2.5
            elif T <= 6.0: return 1.0 / T
            else: return 6.0 / (T**2)
        elif site_class == "Site Class C (Medium)":
            if T <= 0.1: return 1.0 + 15*T
            elif T <= 0.6: return 2.5
            elif T <= 6.0: return 1.5 / T
            else: return 9.0 / (T**2)
        else: # Site Class D (Soft)
            if T <= 0.1: return 1.0 + 15*T
            elif T <= 0.8: return 2.5
            elif T <= 6.0: return 2.0 / T
            else: return 12.0 / (T**2)

# --- Helper Functions for Clean UI ---
def info_number_input(label, min_val, step, key, info_title, info_body, format_str=None, images=None):
    col1, col2 = st.sidebar.columns([0.85, 0.15])
    with col1: result = st.number_input(label, min_value=min_val, step=step, format=format_str, key=key)
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        with st.popover("ℹ️"):
            st.markdown(f"**{info_title}**")
            st.markdown(info_body)
            if images:
                for img in images: st.image(img, use_container_width=True)
    return result

def info_selectbox(label, options, key, info_title, info_body, images=None):
    col1, col2 = st.sidebar.columns([0.85, 0.15])
    with col1: result = st.selectbox(label, options, key=key)
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        with st.popover("ℹ️"):
            st.markdown(f"**{info_title}**")
            st.markdown(info_body)
            if images:
                for img in images: st.image(img, use_container_width=True)
    return result

def run_2025():
    st.title("IS 1893:2025 Seismic Base Shear Calculator")
    st.success("Running **2025** Physics Engine: Advanced T_a formulation & True PGA Zone Factors.")

    # --- Constants ---
    E_concrete = 25000000.0  
    density_rc = 25.0        

    valid_site_classes = ["Site Classes A & B (Hard)", "Site Class C (Medium)", "Site Class D (Soft)"]
    valid_struct_types = ["RC MRF (with/without walls or infill)", "Steel MRF", "Other"]

    # ==========================================
    # SIDEBAR
    # ==========================================
    st.sidebar.header("AI Drawing Reader")
    uploaded_files = st.sidebar.file_uploader("Upload Structural Drawings (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="up_25")
    api_key = st.secrets["GROQ_API_KEY"] 

    if uploaded_files:
        if st.sidebar.button("Extract Data", key="btn_25") and api_key:
            with st.spinner(f"Scanning {len(uploaded_files)} drawings..."):
                process_drawings(uploaded_files, api_key)

    st.sidebar.header("Seismic Parameters (2025)")
    zone_factor = info_number_input("Zone Factor (Z)", 0.01, 0.01, "z_val", "2025 Zone Map", "Note: 2025 uses actual PGA (0.15g to 0.75g) depending on return period.")
    importance_factor = info_number_input("Importance Factor (I)", 1.0, 0.1, "i_val", "Importance", "Same as 2016.")
    response_reduction = info_number_input("Elastic Force Reduction (R)", 1.0, 0.5, "r_val", "Reduction Factor", "Note: Redefined as Elastic Force Reduction Factor in 2025.")
    site_class = info_selectbox("Site Class", valid_site_classes, "soil_val", "Site Class", "Replaces Type I, II, III soils.")
    structure_type = info_selectbox("Structure Type for Ta", valid_struct_types, "struct_val", "Ta Formulation", "2025 groups all RC structures into a single advanced formula.")

    # ==========================================
    # MAIN PAGE
    # ==========================================
    st.header("Step 1: Geometry & Plan Density")
    col_geo1, col_geo2, col_geo3 = st.columns(3)
    h = col_geo1.number_input("Building Height (h)", min_value=3.0, key="h_val")
    dx = col_geo2.number_input("Base X-Dimension (dx)", min_value=3.0, key="dx_val")
    dy = col_geo3.number_input("Base Y-Dimension (dy)", min_value=3.0, key="dy_val")
    
    col_m1, col_m2 = st.columns(2)
    num_stories = col_m1.number_input("Number of Stories", min_value=1, max_value=50, key="stories_val")
    p_mw = col_m2.number_input("Masonry Infill Plan Density (ρ_MW %)", min_value=0.0, value=0.0, help="Required for 2025 Ta calculation.")
    eq_direction = st.radio("Earthquake Direction:", ["X-Direction", "Y-Direction"], horizontal=True)

    st.divider()
    st.subheader("Structural Element Editor")
    if st.button("🔍 Open Element Viewer & Editor", type="primary"):
        elements_popup()

    # MATRIX GENERATION
    dl_list, ll_list, k_list, height_from_base_list = [], [], [], []
    cumulative_height = 0.0
    plan_area = dx * dy
    p_cc = 0.0
    p_csw_term = 0.0

    if 'floors' in st.session_state and len(st.session_state.floors) == num_stories:
        floors = st.session_state.floors
        
        # --- 2025 ADVANCED T_a DENSITY CALCULATION ---
        base_floor = floors[0] 
        col_area = sum([float(c.get("b", 0)) * float(c.get("d", 0)) * float(c.get("count", 0)) for c in base_floor.get("columns", [])])
        p_cc = (col_area / plan_area) * 100 if plan_area > 0 else 0.0
        
        for sw in base_floor.get("shear_walls", []):
            L_w = float(sw.get("length", 0))
            t_w = float(sw.get("thickness", 0))
            count = float(sw.get("count", 1))
            p_csw_i = ((L_w * t_w) / plan_area) * 100 if plan_area > 0 else 0.0
            p_csw_term += (p_csw_i * (0.2 + (L_w / h)**2)) * count
        # ---------------------------------------------
            
        for i, floor in enumerate(floors):
            sh = float(floor.get("story_height", h / num_stories))
            ll = float(floor.get("live_load", 300.0))
            cumulative_height += sh
            
            slab_vol = sum([float(s.get("thickness", 0)) * float(s.get("total_area", 0)) for s in floor.get("slabs", [])])
            beam_vol = sum([float(b.get("b", 0)) * float(b.get("d", 0)) * float(b.get("total_length", 0)) for b in floor.get("beams", [])])
            floor_weight = (slab_vol + beam_vol) * density_rc

            col_vol = sum([float(c.get("b", 0)) * float(c.get("d", 0)) * sh * float(c.get("count", 0)) for c in floor.get("columns", [])])
            sw_vol = sum([float(sw.get("length", 0)) * float(sw.get("thickness", 0)) * sh * float(sw.get("count", 0)) for sw in floor.get("shear_walls", [])])
            vert_wt_current = (col_vol + sw_vol) * density_rc
            
            vert_wt_above = 0.0
            if i < num_stories - 1:
                floor_above = floors[i+1]
                sh_above = float(floor_above.get("story_height", 3.0))
                col_vol_above = sum([float(c.get("b", 0)) * float(c.get("d", 0)) * sh_above * float(c.get("count", 0)) for c in floor_above.get("columns", [])])
                vert_wt_above = (col_vol_above) * density_rc

            lumped_dl = floor_weight + (0.5 * vert_wt_current) + (0.5 * vert_wt_above)
            
            total_k = 0.0
            for c in floor.get("columns", []):
                b_val, d_val = float(c.get("b", 0)), float(c.get("d", 0))
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
    df_edited = st.data_editor(df_input, use_container_width=True)

    if st.button("Run 2025 Seismic Analysis"):
        df_calc = df_edited.iloc[::-1].reset_index(drop=True)
        W_array = df_calc["Dead Load (kN)"].values + df_calc["Live Load (kN)"].values
        mass_array = W_array / 9.81  
        K_array = df_calc["Stiffness (kN/m)"].values
        H_array = df_calc["Height from Base (m)"].values
        total_W = np.sum(W_array)
        
        st.subheader(f"Total Seismic Weight (W): {total_W:.2f} kN")

        # --- ESM 2025 ---
        st.header("Step 2: Equivalent Static Method (2025)")
        
        if structure_type == "RC MRF (with/without walls or infill)":
            denominator = np.sqrt(1 + 0.2 * p_cc) + p_csw_term + 0.02 * p_mw
            Ta = (0.075 * (h ** 0.75)) / denominator
            st.info(f"Using advanced 2025 T_a formula. Extracted ρ_CC = {p_cc:.2f}%, ρ_CSW term = {p_csw_term:.4f}")
        elif structure_type == "Steel MRF": Ta = 0.085 * (h ** 0.75)
        else: Ta = 0.09 * h / np.sqrt(dx if eq_direction == "X-Direction" else dy)

        # 2025 ESM Sa/g
        A_NH_stat = get_sa_g_2025(Ta, site_class, "ESM")
        
        # 2025 Ah Formula (NO DIVISION BY 2)
        A_HD_stat = (zone_factor * importance_factor / response_reduction) * A_NH_stat
        VB_stat = A_HD_stat * total_W
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Time Period (Ta)", f"{Ta:.3f} s")
        col2.metric("A_NH (Sa/g)", f"{A_NH_stat:.3f}")
        col3.metric("A_HD (Ah)", f"{A_HD_stat:.4f}")
        col4.metric("Static Base Shear (VB)", f"{VB_stat:.2f} kN")

        # --- RSM 2025 ---
        st.header("Step 3: Dynamic Analysis (RSM 2025)")
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
        
        modal_mass, modal_participation, modal_base_shear = np.zeros(num_stories), np.zeros(num_stories), np.zeros(num_stories)
        r_vec = np.ones(num_stories) 
        
        for i in range(num_stories):
            phi_i = eigenvectors[:, i]
            Mn = np.dot(phi_i.T, np.dot(M_mat, phi_i))
            Ln = np.dot(phi_i.T, np.dot(M_mat, r_vec))
            Pk = Ln / Mn
            Mk = (Ln ** 2) / Mn
            modal_participation[i] = Pk
            modal_mass[i] = Mk
            
            # 2025 RSM Sa/g
            A_NH_dyn = get_sa_g_2025(time_periods[i], site_class, "RSM")
            # 2025 Ah Formula (NO DIVISION BY 2)
            A_HD_dyn = (zone_factor * importance_factor / response_reduction) * A_NH_dyn
            modal_base_shear[i] = A_HD_dyn * Mk * 9.81
        
        df_modes = pd.DataFrame({
            "Mode": [f"Mode {i+1}" for i in range(num_stories)],
            "Time Period (s)": time_periods,
            "Modal Mass (ton)": modal_mass,
            "Mass Part. (%)": (modal_mass / np.sum(mass_array)) * 100,
            "Modal Shear (kN)": np.abs(modal_base_shear)
        })
        st.table(df_modes.style.format({"Time Period (s)": "{:.3f}", "Mass Part. (%)": "{:.2f}%", "Modal Shear (kN)": "{:.2f}"}))
        
        VB_dyn = np.sqrt(np.sum(modal_base_shear**2))
        st.markdown(f"**Dynamic Base Shear (SRSS combination):** `{VB_dyn:.2f} kN`")

        st.header("Step 4: Base Shear Scaling")
        scale_factor = VB_stat / VB_dyn if VB_dyn < VB_stat else 1.0
        
        if scale_factor > 1.0:
            st.warning(f"VB_dyn < VB_stat. Scaling is required!")
            st.success(f"Design Base Shear (Scaled): {VB_dyn * scale_factor:.2f} kN")
        else:
            st.success(f"VB_dyn >= VB_stat. No scaling required.")
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
