import streamlit as st
import json
import base64
from groq import Groq

# --- 1. POPUP DIALOG FOR VERIFICATION ---
# --- 1. POPUP DIALOG FOR VERIFICATION ---
@st.dialog("Review Extracted Structural Elements", width="large")
def elements_popup():
    st.markdown("Verify the dimensions extracted by the AI. Saving will automatically recalculate Floor Weights and Stiffness for the Base Shear analysis.")
    
    if 'floors' not in st.session_state or not st.session_state.floors:
        st.warning("No floor data available. Extract data first.")
        return

    tabs = st.tabs([f"Floor {i+1}" for i in range(len(st.session_state.floors))])
    
    for i, tab in enumerate(tabs):
        with tab:
            floor = st.session_state.floors[i]
            
            # Top row for floor parameters
            col_top1, col_top2, col_top3 = st.columns(3)
            floor["story_height"] = col_top1.number_input(f"Story Height (m)", value=float(floor.get("story_height", 3.0)), key=f"h_{i}")
            floor["live_load_value"] = col_top2.number_input(f"Live Load Value", value=float(floor.get("live_load_value", 300.0)), key=f"ll_val_{i}")
            floor["live_load_unit"] = col_top3.selectbox(f"Live Load Unit", options=["kN", "kN/m2"], index=0 if floor.get("live_load_unit") == "kN" else 1, key=f"ll_unit_{i}")
            
            # --- ADD THIS MISSING LINE ---
            col1, col2 = st.columns(2)
            # -----------------------------
            
            with col1:
                st.subheader("Columns")
                floor["columns"] = st.data_editor(floor.get("columns", []), num_rows="dynamic", key=f"col_{i}")
                st.subheader("Shear Walls")
                floor["shear_walls"] = st.data_editor(floor.get("shear_walls", []), num_rows="dynamic", key=f"sw_{i}")
            with col2:
                st.subheader("Beams")
                floor["beams"] = st.data_editor(floor.get("beams", []), num_rows="dynamic", key=f"beam_{i}")
                st.subheader("Slabs")
                floor["slabs"] = st.data_editor(floor.get("slabs", []), num_rows="dynamic", key=f"slab_{i}")
                
            st.session_state.floors[i] = floor
    
    if st.button("Save & Recalculate Base Shear Matrix"):
        st.rerun()

# --- 2. THE AI PROCESSING ENGINE ---
def process_drawings(uploaded_files, api_key):
    """Handles the Groq API call and dynamically updates session state."""
    
    # Define valid types for fallback validation
    valid_soil_types = ["Hard (Type I)", "Medium (Type II)", "Soft (Type III)"]
    valid_struct_types = ["RC Bare Frame", "RC Frame with Masonry Infill", "Steel Frame"]
    
    prompt = """
    You are a quantity surveyor and structural engineer. Analyze the provided architectural/structural drawings and notes.
    Extract the overall building parameters AND the detailed structural elements per floor.
    
    CRITICAL RULES:
    1. `floors` array must be ordered from Bottom (Story 1) to Top (Roof).
    2. Convert all dimensions to METERS (e.g., 300mm = 0.3m).
    3. Identify the unit of the live load from the drawings (e.g., 'kN/m2' or 'kN').
    4. If an element type does not exist on a floor, leave its array empty [].
    5. Respond ONLY with raw JSON. No markdown blocks.

    {
      "height": null, 
      "base_x": null, 
      "base_y": null, 
      "num_stories": null, 
      "z": null, "i": null, "r": null, 
      "soil_type": null, "structure_type": null, 
      "floors": [
        {
          "floor_name": "string",
          "story_height": 0.0,
          "live_load_value": 0.0,
          "live_load_unit": "string",
          "columns": [{"type": "string", "b": 0.0, "d": 0.0, "count": 0}],
          "beams": [{"type": "string", "b": 0.0, "d": 0.0, "total_length": 0.0}],
          "slabs": [{"type": "string", "thickness": 0.0, "total_area": 0.0}],
          "shear_walls": [{"type": "string", "length": 0.0, "thickness": 0.0, "count": 0}]
        }
      ]
    }
    """
    
    content_payload = [{"type": "text", "text": prompt}]
    for file in uploaded_files:
        base64_image = base64.b64encode(file.getvalue()).decode('utf-8')
        content_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": content_payload}],
            temperature=0.1
        )
        
        clean_text = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_text)

        # Update Session State with extracted global parameters
        if data.get("height"): st.session_state.h_val = float(data["height"])
        if data.get("base_x"): st.session_state.dx_val = float(data["base_x"])
        if data.get("base_y"): st.session_state.dy_val = float(data["base_y"])
        if data.get("num_stories"): st.session_state.stories_val = int(data["num_stories"])
        if data.get("z") is not None: st.session_state.z_val = float(data["z"])
        if data.get("i"): st.session_state.i_val = float(data["i"])
        if data.get("r"): st.session_state.r_val = float(data["r"])
        if data.get("soil_type") in valid_soil_types: st.session_state.soil_val = data["soil_type"]
        if data.get("structure_type") in valid_struct_types: st.session_state.struct_val = data["structure_type"]
        
        # Update Session State with floor arrays
        if data.get("floors") and isinstance(data["floors"], list):
            st.session_state.floors = data["floors"]
            if not data.get("height"):
                st.session_state.h_val = sum([float(f.get("story_height", 0) or 0) for f in data["floors"]])
        
        st.session_state.success_msg = f"✅ Extracted params & detailed elements for {len(data.get('floors', []))} floors!"
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"Error extracting data: {e}")
