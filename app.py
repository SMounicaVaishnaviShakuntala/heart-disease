import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
import plotly.express as px

st.set_page_config(layout="wide")

# ===================== GLOBAL STYLES =====================
st.markdown("""
<style>

/* Background */
/* ===== FINAL MEDICAL DASHBOARD BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #fff5f5 0%,
        #ffe5e5 35%,
        #ffd6d6 70%,
        #fff0f0 100%
    );
}

/* Container padding */

.block-container {
    padding: 4rem 3rem 2rem 3rem !important;
}


/* ===================== TABS STYLING ===================== */

/* Remove underline & move right */
div[data-testid="stTabs"] div[role="tablist"] {
    border-bottom: none !important;
    justify-content: flex-end !important;
}

div[data-testid="stTabs"] {
    border-bottom: none !important;
}

/* Tab button */
button[data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.7) !important;
    padding: 10px 28px !important;
    border-radius: 30px !important;
    margin: 0 8px;
    font-weight: 600;
    border: none !important;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ff3b3b !important;
    color: white !important;
}

/* Hover = Hero Blue */
button[data-baseweb="tab"]:hover {
    background-color: #ff3b3b !important;
    color: white !important;
}

}
 /* ===== RESET FORM TO DEFAULT STREAMLIT LOOK ===== */
div[data-testid="stForm"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
}



/* ===== SLIM & VERTICAL FORM ===== */

/* Center form and reduce width */
div[data-testid="stForm"] {
    max-width: 600px !important;
    margin: auto !important;
}

/* Make inputs slim */
div[data-testid="stNumberInput"] {
    margin-bottom: 18px !important;
}

/* Reduce input box width */
div[data-testid="stNumberInput"] input {
    height: 38px !important;
    font-size: 15px !important;
}

/* Make button centered */
div[data-testid="stFormSubmitButton"] {
    text-align: center !important;
}

/* ===== HERO STYLE FORM ===== */
div[data-testid="stForm"] {
    background: linear-gradient(90deg, #ff3b3b, #ff6b6b) !important;
    max-width: 720px !important;   /* Increased ~20% from 600px */
    margin: auto !important;
    padding: 45px 40px !important;
    border-radius: 30px !important;
    box-shadow: 0px 18px 40px rgba(0,0,0,0.15) !important;
}

/* Labels white */
label {
    color: white !important;
    font-weight: 500 !important;
}

/* Input boxes clean & contrasting */
div[data-testid="stNumberInput"] input {
    background-color: white !important;
    border-radius: 12px !important;
    height: 40px !important;
    border: none !important;
    font-size: 15px !important;
}

/* Space between inputs */
div[data-testid="stNumberInput"] {
    margin-bottom: 20px !important;
}

/* Center button */
div[data-testid="stFormSubmitButton"] {
    text-align: center !important;
}

.result-box {
    margin-top: 25px;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    color: white;
    font-size: 20px;
    font-weight: bold;
    width: 40%;
    margin-left: auto;
    margin-right: auto;
}

iframe {
    border-radius: 30px !important;
    overflow: hidden !important;
}

/* ===== MODERN CARD SECTION (Like Microsoft Style) ===== */

.section2-wrapper {
    margin-top: 60px;
}

.section2-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 10px;
}

.section2-subtitle {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 50px;
}

.modern-card {
    background: #f3f4f6;
    border-radius: 28px;
    padding: 20px;
    transition: all 0.4s ease;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
}

.modern-card:hover {
    transform: translateY(-8px);
    box-shadow: 0px 18px 40px rgba(0,0,0,0.18);
}

.modern-card img {
    width: 100%;
    height: 240px;
    object-fit: cover;
    border-radius: 20px;
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 18px;
    color: #111827;
}

.card-desc {
    font-size: 16px;
    color: #4b5563;
    margin-top: 10px;
    line-height: 1.6;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 40px;
    margin-bottom: 25px;
    color: #1f2937;
}           
        

</style>
""", unsafe_allow_html=True)

# ===================== MODEL =====================
model = joblib.load("heart_best_model.pkl")

def risk_level(p):
    if p < 0.4:
        return "Low Risk", "#2ecc71"
    elif p < 0.7:
        return "Medium Risk", "#f39c12"
    else:
        return "High Risk", "#e74c3c"

# ===================== TOP TABS =====================

tab1, tab2, tab3 = st.tabs(["Home", "Predict", "Insights"])


# ===================== HOME =====================
with tab1:

    
# ===================== HERO SECTION =====================
    components.html("""
    <div style="
        background: linear-gradient(90deg, #ff3b3b, #ff6b6b);
        padding: 35px 40px 25px 40px;
        border-radius: 30px;
        overflow: hidden;
        text-align: center;
        color: white;
        box-shadow: 0px 12px 28px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
        height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    ">

        <h1 style="
            margin-bottom:18px;
            font-size:48px;
            font-weight:700;
        ">
            CardioPredict AI
        </h1>

        <div id="typewriter" style="
            font-size:18px; 
            max-width:800px; 
            margin-left:auto; 
            margin-right:auto;
            line-height:1.7;
            height:100px;
            overflow:hidden;
        "></div>

    </div>

   <script>
const text = `CardioPredict AI uses Machine Learning to analyze essential cardiovascular health parameters such as age, blood pressure, cholesterol levels, heart rate, and chest pain indicators. It estimates the probability of heart disease risk and provides intelligent insights to support early detection, timely intervention, and preventive care.`;

let i = 0;
function typeWriter() {
    if (i < text.length) {
        document.getElementById("typewriter").innerHTML += text.charAt(i);
        i++;
        setTimeout(typeWriter, 18);
    }
}
typeWriter();
</script>
    """, height=280)

    st.markdown('<div class="section-title"> Potential Complications of Undiagnosed Heart Disease</div>', unsafe_allow_html=True)

    # Top Row (3 images)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("heart_attack.png", width=520)

        st.markdown(
            "<div style='font-size:20px; font-weight:700; margin-top:10px;'>Heart Attack</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='font-size:18px; color:#444;'>Blocked arteries stop blood flow to the heart muscle, causing permanent damage.</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.image("heart_failure.png", width=520)

        st.markdown(
            "<div style='font-size:20px; font-weight:700; margin-top:10px;'>Heart Failure</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='font-size:18px; color:#444;'>The heart becomes too weak to pump blood effectively throughout the body.</div>",
            unsafe_allow_html=True
        )

    # 3️⃣ Heart Disease
    with col3:
        st.image("heart_stroke.png", width=520)

        st.markdown(
            "<div style='font-size:20px; font-weight:700; margin-top:10px;'>Brain Stroke</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='font-size:18px; color:#444;'>A blocked or burst blood vessel cuts off oxygen supply to the brain.</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col4, col5, col6, col7 = st.columns([1,2,2,1])


    with col5:
        st.image("dangerous_arrhythmias.png", width=520)

        st.markdown(
            "<div style='font-size:20px; font-weight:700; margin-top:10px;'>Dangerous Arrhythmias</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='font-size:18px; color:#444;'>Abnormal heart rhythms disrupt the heart’s natural electrical activity.</div>",
            unsafe_allow_html=True
        )

    with col6:
        st.image("cardiac_arrest.png", width=520)

        st.markdown(
            "<div style='font-size:20px; font-weight:700; margin-top:10px;'>Cardiac Arrest</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div style='font-size:18px; color:#444;'>The heart suddenly stops beating, leading to immediate loss of blood circulation.</div>",
            unsafe_allow_html=True
        )

    # SECTION 3

    st.markdown('<div class="section-title">Preventive Measures and Risk Reduction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,3])

    with col1:
        st.image("healthy_diet.png", width=520)

    with col2:
        st.markdown(
            "<div style='font-size:26px; font-weight:700;'>Healthy Diet</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:20px; line-height:1.7; color:#444;'>"
            "Maintaining a heart-healthy diet plays a vital role in preventing cardiovascular diseases. "
"A balanced intake of whole grains, fresh fruits, vegetables, lean proteins, and healthy fats "
"helps control cholesterol levels and supports proper blood circulation. Reducing saturated fats, "
"excess salt, and processed foods lowers the risk of plaque buildup in arteries. Consistent healthy "
"dietary habits significantly decrease the chances of heart attack, stroke, and long-term cardiac complications.""</div>",
            unsafe_allow_html=True
        )


    st.markdown("<br><br>", unsafe_allow_html=True)

    col3, col4 = st.columns([3,2])

    with col3:
        st.markdown(
            "<div style='font-size:26px; font-weight:700;'>Regular Physical Activity</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:20px; line-height:1.7; color:#444;'>"
            "Engaging in regular physical activity strengthens the heart muscle and improves blood circulation throughout the body. "
"Activities such as brisk walking, jogging, cycling, or light strength training help control blood pressure, cholesterol levels, and body weight. "
"Consistent exercise reduces the risk of artery blockage and supports overall cardiovascular health. "
"Maintaining an active lifestyle significantly lowers the chances of heart attack, stroke, and long-term cardiac complications.""</div>",
            unsafe_allow_html=True
        )

    with col4:
        st.image("jogging.png", width=520)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col5, col6 = st.columns([2,3])

    with col5:
        st.image("doc_consultation.png", width=520)

    with col6:
        st.markdown(
            "<div style='font-size:26px; font-weight:700;'>Regular Monitoring & Consultation</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:20px; line-height:1.7; color:#444;'>"
           "Routine health checkups and timely medical consultation enable early detection and effective management of heart disease. "
"Regular monitoring of blood pressure, cholesterol levels, and heart rhythm helps identify potential risks before serious complications develop. "
"Professional medical guidance ensures appropriate lifestyle modifications and treatment plans when necessary. "
"Early intervention significantly reduces the risk of heart attack, stroke, and long-term cardiovascular damage."
            "</div>",unsafe_allow_html=True
        )

# ===================== PREDICT =====================
with tab2:

    st.markdown("### Enter Patient Heart Health Parameters")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp_option = st.selectbox(
                        "Chest Pain Type",
                        [
                            "0 - Typical Angina",
                            "1 - Atypical Angina",
                            "2 - Non-anginal Pain",
                            "3 - Asymptomatic"
                        ]
                    )

            cp = int(cp_option[0])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)

        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", value=200)
            fbs_option = st.selectbox(
                        "Fasting Blood Sugar > 120 mg/dl",
                        ["No (<=120 mg/dl)", "Yes (>120 mg/dl)"]
                        )
            thalachh = st.number_input("Maximum Heart Rate Achieved", value=150)
      
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
             submit = st.form_submit_button("Predict Heart Disease Risk")
    if submit:

        # Convert sex to numeric
        sex_value = 1 if sex == "Male" else 0
        fbs = 1 if fbs_option == "Yes (>120 mg/dl)" else 0
        exang = 0
        # Create dataframe with default values for hidden features
        input_data = pd.DataFrame(
            [[
                age,
                sex_value,
                cp,
                trestbps,
                chol,
                fbs,
                1,          # restecg default
                thalachh,
                exang,      # now default
                1.0,        # oldpeak default
                1,          # slope default
                0,          # ca default
                2           # thal default
            ]],
            columns=[
                "age","sex","cp","trestbps","chol","fbs",
                "restecg","thalachh","exang","oldpeak",
                "slope","ca","thal"
            ]
        )
        st.session_state["last_input"] = input_data

        prob = model.predict_proba(input_data)[0][1]
        risk, color = risk_level(prob)

        st.markdown(
            f"""
            <div class='result-box' style='background-color:{color};'>
                Risk Level: {risk}<br>
                Probability: {prob*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

# ===================== INSIGHTS =====================
with tab3:

    st.markdown("## Your AI Risk Insights")

    if "last_input" not in st.session_state:
        st.info("Please predict risk first.")
    else:
        user_data = st.session_state["last_input"]

        # Get feature importance from Random Forest
        importances = model.feature_importances_

        # Create dataframe
        insight_df = pd.DataFrame({
            "Feature": user_data.columns,
            "Contribution": importances
        })

        # Show ONLY features visible in form
        allowed_features = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "thalachh"
        ]

        insight_df = insight_df[
            insight_df["Feature"].isin(allowed_features)
        ]

        # ================= KEEP FORM ORDER =================
        insight_df = insight_df.set_index("Feature")
        insight_df = insight_df.loc[allowed_features].reset_index()

        # ================= CALCULATE TOP 3 =================
        top3 = insight_df.sort_values(
            by="Contribution",
            ascending=False
        ).head(3)

        # ================= GRAPH =================
        import plotly.express as px

        fig = px.bar(
            insight_df,
            x="Feature",
            y="Contribution",
            color_discrete_sequence=["#ff3b3b"]
        )

        fig.update_layout(
            plot_bgcolor="#fff5f5",
            paper_bgcolor="#fff5f5",
            font=dict(color="#2c3e50", size=14),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f3cccc"),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        fig.update_traces(marker_line_width=0)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom2d","pan2d","select2d","lasso2d",
                    "zoomIn2d","zoomOut2d","autoScale2d",
                    "resetScale2d","toImage"
                ]
            }
        )

        st.markdown(
            "<big><b>Note:</b> Higher bars indicate features that have greater impact on heart disease risk according to the trained model.</big>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ================= BOXES BELOW GRAPH =================
        col1, col2 = st.columns(2)

        # ----------- TOP FACTORS BOX -----------
        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #ff3b3b, #ff6b6b);
                padding: 30px;
                border-radius: 25px;
                color: white;
                box-shadow: 0px 12px 28px rgba(0,0,0,0.2);
                min-height: 220px;
            ">
                <h3>Top Factors Influencing Heart Risk</h3>
                <ul style="font-size:22px; line-height:1.8;">
                    <li>{top3.iloc[0]['Feature']}</li>
                    <li>{top3.iloc[1]['Feature']}</li>
                    <li>{top3.iloc[2]['Feature']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # ----------- HEALTH INSIGHT BOX -----------
        with col2:

            main_factor = top3.iloc[0]['Feature']

            advice = {
                "age": "Heart disease risk increases with age. Regular cardiac screening is strongly recommended.",
                "sex": "Men are generally at higher early risk, but women’s risk rises after menopause.",
                "cp": "Chest pain may indicate reduced blood flow to the heart. Seek medical evaluation if persistent.",
                "trestbps": "High blood pressure damages arteries over time. Maintain a low-sodium diet and regular exercise.",
                "chol": "Elevated cholesterol contributes to artery blockage. Monitor levels and reduce saturated fat intake.",
                "fbs": "High blood sugar increases cardiovascular risk. Maintain healthy diet and routine monitoring.",
                "thalachh": "Abnormal heart rate response during activity may indicate underlying cardiac stress."
            }

            suggestion = advice.get(
                main_factor,
                "Maintain a heart-healthy lifestyle and consult a healthcare professional regularly."
            )

            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #ff3b3b, #ff6b6b);
                padding: 30px;
                border-radius: 25px;
                color: white;
                box-shadow: 0px 12px 28px rgba(0,0,0,0.2);
                min-height: 220px;
            ">
                <h3>Personalized Heart Health Insight</h3>
                <p style="font-size:22px; line-height:1.8;">
                    {suggestion}
                </p>
            </div>
            """, unsafe_allow_html=True)