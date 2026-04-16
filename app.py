import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Система оценки риска диабета",
    page_icon="🧪",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================
# CSS
# =========================
st.markdown("""
<style>
@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes glowPulse {
    0% { box-shadow: 0 8px 30px rgba(124, 58, 237, 0.12); }
    50% { box-shadow: 0 12px 36px rgba(124, 58, 237, 0.20); }
    100% { box-shadow: 0 8px 30px rgba(124, 58, 237, 0.12); }
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(168, 85, 247, 0.20), transparent 30%),
        radial-gradient(circle at top right, rgba(139, 92, 246, 0.22), transparent 28%),
        linear-gradient(135deg, #f5f3ff 0%, #f3e8ff 45%, #ede9fe 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.hero {
    border-radius: 28px;
    padding: 34px 28px;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 35%, #a855f7 100%);
    color: white;
    animation: fadeUp 0.7s ease;
    box-shadow: 0 18px 45px rgba(109, 40, 217, 0.28);
}

.hero-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 8px;
    line-height: 1.15;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.92;
}

.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.55);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(124, 58, 237, 0.12);
    animation: fadeUp 0.7s ease;
    margin-bottom: 18px;
}

.card-glow {
    animation: fadeUp 0.7s ease, glowPulse 3s ease-in-out infinite;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #4c1d95;
    margin-bottom: 14px;
}

.helper {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 8px;
}

.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #4c1d95 !important;
    font-weight: 600 !important;
}

.stButton > button {
    width: 100%;
    height: 50px;
    border: none;
    border-radius: 14px;
    color: white;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    box-shadow: 0 10px 24px rgba(124, 58, 237, 0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 14px 28px rgba(124, 58, 237, 0.35);
    background: linear-gradient(135deg, #6d28d9 0%, #9333ea 100%);
    color: white;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(168,85,247,0.12));
    border: 1px solid rgba(124,58,237,0.12);
    padding: 16px;
    border-radius: 18px;
}

[data-testid="stMetricLabel"] {
    color: #6d28d9;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #4c1d95;
    font-weight: 800;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed 0%, #c084fc 100%);
    border-radius: 999px;
}

.explain-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.84));
    border: 1px solid rgba(196,181,253,0.7);
    border-radius: 18px;
    padding: 18px 18px 10px 18px;
    margin-top: 18px;
}

.explain-title {
    color: #4c1d95;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
    color: white;
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
}

.footer-note {
    color: #6b7280;
    font-size: 13px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-title">🧪 Система оценки риска диабета</div>
    <div class="hero-subtitle">
        Система оценки риска диабета на основе машинного обучения
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def build_input_dataframe(model, gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, glucose):
    feature_names = list(model.feature_names_in_)
    input_dict = {col: 0 for col in feature_names}

    gender_num = 0 if gender == "Женский" else 1
    hypertension_num = 0 if hypertension == "Нет" else 1
    heart_num = 0 if heart_disease == "Нет" else 1

    # базовые колонки
    base_values = {
        "гендер": gender_num,
        "gender": gender_num,
        "возраст": age,
        "age": age,
        "гипертония": hypertension_num,
        "hypertension": hypertension_num,
        "heart_disease": heart_num,
        "ИМТ": bmi,
        "BMI": bmi,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }

    for col, value in base_values.items():
        if col in input_dict:
            input_dict[col] = value

    # smoking history: UI -> model values
    smoking_map = {
        "Никогда": "never",
        "В прошлом": "former",
        "Настоящее время": "current",
        "Нет информации": "no_info"
    }

    smoking_value = smoking_map[smoking_history]

    # если модель вдруг ждет одно поле smoking_history
    if "smoking_history" in input_dict:
        smoking_numeric = {
            "never": 4,
            "former": 3,
            "current": 1,
            "no_info": 0
        }
        input_dict["smoking_history"] = smoking_numeric[smoking_value]

    # one-hot колонки для smoking_history
    smoking_column_aliases = {
        "current": [
            "smoking_history_current"
        ],
        "former": [
            "smoking_history_former"
        ],
        "never": [
            "smoking_history_never"
        ],
        "no_info": [
            "smoking_history_no info",
            "smoking_history_no_info"
        ]
    }

    # сначала явно ставим 0 для всех smoking колонок
    for col in feature_names:
        col_lower = col.lower()
        if col_lower.startswith("smoking_history_"):
            input_dict[col] = 0

    # затем включаем нужную
    if smoking_value in smoking_column_aliases:
        for target_col in smoking_column_aliases[smoking_value]:
            for real_col in feature_names:
                if real_col.lower() == target_col.lower():
                    input_dict[real_col] = 1

    # если выбрано "В прошлом", а модель вместо former ждет "not текущий"
    if smoking_value == "former":
        for real_col in feature_names:
            if real_col.lower() == "smoking_history_not текущий":
                input_dict[real_col] = 1

    return pd.DataFrame([input_dict], columns=feature_names)


def generate_chat_response(user_message, risk, risk_text, risk_factors, protective_factors):
    msg = user_message.lower()

    if "риск" in msg:
        return f"Текущая оценка модели: **{risk * 100:.2f}%**. Это **{risk_text} риск диабета**."

    elif "почему" in msg or "объясни" in msg:
        response = f"Модель оценивает риск как **{risk_text}**.\n\n"
        if risk_factors:
            response += "**Факторы, повышающие риск:**\n"
            for factor in risk_factors:
                response += f"- {factor}\n"
        if protective_factors:
            response += "\n**Факторы, снижающие риск:**\n"
            for factor in protective_factors:
                response += f"- {factor}\n"
        return response

    elif "что делать" in msg or "рекоменда" in msg:
        if risk < 0.30:
            return "Риск низкий. Полезно поддерживать здоровый вес, следить за питанием и периодически проверять уровень глюкозы."
        elif risk < 0.70:
            return "Риск средний. Стоит обратить внимание на питание, физическую активность, вес и при необходимости обсудить показатели с врачом."
        else:
            return "Риск высокий. Лучше обратиться к врачу и дополнительно проверить глюкозу, HbA1c и другие показатели."

    elif "диагноз" in msg:
        return "Нет, это не медицинский диагноз. Это только оценка риска на основе модели."

    elif "привет" in msg:
        return "Привет. Я могу объяснить результат, назвать факторы риска и подсказать, что означает текущая оценка."

    else:
        return (
            "Я могу помочь так:\n"
            "- объяснить, почему получился такой риск\n"
            "- назвать факторы риска\n"
            "- подсказать, что означает результат\n"
            "- ответить, является ли это диагнозом"
        )

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Ввод данных пациента</div>', unsafe_allow_html=True)
    st.markdown('<div class="helper">Заполни данные и нажми кнопку расчёта</div>', unsafe_allow_html=True)

    gender = st.selectbox("Пол", ["Женский", "Мужской"])
    age = st.slider("Возраст", 1, 100, 25)
    hypertension = st.selectbox("Гипертония", ["Нет", "Да"])
    heart_disease = st.selectbox("Болезни сердца", ["Нет", "Да"])
    smoking_history = st.selectbox(
        "История курения",
        ["Никогда", "В прошлом", "Настоящее время", "Нет информации"]
    )
    bmi = st.number_input("ИМТ", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
    hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.2, step=0.1)
    glucose = st.number_input("Глюкоза", min_value=50, max_value=300, value=95, step=1)

    predict = st.button("Рассчитать риск")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card card-glow">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Результат</div>', unsafe_allow_html=True)

    if predict:
        try:
            input_data = build_input_dataframe(
                model=model,
                gender=gender,
                age=age,
                hypertension=hypertension,
                heart_disease=heart_disease,
                smoking_history=smoking_history,
                bmi=bmi,
                hba1c=hba1c,
                glucose=glucose
            )

            risk = model.predict_proba(input_data)[0][1]

            st.metric("Риск диабета", f"{risk * 100:.2f}%")
            st.progress(float(risk))

            if risk < 0.30:
                st.success("✅ Низкий риск диабета")
                risk_text = "низкий"
                badge_text = "Низкий риск"
            elif risk < 0.70:
                st.warning("⚠️ Средний риск диабета")
                risk_text = "средний"
                badge_text = "Средний риск"
            else:
                st.error("🚨 Высокий риск диабета")
                risk_text = "высокий"
                badge_text = "Высокий риск"

            risk_factors = []
            protective_factors = []

            if age >= 45:
                risk_factors.append("возраст 45+")
            else:
                protective_factors.append("молодой возраст")

            if hypertension == "Да":
                risk_factors.append("наличие гипертонии")
            else:
                protective_factors.append("отсутствие гипертонии")

            if heart_disease == "Да":
                risk_factors.append("наличие сердечных заболеваний")
            else:
                protective_factors.append("отсутствие сердечных заболеваний")

            if bmi >= 30:
                risk_factors.append("высокий ИМТ")
            elif bmi < 25:
                protective_factors.append("нормальный ИМТ")

            if hba1c >= 6.5:
                risk_factors.append("повышенный HbA1c")
            else:
                protective_factors.append("нормальный HbA1c")

            if glucose >= 140:
                risk_factors.append("повышенный уровень глюкозы")
            else:
                protective_factors.append("нормальный уровень глюкозы")

            if smoking_history in ["В прошлом", "Настоящее время"]:
                risk_factors.append("история курения")
            else:
                protective_factors.append("отсутствие истории курения")

            st.session_state.last_result = {
                "risk": risk,
                "risk_text": risk_text,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors
            }

            st.markdown(
                f"""
                <div class="explain-box">
                    <div class="badge">{badge_text}</div>
                    <div class="explain-title">🧠 Объяснение ИИ</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"Модель оценивает риск диабета как **{risk_text}**.")

            if risk_factors:
                st.write("**Факторы, которые повышают риск:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")

            if protective_factors:
                st.write("**Факторы, которые снижают риск:**")
                for factor in protective_factors:
                    st.write(f"• {factor}")

            st.info("Это не медицинский диагноз.")

        except Exception as e:
            st.error("Ошибка при предсказании. Проверь соответствие признаков модели.")
            st.write("Ожидаемые моделью колонки:")
            st.write(list(model.feature_names_in_))
            st.write("Текущие колонки, отправленные в модель:")
            st.write(list(input_data.columns) if "input_data" in locals() else "Не удалось сформировать input_data")
            st.write("Текст ошибки:")
            st.code(str(e))

    else:
        st.markdown("""
        <div class="explain-box">
            <div class="badge">Ожидание</div>
            <div class="explain-title">Здесь появится результат</div>
            <div class="footer-note">
                Заполни данные слева и нажми кнопку расчёта риска
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💬 AI Chat")

    if st.session_state.last_result is None:
        st.info("Сначала рассчитай риск, потом чат сможет объяснить результат.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_prompt = st.chat_input("Спроси про результат...")

        if user_prompt:
            st.session_state.messages.append({
                "role": "user",
                "content": user_prompt
            })

            with st.chat_message("user"):
                st.markdown(user_prompt)

            result = st.session_state.last_result
            assistant_reply = generate_chat_response(
                user_message=user_prompt,
                risk=result["risk"],
                risk_text=result["risk_text"],
                risk_factors=result["risk_factors"],
                protective_factors=result["protective_factors"]
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

            with st.chat_message("assistant"):
                st.markdown(assistant_reply)

    st.markdown('</div>', unsafe_allow_html=True)