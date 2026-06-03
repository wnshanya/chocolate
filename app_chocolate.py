import pandas as pd
import streamlit as st

from model_chocolate import predict, load_model, open_data, DATA_PATH


def show_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Оценка шоколада",
        page_icon="🍫",
    )

    st.write(
        """
        # 🍫 Предсказание рейтинга шоколада
        Модель предсказывает **рейтинг** шоколадного батончика (от 1 до 5)
        на основе характеристик производителя, происхождения какао и состава.

        Задайте параметры на боковой панели слева и получите прогноз.
        """
    )


def sidebar_input_features(df: pd.DataFrame):
    st.sidebar.header("Параметры шоколада")

    manufacturers = sorted(df["company_manufacturer"].dropna().unique().tolist())
    manufacturer = st.sidebar.selectbox("Производитель", manufacturers)

    locations = sorted(df["company_location"].dropna().unique().tolist())
    location = st.sidebar.selectbox("Страна производителя", locations)

    review_date = st.sidebar.slider(
        "Год обзора", min_value=2006, max_value=2022, value=2019, step=1
    )

    origins = sorted(df["country_of_bean_origin"].dropna().unique().tolist())
    bean_origin = st.sidebar.selectbox("Страна происхождения какао", origins)

    bean_names = sorted(
        df["specific_bean_origin_or_bar_name"].dropna().unique().tolist()
    )
    bean_name = st.sidebar.selectbox("Название бара / происхождение зерна", bean_names)

    cocoa_percent = st.sidebar.slider(
        "Содержание какао, %", min_value=42, max_value=100, value=70, step=1
    )

    ingredients_options = sorted(df["ingredients"].dropna().unique().tolist())
    ingredients = st.sidebar.selectbox("Ингредиенты", ingredients_options)

    characteristics_options = sorted(
        df["most_memorable_characteristics"].dropna().unique().tolist()
    )
    characteristics = st.sidebar.selectbox(
        "Запоминающиеся характеристики", characteristics_options
    )

    ref_val = int(df["ref"].median())

    data = {
        "ref": ref_val,
        "company_manufacturer": manufacturer,
        "company_location": location,
        "review_date": review_date,
        "country_of_bean_origin": bean_origin,
        "specific_bean_origin_or_bar_name": bean_name,
        "cocoa_percent": f"{cocoa_percent}%",
        "ingredients": ingredients,
        "most_memorable_characteristics": characteristics,
    }
    return pd.DataFrame(data, index=[0])


def write_user_data(df: pd.DataFrame):
    st.write("## Заданные параметры")
    display = df.drop(columns=["ref"], errors="ignore").rename(
        columns={
            "company_manufacturer": "Производитель",
            "company_location": "Страна производителя",
            "review_date": "Год обзора",
            "country_of_bean_origin": "Страна какао",
            "specific_bean_origin_or_bar_name": "Название / происхождение",
            "cocoa_percent": "Какао, %",
            "ingredients": "Ингредиенты",
            "most_memorable_characteristics": "Характеристики",
        }
    )
    st.write(display)


def rating_to_label(rating: float) -> str:
    if rating >= 4.0:
        return "Выдающийся"
    elif rating >= 3.5:
        return "Высокорекомендуемый"
    elif rating >= 3.0:
        return "Рекомендуемый"
    elif rating >= 2.0:
        return "Разочаровывающий"
    else:
        return "Неудовлетворительный"


def write_prediction(prediction: float):
    label = rating_to_label(prediction)
    stars = "⭐" * round(prediction)

    st.write("## Прогноз рейтинга")
    st.success(f"{stars}  **{prediction:.2f} / 5.00** — {label}")

    # Визуальная шкала
    st.write("### Шкала рейтинга")
    col1, col2, col3, col4, col5 = st.columns(5)
    thresholds = [
        (col1, 1.0, 2.0, "1–2", "Неудовл."),
        (col2, 2.0, 2.75, "2–2.75", "Разочар."),
        (col3, 2.75, 3.0, "2.75–3", "Рекомен."),
        (col4, 3.0, 3.5, "3–3.5", "Высоко рек."),
        (col5, 3.5, 5.0, "3.5–5", "Выдающ."),
    ]
    for col, low, high, label_range, label_name in thresholds:
        active = low <= prediction <= high
        with col:
            if active:
                st.metric(label_name, label_range, delta="← вы здесь")
            else:
                st.metric(label_name, label_range)


def process_main_page():
    show_main_page()

    try:
        bundle = load_model()
    except FileNotFoundError:
        st.warning(
            "⚠️ Файл модели не найден. Запустите `model_chocolate.py`, "
            "чтобы обучить и сохранить модель."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{bundle['mae']:.4f}")
    col2.metric("RMSE", f"{bundle['rmse']:.4f}")
    col3.metric("R²", f"{bundle['r2']:.4f}")

    try:
        df = open_data(DATA_PATH)
    except FileNotFoundError:
        st.error("Файл данных chocolate.csv не найден.")
        return

    user_df = sidebar_input_features(df)
    write_user_data(user_df)

    prediction = predict(user_df)
    write_prediction(prediction)

    # Дополнительно: распределение рейтингов в датасете
    with st.expander("📊 Распределение рейтингов в датасете"):
        rating_counts = df["rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)
        st.caption(
            f"Всего записей: {len(df)} | "
            f"Средний рейтинг: {df['rating'].mean():.2f} | "
            f"Медиана: {df['rating'].median():.2f}"
        )


if __name__ == "__main__":
    process_main_page()
