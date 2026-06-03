import pandas as pd
from pickle import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "chocolate.csv"
MODEL_PATH = "chocolate_model.mw"

TARGET = "rating"

NUM_COLS = ["ref", "review_date", "cocoa_percent_num"]

CAT_COLS = [
    "company_manufacturer",
    "company_location",
    "country_of_bean_origin",
    "specific_bean_origin_or_bar_name",
    "ingredients",
    "most_memorable_characteristics"
]


def open_data(path=DATA_PATH):
    return pd.read_csv(path)


def preprocess_data(df, train=True):
    df = df.copy()

    if "cocoa_percent" in df.columns:
        df["cocoa_percent_num"] = (
            df["cocoa_percent"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    if train:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
        return X, y

    return df


def fit_and_save_model(path=MODEL_PATH):
    df = open_data()

    X, y = preprocess_data(df, train=True)

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            NUM_COLS
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]),
            CAT_COLS
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    bundle = {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

    with open(path, "wb") as file:
        dump(bundle, file)

    print(f"Модель сохранена в {path}")


def load_model(path=MODEL_PATH):
    with open(path, "rb") as file:
        return load(file)


def predict(user_df, path=MODEL_PATH):
    bundle = load_model(path)
    model = bundle["model"]

    X = preprocess_data(user_df, train=False)

    prediction = model.predict(X)[0]

    return prediction


if __name__ == "__main__":
    fit_and_save_model()
