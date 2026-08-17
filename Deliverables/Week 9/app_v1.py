from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="California Home Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    """Load the trained model bundle from the same folder as app.py."""
    app_dir = Path(__file__).resolve().parent

    # Prefer the clean deployment filename. The second option supports
    # the uploaded filename used while developing this app.
    candidates = [
        app_dir / "model.pkl",
        app_dir / "model(1).pkl",
    ]

    for model_path in candidates:
        if model_path.exists():
            return joblib.load(model_path)

    raise FileNotFoundError(
        "Could not find model.pkl. Place the trained model bundle in the "
        "same directory as app.py and name it 'model.pkl'."
    )


bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]

zip_lut = bundle["zip_lut"]
city_lut = bundle["city_lut"]
district_lut = bundle["district_lut"]
zip_to_district = bundle["zip_to_district"]
city_to_district = bundle["city_to_district"]


def display_county_name(feature_name: str) -> str:
    return feature_name.removeprefix("County_").replace("_", " ")


# Los Angeles is the dropped/reference county, so choosing it leaves all
# County_* features at 0.
county_feature_map = {
    display_county_name(col): col
    for col in features
    if col.startswith("County_")
}
county_options = ["Los Angeles"] + sorted(county_feature_map.keys())

city_options = sorted(city_lut.keys())


def normalize_zip(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits.zfill(5) if digits else ""


def build_feature_row(
    living_area,
    lot_size_acres,
    bedrooms,
    bathrooms,
    year_built,
    stories,
    garage_spaces,
    parking_total,
    latitude,
    longitude,
    county,
    postal_code,
    city,
    view_yn,
    basement_yn,
    pool_private_yn,
    attached_garage_yn,
    fireplace_yn,
    new_construction_yn,
):
    """Transform Streamlit inputs into the exact training feature layout."""
    row = pd.DataFrame(
        np.zeros((1, len(features)), dtype=float),
        columns=features,
    )

    # Numeric / structural features
    numeric_values = {
        "BedroomsTotal": bedrooms,
        "BathroomsTotalInteger": bathrooms,
        "YearBuilt": year_built,
        "Stories": stories,
        "GarageSpaces": garage_spaces,
        "ParkingTotal": parking_total,
        "Latitude": latitude,
        "Longitude": longitude,
        "LivingArea_log": np.log1p(living_area),
        "LotSizeAcres_log": np.log(lot_size_acres),
        # Per the model specification, these indicators remain 0 for
        # newly entered complete values.
        "LotSizeAcres_imputed": 0,
        "YearBuilt_imputed": 0,
    }

    for col, value in numeric_values.items():
        if col in row.columns:
            row.at[0, col] = value

    # Yes/No fields used by the trained model
    binary_values = {
        "ViewYN": int(view_yn),
        "BasementYN": int(basement_yn),
        "PoolPrivateYN": int(pool_private_yn),
        "AttachedGarageYN": int(attached_garage_yn),
        "FireplaceYN": int(fireplace_yn),
        "NewConstructionYN": int(new_construction_yn),
    }
    for col, value in binary_values.items():
        if col in row.columns:
            row.at[0, col] = value

    # County one-hot. Los Angeles is the reference category.
    county_col = county_feature_map.get(county)
    if county_col in row.columns:
        row.at[0, county_col] = 1

    # ZIP one-hot
    zip_code = normalize_zip(postal_code)
    zip_col = zip_lut.get(zip_code, "Zip_Other")
    if zip_col in row.columns:
        row.at[0, zip_col] = 1

    # City one-hot
    city_col = city_lut.get(city, "City_Other")
    if city_col in row.columns:
        row.at[0, city_col] = 1

    # District is resolved from ZIP first, then city if ZIP is unavailable.
    district = zip_to_district.get(zip_code)
    if district is None:
        district = city_to_district.get(city)

    district_col = district_lut.get(district)
    if district_col in row.columns:
        row.at[0, district_col] = 1

    # Enforce the exact model feature order.
    return row[features], {
        "zip_code": zip_code,
        "zip_feature": zip_col,
        "city_feature": city_col,
        "district": district,
        "district_feature": district_col,
    }


st.title("🏠 California Home Price Predictor")
st.caption(
    "Enter property characteristics below to estimate the home's closing price "
    "using the trained regression model."
)

with st.form("prediction_form"):
    st.subheader("Property details")

    col1, col2, col3 = st.columns(3)

    with col1:
        living_area = st.number_input(
            "Living area (sq ft)",
            min_value=1.0,
            value=1800.0,
            step=50.0,
        )
        lot_size_acres = st.number_input(
            "Lot size (acres)",
            min_value=0.0001,
            value=0.15,
            step=0.01,
            format="%.4f",
        )
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            value=3,
            step=1,
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            value=2,
            step=1,
        )
        year_built = st.number_input(
            "Year built",
            min_value=1800,
            max_value=2100,
            value=1990,
            step=1,
        )

    with col2:
        stories = st.number_input(
            "Stories",
            min_value=0,
            value=1,
            step=1,
        )
        garage_spaces = st.number_input(
            "Garage spaces",
            min_value=0,
            value=2,
            step=1,
        )
        parking_total = st.number_input(
            "Total parking spaces",
            min_value=0,
            value=2,
            step=1,
        )
        latitude = st.number_input(
            "Latitude",
            value=34.0522,
            format="%.6f",
        )
        longitude = st.number_input(
            "Longitude",
            value=-118.2437,
            format="%.6f",
        )

    with col3:
        county = st.selectbox(
            "County",
            county_options,
            index=county_options.index("Los Angeles"),
        )
        postal_code = st.text_input(
            "ZIP code",
            value="90001",
            max_chars=10,
        )
        city = st.selectbox(
            "City",
            city_options,
            index=city_options.index("Los Angeles")
            if "Los Angeles" in city_options
            else 0,
        )

    st.subheader("Property features")
    b1, b2, b3 = st.columns(3)

    with b1:
        view_yn = st.checkbox("View")
        basement_yn = st.checkbox("Basement")

    with b2:
        pool_private_yn = st.checkbox("Private pool")
        attached_garage_yn = st.checkbox("Attached garage")

    with b3:
        fireplace_yn = st.checkbox("Fireplace")
        new_construction_yn = st.checkbox("New construction")

    submitted = st.form_submit_button(
        "Predict closing price",
        type="primary",
        use_container_width=True,
    )


if submitted:
    try:
        if lot_size_acres <= 0:
            st.error("Lot size must be greater than 0 acres.")
        elif living_area <= 0:
            st.error("Living area must be greater than 0 square feet.")
        else:
            X, location_debug = build_feature_row(
                living_area=living_area,
                lot_size_acres=lot_size_acres,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                year_built=year_built,
                stories=stories,
                garage_spaces=garage_spaces,
                parking_total=parking_total,
                latitude=latitude,
                longitude=longitude,
                county=county,
                postal_code=postal_code,
                city=city,
                view_yn=view_yn,
                basement_yn=basement_yn,
                pool_private_yn=pool_private_yn,
                attached_garage_yn=attached_garage_yn,
                fireplace_yn=fireplace_yn,
                new_construction_yn=new_construction_yn,
            )

            log_prediction = float(model.predict(X)[0])
            predicted_price = float(np.exp(log_prediction))

            st.success("Prediction complete")
            st.metric(
                "Estimated closing price",
                f"${predicted_price:,.0f}",
            )

            with st.expander("Prediction details"):
                st.write(f"Model class: `{type(model).__name__}`")
                st.write(f"Features supplied to model: `{X.shape[1]}`")
                st.write(f"Predicted log price: `{log_prediction:.6f}`")
                st.write(f"Normalized ZIP: `{location_debug['zip_code']}`")
                st.write(
                    "Resolved district: "
                    f"`{location_debug['district'] or 'Not found'}`"
                )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


with st.sidebar:
    st.header("Deployment note")
    st.write(
        "Keep `app.py` and `model.pkl` in the same repository directory. "
        "The saved model bundle is loaded with joblib."
    )
    st.write(
        "Required Python packages: streamlit, joblib, numpy, pandas, lightgbm, scikit-learn"
    )
