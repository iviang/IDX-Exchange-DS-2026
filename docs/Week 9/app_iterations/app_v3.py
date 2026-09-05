from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from geopy.geocoders import GoogleV3, Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(
    page_title="California Home Price Router",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    """Load models.pkl from the same directory as app.py."""
    model_path = Path(__file__).resolve().parent / "models.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "models.pkl was not found. Put models.pkl in the same directory as app.py."
        )
    return joblib.load(model_path)


bundle = load_bundle()

ROUTER_THRESHOLD = float(bundle["threshold"])
router_lgb_log = bundle["lgb_log"]

stack = bundle["stack"]
stack_order = stack["order"]
stack_base = stack["base"]
stack_ridge = stack["ridge"]

feature_d = bundle["feature_d"]
native_columns = bundle["native"]
cat_cols = bundle["cat_cols"]

zip_lut = bundle["zip_lut"]
city_lut = bundle["city_lut"]
district_lut = bundle["district_lut"]
zip_to_district = bundle["zip_to_district"]
city_to_district = bundle["city_to_district"]


def county_feature_to_name(feature_name: str) -> str:
    return feature_name.removeprefix("County_").replace("_", " ")


county_feature_map = {
    county_feature_to_name(col): col
    for col in feature_d
    if col.startswith("County_")
}


# Feature engineering
def normalize_zip(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return ""
    if len(digits) >= 5:
        return digits[:5]
    return digits.zfill(5)


def resolve_district(zip_code: str, city: str):
    district = zip_to_district.get(zip_code)
    if district is None:
        district = city_to_district.get(city)
    return district

# still using geocoder api
@st.cache_resource
def get_geocoder():
    try:
        key = st.secrets["google_maps_api_key"]
    except Exception:
        key = None
    if key:
        locator = GoogleV3(api_key=key, timeout=10)
        return RateLimiter(locator.geocode, min_delay_seconds=0.2), "google"
    locator = Nominatim(user_agent="ca-house-price-router", timeout=10)
    return (
        RateLimiter(
            lambda q: locator.geocode(q, addressdetails=True, country_codes="us"),
            min_delay_seconds=1.0,
        ),
        "nominatim",
    )


def parse_components(location, backend):
    """Pull ZIP / City / County (display form) out of a geopy result."""
    zip_code = city = county = None
    if backend == "google":
        for comp in location.raw.get("address_components", []):
            types = comp.get("types", [])
            if "postal_code" in types:
                zip_code = comp["long_name"]
            elif "locality" in types:
                city = comp["long_name"]
            elif "administrative_area_level_2" in types:
                county = comp["long_name"]
    else:  # nominatim
        addr = location.raw.get("address", {})
        zip_code = addr.get("postcode")
        city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")
        county = addr.get("county")
    if county:
        county = county.replace(" County", "").strip()
    return zip_code, city, county


def build_model_inputs(
    *,
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
    """Build both representations the router bundle needs: the 1,547-col one-hot
    row (LGBM-log + stack LightGBM/XGBoost) and the 22-col native row (CatBoost)."""
    if living_area <= 0:
        raise ValueError("Living area must be greater than 0.")
    if lot_size_acres <= 0:
        raise ValueError("Lot size must be greater than 0.")

    zip_code = normalize_zip(postal_code)
    district = resolve_district(zip_code, city)

    structural_values = {
        "BedroomsTotal": int(bedrooms),
        "BathroomsTotalInteger": int(bathrooms),
        "LotSizeAcres_imputed": 0,
        "YearBuilt": int(year_built),
        "Stories": int(stories),
        "GarageSpaces": int(garage_spaces),
        "ParkingTotal": int(parking_total),
        "ViewYN": int(view_yn),
        "BasementYN": int(basement_yn),
        "PoolPrivateYN": int(pool_private_yn),
        "AttachedGarageYN": int(attached_garage_yn),
        "FireplaceYN": int(fireplace_yn),
        "NewConstructionYN": int(new_construction_yn),
        "Latitude": float(latitude),
        "Longitude": float(longitude),
        "YearBuilt_imputed": 0,
        "LivingArea_log": float(np.log1p(living_area)),
        "LotSizeAcres_log": float(np.log(lot_size_acres)),
    }

    # One-hot representation
    onehot_df = pd.DataFrame(np.zeros((1, len(feature_d)), dtype=float), columns=feature_d)
    for column, value in structural_values.items():
        if column in onehot_df.columns:
            onehot_df.at[0, column] = value

    county_column = county_feature_map.get(county)  # Los Angeles = reference (all 0)
    if county_column in onehot_df.columns:
        onehot_df.at[0, county_column] = 1

    zip_column = zip_lut.get(zip_code, "Zip_Other")
    if zip_column in onehot_df.columns:
        onehot_df.at[0, zip_column] = 1

    city_column = city_lut.get(city, "City_Other")
    if city_column in onehot_df.columns:
        onehot_df.at[0, city_column] = 1

    district_column = district_lut.get(district)
    if district_column in onehot_df.columns:
        onehot_df.at[0, district_column] = 1

    onehot_df = onehot_df[feature_d]

    # Native CatBoost representation
    native_values = {
        **structural_values,
        "PostalCode": zip_code,
        "City": city,
        "CountyOrParish": county,
        "SchoolDistrict": district if district is not None else "Unknown",
    }
    native_df = pd.DataFrame([native_values], columns=native_columns)
    for column in cat_cols:
        native_df[column] = native_df[column].fillna("Unknown").astype(str)

    location_info = {
        "zip_code": zip_code,
        "zip_feature": zip_column,
        "city_feature": city_column,
        "district": district,
        "district_feature": district_column,
        "county_feature": county_column,
    }
    return onehot_df, native_df, location_info


# Model prediction + router switching
def predict_with_router(onehot_df: pd.DataFrame, native_df: pd.DataFrame):
    """LGBM-log makes a ballpark; <= threshold -> Stack-Ridge, else LGBM-log. Switched, not blended."""
    router_log_prediction = float(router_lgb_log.predict(onehot_df)[0])
    router_price = float(np.exp(router_log_prediction))

    result = {
        "router_log_prediction": router_log_prediction,
        "router_price": router_price,
        "threshold": ROUTER_THRESHOLD,
        "route": None,
        "final_price": None,
        "base_predictions": {},
        "stack_prediction": None,
    }

    if router_price > ROUTER_THRESHOLD:
        result["route"] = "LGBM-log"
        result["final_price"] = router_price
        return result

    base_predictions = []
    for model_name in stack_order:
        model_info = stack_base[model_name]
        model = model_info["model"]
        representation = model_info["rep"]
        if representation == "onehot-district":
            X = onehot_df
        elif representation == "native":
            X = native_df
        else:
            raise ValueError(f"Unknown representation '{representation}' for model '{model_name}'.")
        prediction = float(model.predict(X)[0])
        base_predictions.append(prediction)
        result["base_predictions"][model_name] = prediction

    meta_X = np.asarray(base_predictions, dtype=float).reshape(1, -1)
    stack_prediction = float(stack_ridge.predict(meta_X)[0])

    result["route"] = "Stack-Ridge"
    result["stack_prediction"] = stack_prediction
    result["final_price"] = stack_prediction
    return result


# User interface
st.title("🏠 California Home Price Predictor v3")
st.write(
    "Enter a property address and its characteristics. The app estimates the "
    "price band, then switches to the model intended for that band."
)

route_a, route_b = st.columns(2)
with route_a:
    st.info(f"**Ballpark ≤ ${ROUTER_THRESHOLD:,.0f}:** Stack-Ridge (LightGBM + XGBoost + CatBoost)")
with route_b:
    st.info(f"**Ballpark > ${ROUTER_THRESHOLD:,.0f}:** LGBM-log luxury-tail model")

geocode, backend = get_geocoder()
if backend == "nominatim":
    st.caption("Using the free OpenStreetMap geocoder. Add a `google_maps_api_key` in the "
               "app's Streamlit **Secrets** to switch to Google Maps automatically.")

with st.form("prediction_form"):
    address = st.text_input("📍 Property address", value="6175 Oneida Drive, San Jose, CA 95123")

    st.subheader("Property details")
    col1, col2 = st.columns(2)

    with col1:
        living_area = st.number_input("Living area (sq ft)", min_value=1, value=1409, step=50)

        lot_unit = st.radio("Lot size unit", ["Acres", "Square feet"], horizontal=True)
        if lot_unit == "Acres":
            lot_size_acres = st.number_input("Lot size (acres)", min_value=0.0001,
                                             value=0.138, step=0.01, format="%g",
                                             key="lot_acres")
        else:
            lot_sqft = st.number_input("Lot size (sq ft)", min_value=1, value=6018,
                                       step=100, key="lot_sqft")
            lot_size_acres = lot_sqft / 43560.0  # 43,560 sq ft = 1 acre
            st.caption(f"≈ {lot_size_acres:g} acres")

        bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)

    with col2:
        year_built = st.number_input("Year built", min_value=1800, max_value=2100, value=1968, step=1)
        stories = st.number_input("Stories", min_value=0, value=1, step=1)
        garage_spaces = st.number_input("Garage spaces", min_value=0, value=2, step=1)
        parking_total = st.number_input("Total parking spaces", min_value=0, value=2, step=1)

    st.subheader("Property features")
    flag1, flag2, flag3 = st.columns(3)
    with flag1:
        view_yn = st.checkbox("View")
        basement_yn = st.checkbox("Basement")
    with flag2:
        pool_private_yn = st.checkbox("Private pool")
        attached_garage_yn = st.checkbox("Attached garage")
    with flag3:
        fireplace_yn = st.checkbox("Fireplace")
        new_construction_yn = st.checkbox("New construction")

    submitted = st.form_submit_button("Predict closing price", type="primary",
                                      use_container_width=True)


# Prediction output
if submitted:
    try:
        if not address.strip():
            st.error("Please enter a property address.")
        else:
            location = geocode(address)
            if location is None:
                st.error("Could not find that address. Try adding city, state, and ZIP.")
            else:
                zip_code, city, county = parse_components(location, backend)

                onehot_df, native_df, location_info = build_model_inputs(
                    living_area=living_area,
                    lot_size_acres=lot_size_acres,
                    bedrooms=bedrooms,
                    bathrooms=bathrooms,
                    year_built=year_built,
                    stories=stories,
                    garage_spaces=garage_spaces,
                    parking_total=parking_total,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    county=county,
                    postal_code=zip_code or "",
                    city=city,
                    view_yn=view_yn,
                    basement_yn=basement_yn,
                    pool_private_yn=pool_private_yn,
                    attached_garage_yn=attached_garage_yn,
                    fireplace_yn=fireplace_yn,
                    new_construction_yn=new_construction_yn,
                )

                prediction = predict_with_router(onehot_df, native_df)

                st.divider()
                st.subheader("Prediction")
                metric1, metric2, metric3 = st.columns(3)
                with metric1:
                    st.metric("Estimated closing price", f"${prediction['final_price']:,.0f}")
                with metric2:
                    st.metric("Router ballpark", f"${prediction['router_price']:,.0f}")
                with metric3:
                    st.metric("Selected route", prediction["route"])

                if prediction["route"] == "Stack-Ridge":
                    st.success(f"Ballpark was at or below ${ROUTER_THRESHOLD:,.0f} — Stack-Ridge used.")
                else:
                    st.success(f"Ballpark was above ${ROUTER_THRESHOLD:,.0f} — LGBM-log luxury-tail used.")

                loc1, loc2 = st.columns(2)
                loc1.metric("Latitude", f"{location.latitude:.5f}")
                loc2.metric("Longitude", f"{location.longitude:.5f}")

                with st.expander("Model routing details"):
                    st.write(f"**Router threshold:** ${prediction['threshold']:,.0f}")
                    st.write(f"**LGBM-log ballpark:** ${prediction['router_price']:,.0f}")
                    st.write(f"**Selected route:** {prediction['route']}")
                    if prediction["base_predictions"]:
                        st.markdown("**Stack base-model predictions**")
                        for model_name, model_prediction in prediction["base_predictions"].items():
                            st.write(f"- {model_name}: ${model_prediction:,.0f}")
                        st.write(f"**Ridge stack output:** ${prediction['stack_prediction']:,.0f}")
                    else:
                        st.write("Stack not executed — luxury-tail LGBM-log route selected.")

                with st.expander("Location & input details"):
                    st.write(f"**Geocoded ZIP:** {zip_code or 'Not provided'} "
                             f"| **City:** {city or 'n/a'} | **County:** {county or 'n/a'}")
                    st.write(f"**Resolved school district:** {location_info['district'] or 'Not found'}")
                    st.write(f"**One-hot input shape:** {onehot_df.shape[0]} × {onehot_df.shape[1]}")
                    st.write(f"**Native CatBoost input shape:** {native_df.shape[0]} × {native_df.shape[1]}")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# Sidebar
with st.sidebar:
    st.header("Router deployment")
    st.write("Place these files together in the Streamlit repository:")
    st.code("app.py\nmodels.pkl\nrequirements.txt", language="text")
    st.write(f"**Switch threshold:** ${ROUTER_THRESHOLD:,.0f}")
    st.caption("The router is a hard switch between model outputs; the LGBM-log and "
               "Stack-Ridge predictions are not blended.")