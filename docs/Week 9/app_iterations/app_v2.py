from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- V2: address geocoding---
from geopy.geocoders import GoogleV3, Nominatim
from geopy.extra.rate_limiter import RateLimiter

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

# V2: geocoding
@st.cache_resource
def get_geocoder():
    """Use Google Maps if a key is in Streamlit secrets."""
    try:
        key = st.secrets["google_maps_api_key"]
    except Exception:
        key = None
 
    if key:
        locator = GoogleV3(api_key=key, timeout=10)
        return RateLimiter(locator.geocode, min_delay_seconds=0.2), "google"
 
    locator = Nominatim(user_agent="ca-house-price-predictor", timeout=10)
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
    "Enter a property address and its characteristics to estimate the home's "
    "closing price. Coordinates and neighborhood are looked up automatically."
)

geocode, backend = get_geocoder()
if backend == "nominatim":
    st.info("Using the free OpenStreetMap geocoder. Add a `google_maps_api_key` in "
            "the app's Streamlit **Secrets** to switch to Google Maps automatically.")
    
with st.form("prediction_form"):
    address = st.text_input(
        "📍 Property address",
        value="6175 Oneida Drive, San Jose, CA 95123"
    )

    st.subheader("Property details")

    col1, col2 = st.columns(2)

    with col1:
        living_area = st.number_input("Living area (sqft)", min_value=1, value=1800, step=50)
        lot_unit = st.radio("Lot size unit", ["Acres", "Square feet"], horizontal=True)
        if lot_unit == "Acres":
            lot_size_acres = st.number_input("Lot size (acres)", min_value=0.0001,
                                             value=0.15, step=0.01, format="%.g",
                                             key="lot_acres")
        else:
            lot_sqft = st.number_input("Lot size (sq ft)", min_value=1,
                                       value=6534, step=100, key="lot_sqft")
            lot_size_acres = lot_sqft / 43560.0  # 43,560 sq ft = 1 acre
            st.caption(f"≈ {lot_size_acres:.g} acres")

        bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)

    with col2:
        year_built = st.number_input("Year built", min_value=1800, max_value=2100, value=1990, step=1)
        stories = st.number_input("Stories", min_value=0, value=1, step=1)
        garage_spaces = st.number_input("Garage spaces", min_value=0, value=2, step=1)
        parking_total = st.number_input("Total parking spaces", min_value=0, value=2, step=1)

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
        if not address.strip():
            st.error("Please enter a property address.")
        elif lot_size_acres <= 0:
            st.error("Lot size must be greater than 0 acres.")
        elif living_area <= 0:
            st.error("Living area must be greater than 0 square feet.")
        else:
            location = geocode(address)
            if location is None:
                st.error("Could not find that address. Try adding city, state, and ZIP.")
            else:
                zip_code, city, county = parse_components(location, backend)
 
                X, location_debug = build_feature_row(
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
 
                log_prediction = float(model.predict(X)[0])
                predicted_price = float(np.exp(log_prediction))
 
                st.success("Prediction complete")
                st.metric("Estimated closing price", f"${predicted_price:,.0f}")
 
                m1, m2 = st.columns(2)
                m1.metric("Latitude", f"{location.latitude:.5f}")
                m2.metric("Longitude", f"{location.longitude:.5f}")
 
                with st.expander("Prediction details"):
                    st.write(f"Model class: `{type(model).__name__}`")
                    st.write(f"Features supplied to model: `{X.shape[1]}`")
                    st.write(f"Predicted log price: `{log_prediction:.6f}`")
                    st.write(f"Geocoded ZIP: `{location_debug['zip_code']}`  "
                             f"(county: `{county or 'not found'}`)")
                    st.write(f"Resolved district: `{location_debug['district'] or 'Not found'}`")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


with st.sidebar:
    st.header("Deployment note")
    st.write(
        "Keep `app.py` and `model.pkl` in the same repository directory. "
        "The saved model bundle is loaded with joblib."
    )
    st.write(
        "Required Python packages: streamlit, joblib, numpy, pandas, lightgbm, scikit-learn, geopy."
    )
