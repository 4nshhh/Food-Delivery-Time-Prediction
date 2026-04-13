import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


st.title("Food Delivery Time Prediction System")
st.caption("Machine Learning powered delivery time estimation using XGBoost")

st.subheader("Project Description")

st.write("""
This app estimates food delivery time using an **XGBoost regression model**
trained on ~45,000 delivery records.  

The model considers multiple real-world factors such as **distance, traffic density,
weather conditions, delivery agent rating, vehicle type, and order characteristics**
to provide an estimated delivery time.
""")

st.markdown("---")

st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="R² Score", value="0.78")
with col2:
    st.metric(label="MAE", value="3.46 min")
with col3:
    st.metric(label="RMSE", value="4.35 min")

st.markdown("---")


col4, col5, col6 = st.columns(3)

with col4:

    distance = st.number_input("Distance(km):", min_value=4.0, value=5.0, step=0.1, max_value=13.0)
    rating = st.slider("Delivery Person Rating:", min_value=1.0,max_value=5.0, step=0.1,value=4.2)
    age = st.number_input("Delivery Person Age:",min_value=18,max_value=60,step=1)
    order_date = st.date_input("Order Date:")
    no_of_deliveries = st.selectbox("Number Of Deliveries:", [0, 1, 2, 3])

with col5:
    order_time = st.time_input("Order Time:", step=300)
    pickup_time = st.time_input("Pickup Time: ", step=300)
    weather = st.selectbox("Weather Condition:",['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy'])
    traffic = st.selectbox("Traffic Intensity:", ["Low","Medium","High","Jam"] )
    vehicle_condition = st.selectbox("Vehicle Condition:",["Good", "Average", "Bad"])

with col6:
    order_type = st.selectbox("Order Type:",['Snack ', 'Drinks ', 'Buffet ', 'Meal '])
    vehicle_type = st.selectbox("Vehicle Type:",['Motorcycle', 'Scooter', 'Electric Scooter'])
    festival = st.selectbox("Festival:",['No ', 'Yes '])
    city_type = st.selectbox("City Type:",['Urban ', 'Metropolitian ', 'Semi-Urban '])

# st.write(distance, rating,age, order_date,no_of_deliveries,t, pickup_time, weather, traffic, vehicle_condition, order_type, vehicle_type, festival, city_type)

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Extract Features
        X["order_hour"] = X["Time_Orderd"].dt.hour # Extracting just hour of order
        X["order_weekday"] = X["Order_Date"].dt.weekday # The day on which order took place
        X["pickup_hour"] = X["Time_Order_picked"].dt.hour # Hour of order pickup

        # Dropping Unnecessary Columns
        X = X.drop(columns = ["Time_Orderd", "Order_Date", "Time_Order_picked", "Delivery_person_ID", "ID"], axis = 1, errors="ignore")

        return X

traffic_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Jam": 4
}

# ['motorcycle ', 'scooter ', 'electric_scooter ']

condition_map = {
    "Good" : 2,
    "Average" : 1,
    "Bad" : 0
}

vehicle_map = {
    "Motorcycle" : "motorcycle ",
    "Scooter" : "scooter ",
    "Electric Scooter" : "electric_scooter "
}

traffic = traffic_map[traffic]
vehicle_condition = condition_map[vehicle_condition]
vehicle_type = vehicle_map[vehicle_type]
no_of_deliveries = float(no_of_deliveries)
age = float(age)
order_time = pd.to_datetime(order_date.strftime("%Y-%m-%d") + " " + order_time.strftime("%H:%M:%S"))
order_date = pd.to_datetime(order_date.strftime("%Y-%m-%d"))
pickup_time = pd.to_datetime(order_date.strftime("%Y-%m-%d") + " " + pickup_time.strftime("%H:%M:%S"))

input_dict = {
    'Delivery_person_Age' : age,
    'Delivery_person_Ratings' : rating,
    'Order_Date' : order_date,
    'Time_Orderd' : order_time,
    'Time_Order_picked' : pickup_time,
    'Weather_conditions' : weather,
    'Road_traffic_density' : traffic,
    'Vehicle_condition' : vehicle_condition,
    'Type_of_order' : order_type,
    'Type_of_vehicle' : vehicle_type,
    'multiple_deliveries' : no_of_deliveries,
    'Festival' : festival,
    'City_type' : city_type,
    'Distance(km)' : distance
}

input_df = pd.DataFrame([input_dict])

@st.cache_resource
def load_model():
    return joblib.load("delivery_time_pipeline.pkl")

pipeline = load_model()

predict = st.button("Predict")

if predict:
    if pickup_time < order_time:
        st.warning("Pickup time cannot be earlier than order time")
    else:
        prediction = pipeline.predict(input_df)
        speed = distance/(float(prediction)/60)
        lower = prediction - 4
        upper = prediction + 4
        st.subheader("Prediction Result")
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #16a34a, #22c55e);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        ">
        🚀 Estimated Delivery Time: {prediction[0]:.2f} minutes
        </div>
        """, unsafe_allow_html=True)

        st.write(f"Range: {lower[0]:.2f} - {upper[0]:.2f} minutes")
        st.info(f"Estimated Delivery Speed: {speed:.2f} km/h")

st.write("---")

st.components.v1.html("""
<div style="
    text-align: center;
    font-size: 14px;
    color: #94a3b8;
    padding: 10px;
">
    👨‍💻 <b>Ansh Panchal</b> | AIML Student<br>
    📊 Passionate about Data Science & ML Engineering<br><br>

    🔗 Connect with me:<br>
    <a href="https://www.linkedin.com/in/4nshh/" target="_blank" style="color:#22c55e;">LinkedIn</a> |
    <a href="https://github.com/4nshh" target="_blank" style="color:#22c55e;">GitHub</a>
</div>
""", height=150)