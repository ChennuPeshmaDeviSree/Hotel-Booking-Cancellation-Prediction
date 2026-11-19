import streamlit as st
import requests

st.title("Hotel Booking Cancellation Prediction")

hotel=st.selectbox("Hotel Type",
    ['Resort Hotel', 'City Hotel'])

lead_time=st.number_input("Lead Time",0,737,342)

arrival_date_year=st.selectbox("Arrival_Year",[2015, 2016, 2017])

arrival_date_month=st.selectbox("Arrival_Month",['July', 'August', 'September', 'October', 'November', 'December','January', 'February', 'March', 'April', 'May', 'June'])

arrival_date_week_number=st.number_input("Arrival_Date_Week_Number",min_value=1,max_value=53,value=27)

arrival_date_day_of_month=st.number_input("Arrival_Date_Day_of_Month",min_value=1,max_value=31,value=1)

stays_in_weekend_nights=st.number_input("Stays_in_Weekend_Nights",min_value=0,max_value=19,value=0)

stays_in_week_nights=st.number_input("Stays_in_Week_Nights",min_value=0,max_value=50,value=0)

adults=st.number_input("Adults",min_value=0,max_value=55,value=2)

children=st.number_input("Children",min_value=0,max_value=10,value=0)

babies=st.number_input("Babies",min_value=0,max_value=10,value=0)

meal=st.selectbox("Meal",['BB', 'FB', 'HB', 'SC', 'Undefined'])

country = st.text_input("Country Code", value="PRT")

market_segment = st.selectbox("Market Segment",["Online TA", "Offline TA/TO", "Direct","Groups", "Corporate", "Complementary","Aviation", "Undefined"])

distribution_channel = st.selectbox("Distribution Channel",["TA/TO", "Direct", "Corporate", "GDS", "Undefined"])

is_repeated_guest = st.selectbox("Repeated Guest", [0, 1])

previous_cancellations = st.number_input("Previous Cancellations", min_value=0,max_value=26,value=0)

previous_bookings_not_canceled = st.number_input("Previous Bookings Not Cancelled", min_value=0,max_value=72,value=0)

reserved_room_type = st.selectbox("Reserved Room Type", ['C', 'A', 'D', 'E', 'G', 'F', 'H', 'L', 'P', 'B'])

assigned_room_type = st.selectbox("Assigned Room Type", ['C', 'A', 'D', 'E', 'G', 'F', 'I', 'B', 'H', 'P', 'L', 'K'])

booking_changes = st.number_input("Booking Changes", min_value=0,max_value=20,value=0)

deposit_type = st.selectbox("Deposit Type",["No Deposit", "Non Refund", "Refundable"])

agent = st.number_input("Agent ID", min_value=0, value=0)

company = st.number_input("Company ID", min_value=0, value=0)

days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, value=0)

customer_type = st.selectbox("Customer Type",["Transient", "Transient-Party", "Contract", "Group"])

adr = st.number_input("Average Daily Rate", min_value=0.0, value=100.0)

required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, value=0)

total_of_special_requests = st.number_input("Total Special Requests", min_value=0, value=0)

# Prediction Button
if st.button("Predict Cancellation"):
    data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country.strip().upper(),
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "agent": agent,
        "company": company,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests
    }
    try:
        API_URL = st.secrets.get(
            "API_URL",
            "https://hotel-booking-cancellation-prediction-1-ouze.onrender.com"
        )


        response = requests.post(
        f"{API_URL.rstrip('/')}/predict",
        json={"data": data},
        timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.error("⚠️ Booking is predicted to be CANCELLED")
            else:
                st.success("✅ Booking is predicted NOT to be CANCELLED")

            st.metric(
                "Cancellation Probability",
                f"{result['probability']:.2%}"
            )

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("FastAPI backend is not running. Start the backend first.")

    except Exception as e:
        st.error(f"Error: {e}")
