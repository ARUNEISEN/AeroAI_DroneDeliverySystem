# Drone Delivery ETA Prediction

Predict the **Estimated Time of Arrival (ETA)** for drone deliveries using historical data.  
This helps improve operational efficiency, customer satisfaction, and resource planning.

---

## Dataset

Key columns:

- `order_id`, `drone_id`, `drone_type`  
- `distance_km`, `payload_weight_kg`, `drone_speed_kmph`  
- `battery_efficiency`, `climate_condition`  
- `wind_speed_kmph`, `temperature_c`, `humidity_percent`  
- `source_area`, `destination_area`, `traffic_condition`  
- `ETA_actual_min` (target)

---

## Features & Engineering

- Log transformations: `distance_km_log`, `payload_kg_log`  
- Physics-based ETA: `ETA_physics = distance / speed * 60`  
- Battery penalty for low efficiency  
- Weather severity scoring  
- Missing value indicators

---

## Models Tested

- **Gradient Boosting Regressor** ✅ Best  
- Random Forest Regressor  
- Ridge Regression  
- Linear Regression  
- Lasso Regression  

**Evaluation (Best Model):**

| Metric | Value      |
|--------|------------|
| MAE    | 1.344 min  |
| RMSE   | 1.712 min  |
| R²     | 0.882      |

---

