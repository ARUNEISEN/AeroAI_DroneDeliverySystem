def handle_ml_query(action, parameters):
    if action == "eta_info":
        return "ETA model MAE: 1.34 minutes. Gradient Boosting selected."

    elif action == "detection_info":
        return "YOLOv8 model with 95.2% mAP accuracy."

    return "Unknown ML request."
