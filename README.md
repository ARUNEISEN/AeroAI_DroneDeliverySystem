🚁 Aero AI – Intelligent Multi-Agent Drone Delivery System

    An end-to-end AI-powered Drone Intelligence & Delivery Analytics Platform integrating:

        🤖 Multi-Agent LLM System

        🧠 Machine Learning (Gradient Boosting)

        👁 Computer Vision (YOLOv8)

        ☁ AWS Cloud Services (RDS, S3, SES)

        📊 Dynamic AI-SQL Analytics

📌 Project Overview

        Aero AI is a production-ready intelligent drone monitoring and delivery system that:

        Detects drone type & health condition using YOLO

        Stores uploaded images in Amazon S3

        Stores metadata in Amazon RDS

        Predicts drone delivery ETA using Gradient Boosting ML model

        Dynamically generates SQL queries using LLM

        Generates PDF reports

        Sends reports via email using Amazon SES

        Provides conversational AI interface using Streamlit

🏗 System Architecture
    1️⃣ Computer Vision Pipeline (YOLO)

        User uploads drone image

        YOLOv8 model detects:

        Drone Type (multi_rotor, single_rotor, hybrid)

        Drone Health (missing_part, wing_damage, healthy)

        Metadata stored in:

        📦 Amazon S3 (Image storage)

        🗄 Amazon RDS (Metadata storage)

    2️⃣ Machine Learning – ETA Prediction

        Drone delivery time is predicted using:

        🌳 Gradient Boosting Regressor

        Features:

        Distance

        Drone type

        Weather conditions

        Payload weight

        Historical delivery data

        Model predicts:

        📦 Estimated Time of Arrival (ETA) for drone delivery

    3️⃣ AI Multi-Agent System

        The system uses an LLM-powered routing architecture:

        Agent	Responsibility
        Data Agent	Converts natural language → SQL → Fetches from RDS
        Report Agent	Generates PDF reports
        Email Agent	Sends reports using SES
        ML Agent	Handles ETA predictions
        CV Agent	Handles YOLO inference
        Router Agent	Decides which agent to call

        LLM: Local inference using Ollama (Mistral)

☁ AWS Cloud Integration
        🗄 Database

        Amazon RDS

        Stores drone metadata:

        drone_type

        health_status

        confidence scores

        upload timestamps

        model version

        📦 Image Storage

        Amazon S3

        Stores uploaded drone images

        📧 Email Service

        Amazon Simple Email Service

        Sends reports and notifications

        📊 Example User Queries

        Users can ask:

        🔍 Data Queries

        How many wing damage drones?

        Number of multi-rotor drones uploaded?

        Latest drone status?

        Drone type distribution?

        📈 ML Queries

        Predict delivery time for 12 km multi-rotor drone

        ETA for hybrid drone with 2kg payload

        📄 Report Queries

        Generate health report

        Generate drone performance report

        📧 Email Queries

        Send report to my email

        🧠 Machine Learning Model
        🎯 Algorithm

Gradient Boosting Regressor

    Why Gradient Boosting?

        Handles non-linear relationships

        Robust to overfitting

        Performs well on tabular delivery data

        High prediction accuracy

👁 Computer Vision Model
        YOLOv8 Model Capabilities:

        Drone Type Classification

        Drone Health Detection

    
🏛 Multi-Agent Architecture
        User
        ↓
        Streamlit UI
        ↓
        LLM Router Agent
        ↓
        ---------------------------------
        | Data Agent → RDS              |
        | ML Agent → ETA Model          |
        | CV Agent → YOLO Model         |
        | Report Agent → PDF Generator  |
        | Email Agent → SES             |
        ---------------------------------

🔥 Key Features

        ✅ AI-powered SQL generation
        ✅ Secure SELECT-only execution
        ✅ Conversation memory
        ✅ Dynamic routing using LLM
        ✅ ML-based ETA prediction
        ✅ YOLO-based CV detection
        ✅ Cloud-native architecture
        ✅ Placement-ready modular structure

🗂 Project Structure
        aero-ai_dronedeliverysystem/
        │
        ├── agents/
        │   ├── orchestrator.py
        │   ├── Data_Agent.py
        │   ├── ML_Agent.py
        │   ├── CV_Agent.py
        │   ├── Report_Agent.py
        │   ├── Email_Agent.py
        │   ├── report_generator.py
        │   └── email_service.py
        │
        ├── models/
        │   ├── eta_gradient_boost.pkl
        │   ├── yolov8_drone.pt
        │
        ├── Config.py
        ├── app.py
        ├── requirements.txt
        └── README.md


👨‍💻 Author

        Arunkumar Sekar
        AI | Automation | Data Science Enthusiast
        Building scalable AI-driven cloud systems.