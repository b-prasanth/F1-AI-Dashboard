# F1-AI-Dashboard

An AI-powered dashboard for Formula 1 analysis, providing insights and statistics using extensive F1 data.

## Project Overview

This project combines a Python backend for data processing and AI modeling with a React frontend for visualizing F1 statistics, race data, and predictions.

## Tech Stack

- **Backend:** Python
- **Frontend:** React (within `f1-ui`)
- **Database:** SQL Database (MySQL)
- **AI/ML:** XGBoost, RAG implementation

## Project Structure

- `backend/`: Contains the Python backend code, models, and data processing scripts.
- `api/`: Contains the FastAPI/backend server endpoints (`api1.py`, `api2.py`, `api3.py`).
- `f1-ui/`: Contains the React frontend application.
- `F1_startup.sh`: Startup script for the project, which launches backend services and the frontend.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js and npm

### Backend Setup

1.  Navigate to the project root.
2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
3.  Activate the virtual environment:
    - macOS/Linux: `source venv/bin/activate`
    - Windows: `venv\Scripts\activate`
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Frontend Setup

1.  Navigate to the `f1-ui` directory:
    ```bash
    cd f1-ui
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```

## Running the Application

To start the application, you can use the provided startup script:

```bash
./F1_startup.sh
```

Alternatively, you can run the backend and frontend separately.

## License

[License Name]
