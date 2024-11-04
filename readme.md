# Project Setup Guide (Windows)

This guide provides instructions for setting up a Python virtual environment on Windows and installing the required dependencies.

## Prerequisites

- **Python 3.7** or later
- **pip** package manager
- OPENAI API KEY

## Please add you'r OPENAI KEY in .env file

## Step 1: Create a Virtual Environment

To isolate dependencies for this project, create a virtual environment.

1. Open Command Prompt or PowerShell and navigate to your project directory:

   ```powershell
   cd path\to\your\project

   python -m venv venv

## Step 3: Activate the Virtual Environment
   .\venv\Scripts\activate

## Step 3: Install Dependencies

   pip install -r requirements.txt

## Step 4: Verify Installation

    pip list
## Step 5: Deactivate the Virtual Environment

    deactivate
## Step6: Run app using streamlit

    streamlit run stock_prediction_app.py
