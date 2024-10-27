## Electricity-Demand-Projection-Model

This project is an LSTM-based model to forecast electricity demand, using historical consumption data and environmental features. The model takes in past consumption, temperature, and other temporal features to make predictions over a given date range.

## Table of Contents
- Project Overview
- Dataset
- Features
- Setup in Google Collab
- Usage
- Model Structure
- Results

## Project Overview
This project leverages a Long Short-Term Memory (LSTM) model to predict electricity consumption over a future time period. The model is trained on past consumption data alongside other features such as temperature, day of the week, and month to enhance prediction accuracy. 

## Dataset
The dataset used in this project must contain historical records of the following:
- **Date**: Date of the record
- **Consumption**: Electricity consumption on the given date
- **Temperature**: Ambient temperature on the given date
- **Holiday**: Binary indicator for whether the day was a holiday (1) or not (0)

## Features
The model extracts and scales the following features from the dataset:
- **Consumption**: Target variable (past consumption values)
- **Temperature**: Daily temperature value
- **Holiday**: Indicates if a day is a holiday
- **Day**: Day of the month
- **Month**: Month of the year
- **DayOfWeek**: Day of the week (0-6)

## Setup in Google Colab
1. **Upload Dataset**: Upload `dataset.csv` to the Colab environment.
2. **Install Dependencies**: Google Colab includes most libraries by default, but ensure the following libraries are installed:
   ```python
   pip install numpy pandas scikit-learn keras matplotlib

## Usage
1. Run the Code Cells: Ensure all code cells are executed in order.
2. Upload Data:
   - Use Colab's file upload feature to upload dataset.csv.
3. Provide Prediction Dates: Enter the start and end dates for prediction when prompted.

## Model Structure
The model is built with Keras and includes the following layers:

- LSTM Layer 1: 50 units, returns sequences
- Dropout Layer 1: 20% dropout
- LSTM Layer 2: 50 units, no return sequences
- Dropout Layer 2: 20% dropout
- Dense Layer: 25 units, followed by a final Dense layer with 1 unit (output layer)

## Results
After running all cells:

- Predicted Output: Predicted consumption values for the given date range are printed.
- Plot: The model visualizes the predicted electricity consumption over time.

## Dependencies

Google Colab (with the following libraries installed by default):
- numpy
- pandas
- scikit-learn
- keras
- matplotlib

## Link to download the dataset
- https://drive.google.com/file/d/1GRwWq6RmEMiUxzNL7sgg3cIdVvBJz-vn/view?usp=sharing
