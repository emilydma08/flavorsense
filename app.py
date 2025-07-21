from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim=1, num_layers=1, dropout_rate=0):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.output_layer = nn.Linear(self.hidden_dim, 4)

    def forward(self, x):
        a = self.input_layer(x)
        for _ in range(self.num_layers):
            a = self.hidden_layer(a)
        a = self.output_layer(a)
        return a

# Load the scaler and column order
scaler = joblib.load("scaler.pkl")
column_order = joblib.load("one_hot_columns.pkl")

# Initialize the model
input_size = len(column_order)
model = NeuralNetwork(input_size=input_size, hidden_dim=128, num_layers=4, dropout_rate=0.5)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Grab form data
    user_input = {
        'age': float(request.form['age']),
        'sleep_cycle': request.form['sleep_cycle'],
        'exercise_habits': request.form['exercise_habits'],
        'climate_zone': request.form['climate_zone'],
        'historical_cuisine_exposure': request.form['historical_cuisine_exposure']
    }

    # Convert to DataFrame
    df = pd.DataFrame([user_input])

    # Scale age (in-place)
    df['age'] = scaler.transform(df[['age']])

    # One-hot encode categorical columns
    df = pd.get_dummies(df)

    # Add any missing columns (i.e., ones that weren't in user input but are in training data)
    for col in column_order:
        if col not in df:
            df[col] = 0

    # Ensure correct column order
    df = df[column_order]

    # Convert to float32 tensor
    input_tensor = torch.tensor(df.values.astype(np.float32))

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    label_map = {
    0: 'Sweet',
    1: 'Spicy',
    2: 'Sour',
    3: 'Salty',
    }

    category = label_map[prediction]
    return render_template("result.html", prediction=category)


if __name__ == '__main__':
    app.run(debug=True)
