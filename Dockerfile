# Use official Python 3.11 slim image
FROM python:3.11.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Expose the port your Flask app runs on (Render default is 10000)
EXPOSE 10000

# Command to run your app with Gunicorn on port 10000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
