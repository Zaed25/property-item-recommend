# Use an official, lightweight Python image as the starting point
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from your project into the container
COPY requirements.txt .

# Run the pip install command inside the container to install all needed libraries
# --no-cache-dir is an optimization that keeps the container size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of your project files (main.py, .pkl, .keras, etc.) into the container
COPY . .

# The command to run when the container starts.
# This tells Uvicorn to run the 'app' object from the 'main' file.
# --host 0.0.0.0 is crucial for making the app accessible from the internet.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]