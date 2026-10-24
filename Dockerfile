# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the packages specified in your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (code, model pickle, data) into the container
COPY . .

# Expose the default port that Streamlit runs on
EXPOSE 8501

# Run the Streamlit application when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]