# Use the official Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY /api/requirements.txt .

# Install dependencies from the `requirements.txt`
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the `api` folder into the container
COPY ./api /app

#workers = 2 * CPU cores + 1

# Command to run the FastAPI application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80", "main:app"]
