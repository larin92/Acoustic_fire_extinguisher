# Use the slim version of Python 3.9
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock into the container at /app
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies in the container
RUN pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install --system --deploy

# Copy the rest of your application's code into the container at /app
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Command to run your application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]