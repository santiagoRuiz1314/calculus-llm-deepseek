FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py calculus_solver.py ./
COPY calculus_examples.json ./

# Create a data directory for persisting example data
RUN mkdir -p /data
ENV EXAMPLES_FILE=/data/calculus_examples.json

# Copy the examples if not exists
RUN if [ ! -f /data/calculus_examples.json ]; then cp calculus_examples.json /data/; fi

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
