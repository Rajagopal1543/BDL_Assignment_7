# use python runtime as parent image
FROM python:3.10-slim

# set working dir
WORKDIR /app

# copy current dir
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# For fastapi
EXPOSE 8000

# For prometheus
EXPOSE 8001

# Run fastapi
CMD ["uvicorn", "task1:app", "--host", "0.0.0.0", "--port", "8000"]