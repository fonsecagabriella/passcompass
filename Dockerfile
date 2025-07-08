FROM python:3.10-slim

# install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
WORKDIR /app
COPY webapp/ .

# Flask listens on 8080 by default in your code
EXPOSE 8080

# default envs (overwritable at `docker run -e`)
ENV ENVIRONMENT=local \
    MODEL_NAME=passcompass_generic \
    MODEL_ALIAS=best_202506 \
    MLFLOW_TRACKING_URI=0.0.0.0

# production server (can also keep `python app.py` for dev)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

