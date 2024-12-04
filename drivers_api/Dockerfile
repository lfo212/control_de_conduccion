FROM python:3.8-alpine


COPY requirements_api.txt ./
RUN pip3.8 install --no-cache-dir -r requirements_api.txt
RUN mkdir photos
COPY . .

# Make port 8000 available outside this container
EXPOSE 8000

ENTRYPOINT ["uvicorn", "drivers_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--ssl-keyfile=localhost+2-key.pem", "--ssl-certfile=localhost+2.pem"]