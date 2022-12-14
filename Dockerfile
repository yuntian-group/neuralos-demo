FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN mkdir -p /data
RUN chmod 777 /data
ENV HUGGINGFACE_HUB_CACHE=/data

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]