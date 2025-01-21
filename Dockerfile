FROM python:3-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY . .


RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir


ENTRYPOINT ["python", "-u", "src/chess_board_recognizer/train.py"]
