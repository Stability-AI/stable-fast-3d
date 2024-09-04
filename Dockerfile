FROM python:slim

WORKDIR /app
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git build-essential python3-setuptools
RUN git clone https://github.com/Stability-AI/stable-fast-3d.git
WORKDIR /app/stable-fast-3d
COPY server.py .
COPY env.server .env
COPY requirements.txt .
RUN mkdir model
RUN rm __init__.py
RUN python -m pip install torch torchvision torchaudio setuptools==69.5.1 wheel
RUN python -m pip install -r requirements.txt
CMD ["fastapi", "run", "/app/stable-fast-3d/server.py", "--port", "8000"]
