FROM python:3.12

WORKDIR /root

COPY .github/requirements.txt .github/requirements.txt

RUN apt update && apt install poppler-utils -y && \
    cd .github && pip install --index-url https://pypi.org/simple --no-cache-dir -r requirements.txt

COPY . .
CMD python3 main.py http