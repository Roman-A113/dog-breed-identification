FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements-base.txt .

RUN pip install --no-cache-dir -r requirements-base.txt

COPY solve.py .

ENTRYPOINT ["python", "solve.py"]