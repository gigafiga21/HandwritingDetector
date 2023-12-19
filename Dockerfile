FROM jhale1805/python-poetry:1.0.10-py3.9-slim

WORKDIR /laboratory-work-5
COPY . .

RUN poetry lock && poetry install --no-root

CMD ["poetry", "run", "python", "source/server.py"]
