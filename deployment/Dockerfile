FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 surveilx
COPY pyproject.toml ./
COPY surveilx/ ./surveilx/
COPY training/ ./training/
RUN pip install --no-cache-dir .
COPY --from=frontend /frontend/dist ./frontend/dist
RUN mkdir data && chown -R surveilx:surveilx /app
USER surveilx
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "surveilx.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
