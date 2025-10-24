# Stage 1: builder - install dependencies and build wheels (keeps final image small)
FROM python:3.13-slim AS builder

# install build deps for Pillow, pandas, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libffi-dev \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage docker layer cache
COPY requirements.txt /app/

# If you use pip requirements:
RUN python -m pip install --upgrade pip setuptools wheel
# Install into an isolated wheelhouse to speed up final stage
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# Stage 2: runtime image
FROM python:3.13-slim

# runtime OS deps needed by Pillow / pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    libpq5 \
    locales \
  && rm -rf /var/lib/apt/lists/*

# create non-root user
ENV APP_USER=nexo
RUN useradd --create-home --shell /bin/bash $APP_USER

WORKDIR /app

# copy wheels from builder and install
COPY --from=builder /wheels /wheels
# copy project
COPY . /app

# install pip from wheels (faster + reproducible)
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --find-links=/wheels -r requirements.txt

# Ensure .env is not copied into images in production: .env is in .dockerignore by default.
# Create a directory for logs/static if needed
RUN mkdir -p /app/logs /app/static && chown -R $APP_USER:$APP_USER /app

# Switch to non-root user
USER $APP_USER

# Expose port used by Uvicorn/gunicorn
ENV PORT=8000
EXPOSE 8000

# default command — run uvicorn if FastAPI app exists at backend.main:app
# change to your actual entrypoint module path if different
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --loop auto --workers 1"]