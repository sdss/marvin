# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install library files
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libpq-dev gcc build-essential mime-support && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define an argument for the Git tag, defaulting to "main"
ARG MARVIN_TAG=main

# Clone the repository and checkout the specific tag
RUN git clone https://github.com/sdss/marvin.git /app && \
    git checkout ${MARVIN_TAG}

# Install any needed packages
RUN pip install --no-cache-dir ".[web,db]"

# Resolve some dependency issues
RUN pip install gunicorn
RUN pip install "jinja2<3.1"
RUN pip install "packaging<21"
RUN pip install "itsdangerous==2.0.1"
RUN pip install "pillow<10"

# Create the log directory and ensure it's writable
RUN mkdir -p /tmp/marvin/logs && \
    chmod -R 755 /tmp/marvin

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Copy the gunicorn config file into the container
COPY ./gunicorn_config.py /app/python/gunicorn_config.py

# Create the marvin config file
RUN mkdir -p /root/.marvin && chmod -R 755 /root/.marvin
RUN echo "use_sentry: False\nadd_github_message: False\ncheck_access: False" > /root/.marvin/marvin.yml

# Update permissions
RUN chmod -R 755 /app/python/marvin/web

# Set environment variables
ENV FLASK_APP="marvin.web.uwsgi_conf_files.app:app"

# Change to python dir
WORKDIR /app/python

# Run the application with uWSGI
CMD ["gunicorn", "-c", "gunicorn_config.py", "marvin.web.uwsgi_conf_files.app:app"]
