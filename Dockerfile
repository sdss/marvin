# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository and checkout the specific tag
# RUN git clone https://github.com/sdss/marvin.git /app && \
#     git checkout tags/2.8.0
COPY . /app

#RUN pip install "psycopg2<2.9"
#RUN pip install psycopg2-binary

# Install deps for psycopg2
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install psycopg2


# Install any needed packages
RUN pip install --no-cache-dir ".[web,db]"

# Install uWSGI
RUN pip install gunicorn uvicorn
RUN pip install "jinja2<3.1"
RUN pip install "packaging<21"
RUN pip install "itsdangerous==2.0.1"
RUN pip install "pillow<10"

# Create the log directory and ensure it's writable
RUN mkdir -p /tmp/marvin/logs && \
    chmod -R 755 /tmp/marvin

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Copy your uWSGI ini file into the container
#COPY python/marvin/web/uwsgi_conf_files/docker.py /app/python/marvin/web/uwsgi_conf_files/docker.py

RUN mkdir -p /root/.marvin && chmod -R 755 /root/.marvin
RUN echo "use_sentry: False\nadd_github_message: False\ncheck_access: False" > /root/.marvin/marvin.yml

ENV FLASK_ENV=docker
ENV FLASK_APP="marvin.web.uwsgi_conf_files.app:app"

WORKDIR /app/python

# Run the application with uWSGI
#CMD ["uwsgi", "--ini", "/app/python/marvin/web/uwsgi_conf_files/docker.ini"]
#CMD ["gunicorn", "-w", "1", "-b", ":8000", "marvin.web.uwsgi_conf_files.app:app"]
CMD ["gunicorn", "-c", "/app/python/marvin/web/uwsgi_conf_files/docker.py", "marvin.web.uwsgi_conf_files.app:app"]