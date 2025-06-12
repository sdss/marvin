

# Marvin Docker

Describes the setup of a containerized system hosting the Marvin
web application.


## Initial Setup

There are two bind mounts:

- The SDSS SAS filesystem
- The host machine pgpass config file

1. Set a `SAS_BASE_DIR` environment variable that points to the
top level directory of the SDSS SAS filesystem.

2. Create or check that a pgpass config file exists at `$HOME/.pgpass`, which contains the manga database connection string info `host.docker.internal:5432:manga:marvin:(password)`.  Replace the `(password)` with the local database password.

The marvin docker setup attempts to connect to the local host machine postgres database directly using `host.docker.internal`


## Run docker compose

All commands are relative to within the docker folder.  From the top-level repo, run `cd docker`

To build and run the docker compose system:
```bash
docker compose up
```
To force a build, you can do: `docker compose up --build`

Navigate to `http://localhost:8080/marvin/`.

To bring the system down:
```bash
docker compose down
```

The docker compose system starts three services:
- Marvin Webapp - the backend web application, mounted to port 8000
- Nginx - the nginx web service, mounted to port 8080
- Redis - a redis database for caching

The final web application is available at `localhost:8080/marvin`

## Build and Tag Version

By default the marvin docker build will checkout the latest `main` branch of marvin on github and tag the image as `marvin:latest`.  To
change this to a specific branch or tag of marvin, e.g. `2.8.0` or `branch_name`, set the `MARVIN_TAG` environment variable.  Then
rebuild the system with `docker compose up --build`.  This will checkout that branch or tag of the marvin repository and create
a new marvin docker image with that tag name.

## Developer Notes

To run the local developer docker, run the following command from the top level marvin project directory:
```bash
docker compose -f docker/docker-compose-dev.yml up --build
```

If you do not have a local postgres manga database, you can create a tunnel to the manga db at JHU.