# Services Directory

Welcome to the Services Directory! This directory contains a Python script (`run_service.py`) that provides an interface to manage and execute different services based on provided command line arguments.

## Overview

The Python script operates on various services, from REST APIs like FDIC or YouTube to proxies for an array of applications. The script offers the flexibility to execute a single service or multiple services all at once based on the provided parameters.

## Prerequisites

Before running the script, ensure that the following dependencies are installed and correctly configured:

- Python 3.x
- tmux
- Docker
- Java 8 and Java 11 (depending on the service)
- mitmproxy
- Jacoco

Please ensure that "python" is aliased to Python 3.x and the paths for Java 8 and Java 11 are correctly specified in `java8.env` and `java11.env`. Refer to `setup.sh` in the root directory to get information how to install these requirements.

## How to Use

The script can be executed from the command line using the following syntax:

```
python run_service.py <service_name> <token>
```

The script accepts two parameters:

- `<service_name>`: This is the name of the service you wish to run. The valid options include "fdic", "genome-nexus", "language-tool", "ocvn", "ohsome", "omdb", "rest-countries", "spotify", "youtube", and "all".
- `<token>`: This is a token required for certain services. If the service does not require an authorization token, please use "no_token" as the value.

For instance, to execute the "language-tool" service, the command would be:

```
python run_service.py language-tool no_token
```

## Supported Services

The script supports the following services:

### JDK 8_1 Services (Requires Build)
These services must be built before running. Use `bash services/build_jdk8_1_services.sh` to build them:

- **Features Service**: A REST API for managing product features and configurations. Runs on port 30100.
  - Command: `python run_service_mac.py features-service no_token`
  - Requires: Java 8, Maven
  
- **NCS (Numerical Constraint Service)**: An artificial REST service for constraint testing. Runs on port 30200.
  - Command: `python run_service_mac.py ncs no_token`
  - Requires: Java 8, Maven
  
- **SCS (String Constraint Service)**: An artificial REST service for string operations. Runs on port 30300.
  - Command: `python run_service_mac.py scs no_token`
  - Requires: Java 8, Maven

### Other Services

- FDIC: Runs a reverse proxy for FDIC.
- Genome Nexus: Runs a Genome Nexus server and its reverse proxy.
- Language Tool: Runs the Language Tool server and its reverse proxy.
- OCVN: Runs the OCVN server and its reverse proxy.
- Ohsome: Runs a reverse proxy for Ohsome.
- OMDB: Runs a reverse proxy for OMDB. Obtain a OMDB API key by visiting https://www.omdbapi.com/apikey.aspx. After obtaining the key, replace `YOUR_TOKEN_HERE` with your API key in the `omdb.py` file.
- Rest Countries: Runs a reverse proxy for Rest Countries.
- Spotify: Runs a reverse proxy for Spotify. Obtain a Spotify API key by visiting https://developer.spotify.com/console/get-playlists/ and clicking "Get Token". You'll need to provide this token when executing the service running script.
- YouTube: Runs the YouTube server and its reverse proxy.
- All: Runs all of the above services.

For each service, the script runs the necessary setup and startup commands, including starting servers, running proxies, replacing tokens in necessary files, and more.

## Building JDK 8_1 Services

Before running features-service, ncs, or scs, you must build them first:

```bash
# Make the build script executable
chmod +x services/build_jdk8_1_services.sh

# Run the build script
bash services/build_jdk8_1_services.sh
```

This will:
1. Navigate to the aratrl-service/jdk8_1 directory
2. Load the Java 8 environment
3. Run `mvn clean install -DskipTests` to build all services
4. Generate the necessary JAR files and classpaths

The build process may take several minutes to complete.

## Troubleshooting

If a service fails to start, you can check the Tmux session for that service to see any error messages. First, list the currently running Tmux sessions with:

```
tmux ls
```

Then, attach to a Tmux session with the following command:

```
tmux attach -t session-name
```

Replace `session-name` with the name of the Tmux session for the service (e.g., `fdic-proxy`). To detach from a Tmux session, press `Ctrl + b` followed by `d`.
