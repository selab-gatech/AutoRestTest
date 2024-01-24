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

Please ensure that "python" is aliased to Python 3.x and the paths for Java 8 and Java 11 are correctly specified in `java8.env` and `java11.env`.

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
