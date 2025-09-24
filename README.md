# AutoRestTest

### Introduction
***AutoRestTest*** is a complete testing software for automated API testing that combines the utility of graph
theory, Large Language Models (LLMs), and multi-agent reinforcement learning (MARL) to parse the OpenAPI Specification
and create enhanced comprehensive test cases.

Watch this [demonstration video](https://www.youtube.com/watch?v=VVus2W8rap8) of AutoRestTest to learn how it solves complex challenges in automated REST API testing, as well as its configuration, execution steps, and output.

> [!NOTE]
> Following the release of the demonstration video, the code base has been refactored.
> Refer to this `README.md` for the most current setup and execution details.

<p align="center">
  <a href="https://www.youtube.com/watch?v=VVus2W8rap8">
    <img src="https://img.youtube.com/vi/VVus2W8rap8/0.jpg" alt="Watch the video">
  </a>
</p>

The program uses OpenAI's cutting-edge LLMs for natural-language processing during the creating of the reinforcement 
learning tables and graph edges. 

> [!Important]
> An OpenAI API key is required to use the software. The key can be obtained 
> from [OpenAI's API website](https://openai.com/index/openai-api/). The cost for software per execution is linearly 
> dependent on the number of available operations with the API. For reference, when testing the average API with
> ~15 operations, the cost was approximately $0.1. 

## Installation

We recommend using Poetry with `pyproject.toml` for dependency management and scripts. A `poetry.lock` file pins exact versions.

Steps:
1. Clone the repository.
2. Ensure Python 3.10.x is available (project targets `>=3.10,<3.11`).
3. Install dependencies with Poetry (uses `poetry.lock` if present):
   - `poetry install`
4. Create a `.env` file in the project root and add:
   - `OPENAI_API_KEY='<YOUR_API_KEY>'`

Alternatives (provided but not recommended):
- `pip install -r requirements.txt`
- `conda env create -f autoresttest.yaml`

Optionally, if the user wants to test specific APIs, they can install their OpenAPI Specification files within a folder
in the root directory. For convenience, we have provided a large array of OpenAPI Specifications for popular and 
widely-used APIs. These Specification scan be seen in the `aratrl-openapi` and `specs` directories.

At this point, the installation step is complete and the software can be executed. However, it is important that the
following configuration steps are completed for purposeful execution.

## Configuration

There is a wide array of configuration options available within the codebase. All configuration options are easily accessible via a single TOML file at the project root: `configurations.toml`.

Below are the relevant settings and where to find them in `configurations.toml`.

#### 1. Specifying the API Specification

If you intend to use your own OpenAPI Specification file as described in the Installation section, set the relative path (from the project root) to that file in `configurations.toml` under `[spec].location`.

Only `.yaml` and `.json` files are supported (OpenAPI 3.0). Example: `aratrl-openapi/market2.yaml`.

#### 2. Configuring Reinforcement Learning Parameters

Configure Q-learning parameters in `configurations.toml`:
- `[q_learning].learning_rate` (default: `0.1`)
- `[q_learning].discount_factor` (default: `0.9`)
- `[q_learning].max_exploration` (epsilon; default: `1`, decays over time to `0.1`)

Instead of limiting episodes, the program limits RL iterations using a time budget. Set the duration (in seconds) under:
- `[request_generation].time_duration` (default: `1200`)

> [!TIP]
> The above steps alter the variables across all four agents used within the software. If the user desires to change
> the individual agent parameters, they can navigate to the `src/autoresttest/agents` files and change the parameters.

#### 3. Configuring the OpenAI Model

To manage cost and performance, adjust the LLM settings in `configurations.toml`:
- `[llm].engine` (default: `gpt-4o-mini`)
- `[llm].temperature` (default: `0.7`)

> [!WARNING]
> The software heavily uses the **JSON mode** from recent OpenAI API engines. All recent models should support the JSON mode. 
> The console output will list token usage for analyzing tool costs.

#### 4. Use of Cached Graphs and Reinforcement Learning Tables

The software can cache the graph and Q-tables to reduce cost and speed up repeated runs. Configure this behavior under:
- `[cache].use_cached_graph` (default: `true`)
- `[cache].use_cached_table` (default: `true`)

> [!NOTE]
> When enabled, these options store and read cached data under the `cache/` directory at the project root (for example, `cache/graphs/` and `cache/q_tables/`).
> If disk usage becomes a concern, you can delete the cached project information in `cache/` after use; the data will be regenerated on the next run when needed.

#### 5. Optional Header Agent

AutoRestTest contains an optional Header agent responsible for testing the API with different authentication headers. Due to difficulties 
of different authentication flows, the Header agent is only able to use Basic Authentication. By default, the agent is disabled.

Toggle via `configurations.toml`:
- `[agents.header].enabled` (default: `false`)

> [!CAUTION]
> The Header agent Q-table should be rerun when executing services with local databases that refresh, as the user
> credentials may become invalid.

## Execution

Run the script using Poetry, after following the installation instructions:
```
poetry run autoresttest
```

To indicate the specification for execution, set `[spec].location` in `configurations.toml`. This path must be relative to the project root.

### Docker Execution

For ease of use, the software can be executed using Docker. The user can apply the following commands from the 
root directory sequentially to execute AutoRestTest using the pre-built Docker image:

```
docker build -t autoresttest .
docker run -it autoresttest
```

Ensure that `configurations.toml` is configured correctly before executing the software.

## Results

Throughout AutoRestTest's execution, its command line interface (CLI) will provide the user with constant updates regarding
the current step of the process. In addition, during request generation, the software will output information related to
the amount of time elapsed, the number of successfully processed (2xx) operations, unique server errors (5xx), the time remaining, 
and the distribution of all status codes. After request generation and reinforcement learning is completed, the software will output
the cost of the program in USD (associated with the cost of using LLMs). 

#### Report Generation and Data Files

AutoRestTest will generate a series of files in the `data` directory within the root folder for easy access to execution results. 
The following are the files generated:
- `report.json`: Contains the compiled report of the API testing process, including key summary statistics and information.
- `operation_status_codes.json`: Contains the distribution of status codes for each operation.
- `q_tables.json`: Contains the completed Q-tables for each agent.
- `server_errors.json`: Contains the unique server errors encountered during the testing process. Only JSON-seriable errors are stored.
- `successful_parameters.json`: Contains the successful parameters for each operation.
- `successful_responses.json`: Contains the successful responses for each operation.
- `successful_bodies.json`: Contains the successful object request bodies for each operation.
- `successful_primitives.json`: Contains the successful primitive request bodies for each operation.

These files contain the necessary information for analysis into the success of AutoRestTest. 

> [!NOTE]
> The output files can grow in size according to the number of operations and the duration of execution. 
> For example, the file containing successful responses can grow to be several **gigabytes** when executing for a long duration. 
> It is recommended to clear the `data` directory when the files are no longer needed.
