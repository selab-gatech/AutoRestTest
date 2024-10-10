# AutoRestTest

### Introduction
***AutoRestTest*** is a complete testing software for automated API testing that combines the utility of graph
theory, Large Language Models (LLMs), and multi-agent reinforcement learning (MARL) to parse the OpenAPI Specification
and create enhanced comprehensive test cases.

Watch this [demonstration video](https://www.youtube.com/watch?v=VVus2W8rap8) of AutoRestTest to learn how it solves complex challenges in automated REST API testing, as well as its configuration, execution steps, and output.

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

The following are the steps to install the *AutoRestTest* software:
1. Clone the Github repository.
2. Install the required dependencies:
   - Either use `requirements.txt` to install the dependencies or the Conda environment file `auto-rest-test.yml`
3. Create a `.env` file in the root directory and add the following environmental variable:
   - `OPENAI_API_KEY = '<YOUR_API_KEY>'` 

> [!TIP]
> For compatability across different operating systems, it is recommended to use the Conda environment file

Optionally, if the user wants to test specific APIs, they can install their OpenAPI Specification files within a folder
in the root directory. For convenience, we have provided a large array of OpenAPI Specifications for popular and 
widely-used APIs. These Specification scan be seen in the `aratrl-openapi` and `specs` directories.

At this point, the installation step is complete and the software can be executed. However, it is important that the
following configuration steps are completed for purposeful execution.

## Configuration

There is a wide array of configuration options available within the codebase. For convenience, the configuration options
are easily accessible within the `configurations.py` file in the root directory. The file contains information
regarding each of the parameters that can be altered. The following are additional instructions for each of the configuration choices.
Note that the parameteres referenced in the following steps are found in the `configurations.py` file.

#### 1. Specifying the API Directory

If the user intends to use their own OpenAPI Specification files as described in the *Installation* section, 
they can change the **SPECIFICATION_DIRECTORY** variable within the `configurations.py` file to the name of the directory containing the
added specifications.

#### 2. Configuring the Reinforcement Learning Parameters

The user can configure the reinforcement learning parameters, such as the learning rate (**LEARNING_RATE**), the discount factor (**DISCOUNT_FACTOR**), 
and the epsilon-greedy exploration value (**EXPLORATION_RATE**) used in Q-learning. As default, the parameters are set to the following values:
- Learning Rate: 0.1
- Discount Factor: 0.9
- Epsilon: 0.3

Instead of limiting the amount of episodes, the program limits the amount of reinforcement learning iterations using a 
time-based approach. The default value is set to 10 minutes. This can be altered by changing the **TIME_DURATION** variable in the configurations file. The units are in seconds.

> [!TIP]
> The above steps alter the variables across all four agents used within the software. If the user desires to change
> the individual agent parameters, they can navigate to the `src/reinforcement/agents.py` file and change the parameters.

#### 3. Configuring the OpenAI Model

If the user wants to reduce the cost of executing the software, they can change the OpenAI models used throughout 
the program. The user can use the **OPENAI_LLM_ENGINE** variable to specify an appropriate model. By default, the **gpt-4o**
model is selected given its high performance, cost-effectiveness, and high token limit.

> [!WARNING]
> The software heavily uses the **JSON mode** from recent OpenAI API engines. All GPT 3.5-Turbo, 4-Turbo, and 4o engines support the JSON mode. 
> Additionally, the cost of execution is only provided for the most recent versions of the listed OpenAI engines.

#### 4. Use of Cached Graphs and Reinforcement Learning Tables

The software uses a caching mechanism to store the graph edges and reinforcement learning tables for each OpenAPI 
Specifications after generation. This is done to reduce the cost of execution and to speed up the process on repeated
trials. The user can determine whether they want to use the cached graphs and tables by changing the **USE_CACHED_GRAPH**
and **USE_CACHED_TABLE** variables. By default, both variables are set to **False**.

### 5. Optional Header Agent

AutoRestTest contains an optional Header agent responsible for testing the API with different authentication headers. Due to difficulties 
of different authentication flows, the Header agent is only able to use Basic Authentication. By default, the agent is disabled.

> [!CAUTION]
> The Header agent Q-table should be rerun when executing services with local databases that refresh, as the user
> credentials may become invalid.

## Execution

The software can be executed by running the `AutoRestTest.py` file from the root directory using the following command:
```
python3 AutoRestTest.py
```
To indicate the specification for execution, change the **SPECIFICATION_LOCATION** variable in the `configurations.py` file.

### Docker Execution

For ease of use, the software can be executed using Docker. The user can apply the following commands from the 
root directory sequentially to execute AutoRestTest using the pre-built Docker image:

```
docker build -t autoresttest .
docker run -it autoresttest
```

Ensure that the `configurations.py` file is configured correctly before executing the software.

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




