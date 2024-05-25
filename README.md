# AutoRestTest

### Introduction
***AutoRestTest*** is a complete testing software for automated API testing that combines the utility of graph
theory, Large Language Models (LLMs), and multi-agent reinforcement learning (MARL) to parse the OpenAPI Specification
and create enhanced comprehensive test cases.

The program uses OpenAI's cutting-edge LLMs for natural-language processing during the creating of the reinforcement 
learning tables and graph edges. 

> [!Important]
> An OpenAI API key is required to use the software. The key can be obtained 
> from [OpenAI's API website](https://openai.com/index/openai-api/). The cost for software per execution is linearly 
> dependent on the number of available operations with the API. For reference, when testing the average API with
> ~15 operations, the cost was approximately $0.4. 

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

There is a wide array of configuration options available within the codebase. 

#### 1. Specifying the API Directory

If the user intends to use their own OpenAPI Specification files as described in the *Installation* section, 
they can change the **spec_dir** variable within the `AutoRestTest.py` file to the name of the directory containing the
added specifications.

#### 2. Configuring the Reinforcement Learning Parameters

The user can configure the reinforcement learning parameters, such as the learning rate (**alpha**), the discount factor (**gamma**), 
and the epsilon-greedy exploration value (**epsilon**). These parameters are assigned when instantiating the `q_learning` 
object in the `AutoRestTest.py` file. As default, the parameters are set to the following values:
- Learning Rate: 0.1
- Discount Factor: 0.9
- Epsilon: 0.3

Instead of limiting the amount of episodes, the program limits the amount of reinforcement learning iterations using a 
time-based approach. The default value is set to 10 minutes. This can be altered by changing the **time_limit** variable
when instantiating the `q_learning` object. The units are in seconds.

> [!TIP]
> The above steps alter the variables across all four agents used within the software. If the user desires to change
> the individual agent parameters, they can navigate to the `src/reinforcement/agents.py` file and change the parameters.

#### 3. Configuring the OpenAI Model

If the user wants to reduce the cost of executing the software, they can change the OpenAI models used throughout 
the program. The model is assigned when instantiating the `SmartValueGenerator` class throughout the codebase.
The user can specify the **engine** variable to an appropriate model. By default, a combination of the **gpt-4o** and 
**gpt-3.5-turbo-0125**  engines are used throughout, with their application tactically chosen based on the context.

> [!WARNING]
> The software heavily uses the **json_mode** of recent OpenAI API engines. If the user desires to change the model,
> it is important to ensure that the model supports the mode.

#### 4. Use of Cached Graphs and Reinforcement Learning Tables

The software uses a caching mechanism to store the graph edges and reinforcement learning tables for each OpenAPI 
Specification after generation. This is done to reduce the cost of execution and to speed up the process on repeated
trials. The user can determine whether they want to use the cached graphs and tables by changing the **use_cache_graph**
and **use_cache_table** variables in the `AutoRestTest.py` file. By default, both variables are set to **True**.

> [!IMPORTANT]
> When the **Header Agent** initializes its learned values table, it attempts to use any registration operations available
> to then construct valid token headers. If the user attempts to use a cached table on a service running locally, the 
> account registration may have reset, in which case the reforcement learning tables should be repeated. The graph construction
> is not affected by this.

## Execution

The software can be executed by running the `AutoRestTest.py` file from the root directory using the following command:
```
python3 AutoRestTest.py <scope> <local> -s <service>
```
We define the arguments as follows:

***scope***: The scope of the testing. The user can choose between **one** or **all**. The **one** scope tests a single
  OpenAPI Specification file as dictated by the "service" argument, while the **all** scope tests all the OpenAPI 
Specification files in the directory specified in the *Installation* step.

***local***: A boolean value that determines whether the service being tested is local or not. If the value is **true**, 
   the software will query the URL specified in the OpenAPI Specification file. If the value is **false**, the
software will use a specified URL for the service (used for reverse-proxy testing). The program used for determining the
URL is in the `AutoRestTest.py` file in the `get_api_url()` function.
For most use cases, assign **local** to **true**.

***service***: The name of the OpenAPI Specification file to test. This argument is only used when the **scope** 
argument is set to **one**. The user must specify the name of the file without the `.yaml` or `.json` extension. 
For example, if the user wants to test the OpenAPI Specification file `genome-nexus.yaml`, the argument would be 
`-s genome-nexus`.

## Results

When instantiating the graph and table, the software will print updates on which step it is on. 
When starting the reinforcement learning step, where the API is thoroughly tested, the software will consistently 
print the occurrences of each operation status code. After the reinforcement learning step is complete, the software 
will print the reinforcement learning tables updated with the learned values, and the cost of the complete execution
(the cost of the LLMs in USD).



