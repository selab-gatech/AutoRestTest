# Description: This file contains the configurations for AutoRestTest.
# Change the values for the variables as described in the README.

SPECIFICATIONS_DIRECTORY = 'aratrl-openapi/' # The directory where the OpenAPI specifications are stored
LOCAL_TEST = True # Specifies whether the test is local (true/false). If you are not running the test locally, implement the "get_api_url" function in AutoRestTest.py with your URL configurations.
NAIVE = False # Specifies whether to use the naive approach. Naive approach is no longer supported. Use False.
OPENAI_LLM_ENGINE = "gpt-4o" # The OpenAI language model engine to use for the value agent generation.
# Note: The OpenAI engine must be compatible with the JSON mode. Also, for the cost output to be accurate, the engine must be either "gpt-3.5-turbo-0125" or "gpt-4o".

# The following variables specify which specification to run the program on.
SPECIFICATION_NAME = 'market2' # The name of the specification in the SPECIFICATIONS_DIRECTORY to execute the program on.

# The following variables specify the caching configurations. Ensure that you have ran the program once on the specification before setting these values to true.
USE_CACHED_GRAPH = False # Specifies whether to use the cached Semantic Operation Dependency Graph (true/false).
USE_CACHED_TABLE = False # Specifies whether to use the cached Q-table for the Value Agent (true/false). This will avoid rerunning the LLM value generation.

# The following variables are responsible for the Q-learning agent configurations.
LEARNING_RATE = 0.1 # The learning rate for the Q-learning agent.
DISCOUNT_FACTOR = 0.9 # The discount factor for the Q-learning agent.
EXPLORATION_RATE = 0.7 # The exploration rate specifically for the Operation Agent. The remaining agents use a joint probability distribution.

# The following variables are responsible for the request generation configurations.
TIME_DURATION = 1800 # The time duration for the request generation process.
MUTATION_RATE = 0.2 # The mutation rate for the request generation process.



