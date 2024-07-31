# Description: This file contains the configurations for AutoRestTest.
# Change the values for the variables as described in the README.

SPECIFICATION_LOCATION = "aratrl-openapi/project.yaml" # The location of the Specification file relative to the root directory.
# Note: Only .yaml and .json files are supported. The Specification file must be in the OpenAPI 3.0 format.

OPENAI_LLM_ENGINE = "gpt-4o-mini" # The OpenAI language model engine to use for the value agent generation.
DEFAULT_TEMPERATURE = 0.7 # The default temperature for the OpenAI language model.
# Note: The OpenAI engine must be compatible with the JSON mode. Also, for the cost output to be accurate, the engine must be either "gpt-3.5-turbo-0125" or "gpt-4o".

ENABLE_HEADER_AGENT = False # Specifies whether to enable the Header Agent (True/False).
# Note: The header agent uses Basic tokens and is only beneficial for certain APIs that require such authorization.

# The following variables specify the caching configurations. Ensure that you have ran the program once on the specification before setting these values to true.
USE_CACHED_GRAPH = True # Specifies whether to use the cached Semantic Operation Dependency Graph (true/false).
USE_CACHED_TABLE = True # Specifies whether to use the cached Q-table for the Value Agent and optional Header Agent (true/false). This will avoid rerunning the LLM.
# Note: Assign the caching to False if you have made changes to the graph construction or table generation for the changes to take effect.

# The following variables are responsible for the Q-learning agent configurations.
LEARNING_RATE = 0.1 # The learning rate for the Q-learning agent.
DISCOUNT_FACTOR = 0.9 # The discount factor for the Q-learning agent.
EXPLORATION_RATE = 0.1 # The exploration rate using the epsilon-greedy policy.

# The following variables are responsible for the request generation configurations.
TIME_DURATION = 1200 # The time duration for the request generation process.
MUTATION_RATE = 0.2 # The mutation rate for the request generation process.



