import numpy as np
import time
import requests
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from math import log2
from mabwiser.mab import MAB, LearningPolicy


class FuzzyART:
    def __init__(self, input_size, vigilance=0.8, learning_rate=1.0, choice=0.0001):
        self.input_size = input_size
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.choice = choice
        self.categories = []

    def _fuzzy_and(self, a, b):
        return np.minimum(a, b)

    def _compute_match(self, input_vec, category):
        return np.sum(self._fuzzy_and(input_vec, category)) / np.sum(input_vec)

    def _compute_choice(self, input_vec, category):
        Tj = np.sum(self._fuzzy_and(input_vec, category)) / (self.choice + np.sum(category))
        return Tj

    def train(self, input_vec):
        for i, category in enumerate(self.categories):
            match = self._compute_match(input_vec, category)
            if match >= self.vigilance:
                self.categories[i] = (
                    self.learning_rate * self._fuzzy_and(input_vec, category) +
                    (1 - self.learning_rate) * category
                )
                return i
        self.categories.append(input_vec.copy())
        return len(self.categories) - 1

    def predict(self, input_vec):
        best_choice = -1
        best_value = -np.inf
        for i, category in enumerate(self.categories):
            match = self._compute_match(input_vec, category)
            if match >= self.vigilance:
                choice_value = self._compute_choice(input_vec, category)
                if choice_value > best_value:
                    best_choice = i
                    best_value = choice_value
        return best_choice

# Normalization range
normalisation_low = 0.0
normalisation_high = 100.0

def scale_cost(cost, low_cost, high_cost):
    """
    Scales the cost to a 0.0 - 1.0 range, with values outside the range
    truncated to the boundaries. Also shifts negative low_cost to zero.

    Args:
        cost (float): The cost to be scaled.
        low_cost (float): The lower bound of the cost range.
        high_cost (float): The upper bound of the cost range.

    Returns:
        float: The scaled cost in the range [0.0, 1.0].
    """
    # Truncate cost at high/low boundaries
    if cost > high_cost:
        cost = high_cost
    elif cost < low_cost:
        cost = low_cost

    # Shift negative low_cost to zero
    if low_cost < 0.0:
        mod = -low_cost  # Equivalent to low_cost / -1.0
        cost += mod
        low_cost += mod
        high_cost += mod

    # Convert the range into 0.0 -> 1.0
    scaled_cost = cost / high_cost
    return scaled_cost

def cost_to_reward(cost):
    """
    Converts cost to reward by inverting it.

    Args:
        cost (float): The cost value.

    Returns:
        float: The corresponding reward value.
    """
    return 1.0 - cost

# Fetch PerceptionData from REST API
def fetch_perception_data():
    try:
        response = requests.get(API_PERCEPTION)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching perception data: {e}")
        return None

# Fetch current config from REST API
def fetch_config():
    try:
        configs_response = requests.get(API_GET_CONFIG)
       
        if configs_response.status_code == 200:
            configs_data = configs_response.json()  
            config_string = configs_data["config"]
                 
        return config_string
    
    except requests.RequestException as e:
        print(f"Error fetching configs: {e}")
        return None
    
# Fetch all config from REST API
def fetch_all_configs():
    try:
        configs_response = requests.get(API_GET_ALL_CONFIGS)
       
        if configs_response.status_code == 200:
            configs_data = configs_response.json()  
            configs_string = configs_data["configs"]
                 
        return configs_string
    
    except requests.RequestException as e:
        print(f"Error fetching configs: {e}")
        return None
    
# set current config to REST API
def set_config(current_config):
    try:
        # Define the payload
        payload = {"config": current_config}
        
        # Send the POST request
        response = requests.post(API_SET_CONFIG, json=payload, headers={"Content-Type": "application/json"})
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Configuration set successfully.")
            return True
        else:
            print(f"Failed to set configuration. Status code: {response.status_code}, Response: {response.text}")
            return False
    
    except requests.RequestException as e:
        print(f"Error setting config: {e}")
        return False
       
    
# Constants
DEFAULT_SERVER_IP = "127.0.0.1"
DEFAULT_SERVER_PORT = 8008
API_PERCEPTION = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/get_perception"
# Fetch configurations from the API
API_GET_CONFIG = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/get_config"
API_GET_ALL_CONFIGS = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/get_all_configs"
API_SET_CONFIG = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/set_config"

# config = fetch_config()
# print(config)
configs = fetch_all_configs()


# Initialize Fuzzy ART and OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, categories=[['image', 'text']])
fuzzy_art = FuzzyART(input_size=0, vigilance=0.9, learning_rate=1.0)

exploredCategories = []
rl_list = {}

# Initialize first reinforcement learning instance
rl = MAB(arms=configs, learning_policy=LearningPolicy.UCB1())
rewards = [0.5] * len(configs)  # Example: Initialize all arms with a reward of 0.5
rl.fit(configs, rewards)  # Fit the model with the reward data



# Process data and fit into Fuzzy ART
max_iterations = 100
iteration = 0
while iteration < max_iterations :

    reward = 0.0

    # Get the action index from the learning instance
    current_config = rl.predict()

    # Print the chosen action index and configuration (if applicable)
    # print(f"Choosing action {ndx}")  # Optionally include: {configs[ndx].string}

    # Set the configuration in the RestAPI
    set_config(current_config)


    time.sleep(2)
    # get data from RESTAPI
    perception_data = fetch_perception_data()

    
    if not perception_data or 'events' not in perception_data:
        print("No valid perception data received.")
        continue

    # get response time
    if perception_data and 'metrics' in perception_data:
        metrics = perception_data['metrics']  # Access the metrics array
    
        if len(metrics) > 0:  # Check if metrics is not empty
            first_metric = metrics[0]  # Get the first metric
            vk = first_metric.get('value', 0)  # Get the value
            vt = first_metric.get('count', 0)  # Get the count

            avg = vk/vt
            reward = avg

    # types = []
    # content_length = []
    image = 0
    text = 0
    content_length_image = 0
    content_length_text = 0

    # extract type and length from json
    for event in perception_data['events']:
        event_name = event['name']
        
        if not event_name:  # Skip unnamed events
            continue

        avg_content_length = event['value'] / event['count']
      
        print(f"{event_name}, {avg_content_length}")

        # Check if event_name is "image/text" and the valid content length 
        if (event_name == "image" or event_name == "text") and avg_content_length != 1.0:
         
            # types.append(event_name)
            # content_lengths.append(avg_content_length)
            if (event_name == "image"):
                image = 1
                content_length_image = avg_content_length
            

            if (event_name == "text"):
                text = 1
                content_length_text = avg_content_length
          

        else:
            # print("No valid events to process.")
            continue

    # get resources from json
    resources = []

    for trace in perception_data['trace']:
        resource = trace['content']
        
        if not resource:  # Skip unnamed events
            continue

        # print(resource)

        resources.append(resource)

    if not resources:
        print("No valid trace to process.")
        continue

    # Normalize the reward (currently "cost", since lower is better)
    normalised = scale_cost(reward, normalisation_low, normalisation_high)

    # Convert "cost" into reward, where higher is better (ML algorithms assume reward as input)
    final_reward = cost_to_reward(normalised)

    # Print the calculated reward details
    # print(" - Calculated reward:")
    # print(f"    - Original: {reward}")
    # print(f"    - Normalised: {normalised}")
    # print(f"    - Reward: {final_reward}")

    # Send reward data for this action to the learning algorithm
    # rl.learn([ndx], [final_reward])  # Update MAB with chosen action and reward

    # Data Processing
    # Normalize content lengths
    # normalized_content_lengths = [( x  / 500000 ) for x in content_lengths]
    # complement normalise value

    if (image == 1):
        normalized_content_length_image = ( content_length_image  / 500000 )
        complement_normalized_content_length_image = (1 - normalized_content_length_image)
    else:
        normalized_content_length_image = 0
        complement_normalized_content_length_image = 0

    if (text == 1):
        normalized_content_length_text = ( content_length_text  / 500000 )
        complement_normalized_content_length_text = (1 - normalized_content_length_text)
    else:
        normalized_content_length_text = 0
        complement_normalized_content_length_text = 0

    


    # Encode MIME types
    # types_encoded = encoder.fit_transform([[t] for t in types])

    # calculate entropy
    # Count frequencies of each unique filename
    file_counts = Counter(resources)
    total_files = len(resources)
    # Calculate probabilities and entropy
    probabilities = [count / total_files for count in file_counts.values()]
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    

    # Combine features
    if fuzzy_art.input_size == 0:  # Initialize input size on first iteration
        fuzzy_art.input_size = 5
    # modify size of entropy
    # entropy_column = [abs(entropy)] * len(normalized_content_lengths)  # Duplicate entropy for each input vector
    # concatenate all features as a list of input_features
    input_features = np.column_stack((normalized_content_length_image, complement_normalized_content_length_image, normalized_content_length_text, complement_normalized_content_length_text, abs(entropy)))

    category = -1
    # Train Fuzzy ART model
   
    input_features = [input_features]
    while input_features:
    
        current_input_feature = input_features.pop() 
        category = fuzzy_art.train(current_input_feature)
        print(current_input_feature)
        print(f"Input assigned to category {category}")

    # Check if the category is not in the explored categories and add it if necessary
    if category not in exploredCategories:
        exploredCategories.append(category)
        # create new rl for diff class
        rl = MAB(arms=configs, learning_policy=LearningPolicy.UCB1())
        rewards = [0.5] * len(configs)  # Example: Initialize all arms with a reward of 0.5
        rl.fit(configs, rewards)  # Fit the model with the reward data
        rl_list[category] = rl
        print("new rl created")
 

    # set next rl from previous rl
    else:
        rl = rl_list[category]
        
    print(rl)
    iteration += 1
       

# Predict category for current inputs
groundtruth = np.array([
    [3.342334e+00, -2.342334e+00, 1.000000e+00, 0.000000e+00],
    [3.342334e+00, -2.342334e+00, 1.000000e+00, 0.000000e+00],
    [2.394000e-02, 9.760600e-01, 0.000000e+00, 1.000000e+00],
    [2.394000e-02, 9.760600e-01, 0.000000e+00, 1.000000e+00],
    [3.888000e-03, 9.961120e-01, 0.000000e+00, 1.000000e+00],
    [4.955200e-01, 5.044800e-01, 0.000000e+00, 1.000000e+00],
    [4.955200e-01, 5.044800e-01, 0.000000e+00, 1.000000e+00],
    [1.339420e-01, 8.660580e-01, 1.000000e+00, 0.000000e+00],
    [1.339420e-01, 8.660580e-01, 1.000000e+00, 0.000000e+00],
    [7.738000e-03, 9.922620e-01, 0.000000e+00, 1.000000e+00],
    [8.002000e-03, 9.919980e-01, 1.000000e+00, 0.000000e+00],
    [1.420000e-04, 9.998580e-01, 0.000000e+00, 1.000000e+00],
    [1.379400e-02, 9.862060e-01, 0.000000e+00, 1.000000e+00],
    [3.139660e-01, 6.860340e-01, 0.000000e+00, 1.000000e+00],
    [3.342334e+00, -2.342334e+00, 1.000000e+00, 0.000000e+00],
    [3.342334e+00, -2.342334e+00, 1.000000e+00, 0.000000e+00],
    [1.652560e-01, 8.347440e-01, 0.000000e+00, 1.000000e+00],
    [2.394200e-02, 9.760580e-01, 0.000000e+00, 1.000000e+00],
    [1.695200e-01, 8.304800e-01, 1.000000e+00, 0.000000e+00],
    [1.695200e-01, 8.304800e-01, 1.000000e+00, 0.000000e+00],
    [7.738000e-03, 9.922620e-01, 0.000000e+00, 1.000000e+00],
    [8.002000e-03, 9.919980e-01, 1.000000e+00, 0.000000e+00],
    [2.160000e-04, 9.997840e-01, 0.000000e+00, 1.000000e+00],
    [1.420000e-04, 9.998580e-01, 0.000000e+00, 1.000000e+00],
    [2.555800e-02, 9.744420e-01, 0.000000e+00, 1.000000e+00]
])

for input_vec in groundtruth:
    predicted_category = fuzzy_art.predict(input_vec)
    print(f"Predicted category for input: {predicted_category}")
