"""
Structured Machine Learning System
This module combines FuzzyART classification with reinforcement learning
to optimize system configuration based on perception data.
"""
import numpy as np
import time
import requests
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from math import log2
from mabwiser.mab import MAB, LearningPolicy
import pandas as pd
import psutil

class Config:
    """System configuration parameters and constants."""
    
    # Server Configuration
    DEFAULT_SERVER_IP = "127.0.0.1"
    DEFAULT_SERVER_PORT = 8008
    
    # API Endpoints
    API_GET_PERCEPTION = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/get_perception"
    API_GET_CONFIG = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/get_config"
    API_GET_ALL_CONFIGS = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/get_all_configs"
    API_SET_CONFIG = f"http://{DEFAULT_SERVER_IP}:{DEFAULT_SERVER_PORT}/meta/set_config"
    
    # ML Configuration
    NORMALIZATION_LOW = 0.0
    NORMALIZATION_HIGH = 100.0
    MAX_ITERATIONS = 1500

    BOOT_TIME = psutil.boot_time()

class APIClient:
    """Handles all API interactions."""
    
    def __init__(self, config):
        self.config = config

    def fetch_perception_data(self):
        """Fetch perception data from API."""
        try:
            response = requests.get(self.config.API_GET_PERCEPTION)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching perception data: {e}")
            return None

    def fetch_config(self):
        """Fetch current configuration from API."""
        try:
            response = requests.get(self.config.API_GET_CONFIG)
            if response.status_code == 200:
                return response.json()["config"]
            return None
        except requests.RequestException as e:
            print(f"Error fetching config: {e}")
            return None

    def fetch_all_configs(self):
        """Fetch all available configurations from API."""
        try:
            response = requests.get(self.config.API_GET_ALL_CONFIGS)
            if response.status_code == 200:
                return response.json()["configs"]
            return None
        except requests.RequestException as e:
            print(f"Error fetching all configs: {e}")
            return None

    def set_config(self, current_config):
        """Set current configuration via API."""
        try:
            payload = {"config": current_config}
            response = requests.post(
                self.config.API_SET_CONFIG,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # print("Configuration set successfully.")
                return True
                
            print(f"Failed to set configuration. Status: {response.status_code}")
            return False
            
        except requests.RequestException as e:
            print(f"Error setting config: {e}")
            return False


class FileHandler:
    def __init__(self):
        # Create a new file with the current timestamp as the filename
        self.filename = self._generate_filename()
        # self._create_file()

    def _generate_filename(self):
        # num = 0
        filename = f"38.csv"

        # while os.path.exists(filename):
        #     num += 1
        #     filename = f"Agent{num}.csv"

        return filename
    
    def _create_file(self):
        """Create a new file with the generated filename."""
        with open(self.filename, 'w') as file:
            # Write the header row
            file.write("Time,Features,Reward,Category,Current Config Index\n")

    def log(self, time, features, reward, response_time, min_norm, max_norm, category, current_config_index):
        """Append log data to the file."""
        with open(self.filename, 'a') as file:
            # Append the data as a comma-separated line
            file.write(f"{time},{features},{reward},{response_time}, {min_norm}, {max_norm}, {category},{current_config_index}\n")


class FuzzyART:
    """Implementation of Fuzzy ART (Adaptive Resonance Theory) neural network."""
    
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
        return np.sum(self._fuzzy_and(input_vec, category)) / (self.choice + np.sum(category))

    def train(self, input_vec):
        """Train the network on a single input vector."""
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
        """Predict category for input vector."""
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


class DataProcessor:
    """Handles data processing and feature extraction."""
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, categories=[['image', 'text']])

    def process_perception_data(self, perception_data):
        """Process raw perception data into features."""
        if not perception_data or 'events' not in perception_data:
            return None

        image, content_length_image, text, content_length_text = self._extract_event_data(perception_data)
        resources = self._extract_resources(perception_data)
        
        if not resources:
            return None

        return self._normalize_features(image, content_length_image, text, content_length_text, resources)

    def _extract_event_data(self, perception_data):
        """Extract event types and content lengths from perception data."""
        image = 0  # 1 means is image
        text = 0
        content_length_image = 0
        content_length_text = 0
        
        for event in perception_data['events']:
            event_name = event['name']
            if not event_name:
                continue

            avg_content_length = event['value'] / event['count']

            if (event_name == "image" or event_name == "text") and avg_content_length != 1.0:
                if (event_name == "image"):
                    image = 1
                    content_length_image = avg_content_length
            
                if (event_name == "text"):
                    text = 1
                    content_length_text = avg_content_length

        return image, content_length_image, text, content_length_text

    def _extract_resources(self, perception_data):
        """Extract resources from perception data."""
        return [trace['content'] for trace in perception_data['trace'] if trace['content']]

    def _normalize_features(self, image, content_length_image, text, content_length_text, resources):
        """Normalize and combine features."""
        # if (image == 1):
        #     normalized_content_length_image = ( content_length_image  / 500000 )
        #     complement_normalized_content_length_image = (1 - normalized_content_length_image)
        # else:
        #     normalized_content_length_image = 0
        #     complement_normalized_content_length_image = 0

        # if (text == 1):
        #     normalized_content_length_text = ( content_length_text  / 500000 )
        #     complement_normalized_content_length_text = (1 - normalized_content_length_text)
        # else:
        #     normalized_content_length_text = 0
        #     complement_normalized_content_length_text = 0
        
        entropy = self._calculate_entropy(resources)

        if entropy == -0.0:
            entropy = 0.0000000001
        
        return np.column_stack((
            content_length_image,
            content_length_text,
            # normalized_content_length_image, 
            # complement_normalized_content_length_image, 
            # normalized_content_length_text, 
            # complement_normalized_content_length_text,
            abs(entropy)
        ))

    def _calculate_entropy(self, resources):
        """Calculate entropy of resource distribution."""
        counts = Counter(resources)
        total = len(resources)
        probabilities = [count / total for count in counts.values()]
        return -sum(p * log2(p) for p in probabilities if p > 0)


class RewardCalculator:
    """Handles reward calculation and normalization."""
    
    def __init__(self, low=0.0, high=100.0):
        self.low = low
        self.high = high
        self.total_response_time = 0

    def calculate_reward(self, metrics, min_norm, max_norm):
        """Calculate reward from metrics."""
        self.low = min_norm
        self.high = max_norm

        if not metrics or len(metrics) == 0:
            return 0.0
            
        first_metric = metrics[0]
        value = first_metric.get('value', 0)
        count = first_metric.get('count', 0)
        
        if count == 0:
            return 0.0
            
        response_time = value / count
        print(response_time)
        self.total_response_time += response_time
        print(f"total response time: {self.total_response_time}")

        normalized = self._scale_cost(response_time)
        return self._cost_to_reward(normalized), response_time

    def _scale_cost(self, cost):
        """Scale cost to normalized range."""
        cost = max(min(cost, self.high), self.low)

        # error handling
        # if self.low < 0.0:
        #     mod = -self.low
        #     cost += mod
        #     adjusted_high = self.high + mod
        #     return cost / adjusted_high
        return cost / self.high

    def _cost_to_reward(self, cost):
        """Convert cost to reward value."""
        reward = 1.0 - cost
        return reward


class RLAgent:
    """Handles all RL-related operations for categories."""

    def __init__(self, config_indices):
        self.config_indices = config_indices
        self.rl_instances = {}

    def initialize_category_rl(self, category):
        """Initialize a new RL instance for a category."""
        rl = MAB(arms=self.config_indices, learning_policy=LearningPolicy.UCB1(alpha=1.0))
        rewards = [0.5] * len(self.config_indices)
        response_times = []
        min_norm = 0
        max_norm = 100
        features = []
        rl.fit(self.config_indices, rewards)
        self.rl_instances[category] = (rl, self.config_indices.copy(), rewards.copy(), response_times, min_norm, max_norm, features)

    def update_rl_instance(self, category, config_index, reward, response_time, feature):
        """Update the RL instance for a category."""
        rl, actions, rewards, response_times, min_norm, max_norm, features = self.rl_instances[category]
        actions.append(config_index)
        response_times.append(response_time)
        rewards.append(reward)

        # Ensure features is a list
        if isinstance(features, np.ndarray):
            features = features.tolist()
        if isinstance(feature, np.ndarray):
            feature = feature.tolist()
            
        features.append(feature)

        # update min and max norm when multiple of 20 input
        if len(response_times) >= 20 and  len(response_times) % 20 == 0:
            # get the most recent 20 response time
            recent_times = response_times[-20:]
          
            if np.percentile(recent_times, 25) > 50:
                min_norm = 0
                max_norm = 1500
            else:
                min_norm = 0  # default value
                max_norm = 100  # default value
         
        # update learning agent status
        rl.partial_fit([config_index], [reward])
        self.rl_instances[category] = (rl, actions, rewards, response_times, min_norm, max_norm, features)

        return rl

    def get_best_config_statistics(self):
        """
        Calculate and store the mean and variance of the rewards for 
        the best configuration for each category.
        
        Returns:
            stats_df (pd.DataFrame): A DataFrame with statistics for each category.
        """
        stats = []
        for category, (rl, actions, rewards, response_times, min_norm, max_norm, features) in self.rl_instances.items():
            
            best_config = rl.predict()  # Get the best config index
            best_rewards = [reward for action, reward in zip(actions, rewards) if action == best_config]

            if best_rewards:
                mean_reward = np.mean(best_rewards)
                variance_reward = np.var(best_rewards)
                count = len(best_rewards)
                total_count = len(rewards)
                percentage = count/total_count
            else:
                mean_reward = variance_reward = 0.0  # Default for categories with no rewards yet
            
            stats.append({
                'Category': category,
                'BestConfigIndex': best_config,
                'MeanReward': mean_reward,
                'VarianceReward': variance_reward,
                'Count': count,
                'TotalCount': total_count,
                'Percentage': percentage
            })

        # Convert stats to a Pandas DataFrame for efficient storage and read operations
        stats_df = pd.DataFrame(stats)
        return stats_df


class System:
    """Main system class that coordinates all components."""
    
    def __init__(self):
        self.config = Config()
        self.api_client = APIClient(self.config)
        self.data_processor = DataProcessor()
        self.reward_calculator = RewardCalculator(
            self.config.NORMALIZATION_LOW,
            self.config.NORMALIZATION_HIGH
        )
        self.fuzzy_art = FuzzyART(input_size=3, vigilance=0.5, learning_rate=0.5)
        self.file_handler = FileHandler()
        self.explored_categories = []
        
        # Initialize configuration and RL agent
        self.configs = self.api_client.fetch_all_configs() # list of configs
        self.config_indices = list(range(len(self.configs)))  # List of indices
        self.rl_agent = RLAgent(self.config_indices)

    def run(self):
        """Main execution loop."""
        iteration = 0

        # initialise first rl
        self.rl_agent.initialize_category_rl(0)
        current_rl, current_actions, current_rewards, response_times, min_norm, max_norm, features = self.rl_agent.rl_instances[0]
        current_config_index = current_rl.predict() # Get the best config of current RL

        while iteration < self.config.MAX_ITERATIONS:

            # Get configuration  
            current_config = self.configs[current_config_index]
    
            # set configuration to REST
            if not self.api_client.set_config(current_config):
                continue
         
            time.sleep(2)  # Wait for web request
            
            # Fetch and process data
            # Record time of fetch prception
            current_time = time.time()
            time_fetch_perception = int((current_time - self.config.BOOT_TIME) * 1000)
            perception_data = self.api_client.fetch_perception_data()
            if not perception_data:
                continue

            # Process features
            features = self.data_processor.process_perception_data(perception_data)
            if features is None:
                continue
          
            # Train FuzzyART and get category
            category = self.train_and_categorize(features)

            # If it's a new category, initialize a new RL instance
            if category not in self.rl_agent.rl_instances:
                self.rl_agent.initialize_category_rl(category)

            # Calculate reward
            min_norm = self.rl_agent.rl_instances[category][4]
            max_norm = self.rl_agent.rl_instances[category][5]
            reward, response_time = self.reward_calculator.calculate_reward(perception_data.get('metrics', []), min_norm, max_norm)
            print(current_config_index, response_time)

            # log stats into file 
            self.file_handler.log(time_fetch_perception, features, reward, response_time, min_norm, max_norm, category, current_config_index)

            # Update actions and rewards for the current RL instance
            current_rl = self.rl_agent.update_rl_instance(category, current_config_index, reward, response_time, features)
            
            # Get the best config of current RL for next prediction
            current_config_index = current_rl.predict()   

            # print statistics of rl agents
            if iteration % 5 == 0:
                stats = self.rl_agent.get_best_config_statistics()
                print(stats)

            iteration += 1

    def train_and_categorize(self, features):
        """Train FuzzyART and get category for features."""
        for feature in features:
            category = self.fuzzy_art.train(feature)
            print(f"Input assigned to category {category}")
            return category
        return -1


def main():
    """Entry point of the application."""
    system = System()
    system.run()

if __name__ == "__main__":
    main()


'''
cd web_server
dana pal.rest WebServer.o

dana client.Cycle
'''