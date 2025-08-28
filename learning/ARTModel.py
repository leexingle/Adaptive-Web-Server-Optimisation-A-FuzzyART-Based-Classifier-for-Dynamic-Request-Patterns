import numpy as np
from sklearn.preprocessing import OneHotEncoder

class FuzzyART:
    def __init__(self, input_size, vigilance=0.8, learning_rate=1.0, choice=0.0001):
        self.input_size = input_size
        self.vigilance = vigilance
        self.learning_rate = learning_rate
        self.choice = choice
        self.categories = []

    # def _normalize_input(self, input_vec):
    #     return np.clip(input_vec, 0, 1)

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
                choice_value = self._compute_choice(input_vec, category)
                self.categories[i] = (
                    self.learning_rate * self._fuzzy_and(input_vec, category) + 
                    (1 - self.learning_rate) * category
                )
                return i  # Return index of the matching category
        
        # No match, create a new category
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

# Sample data processing
type = [
    "image", "image", "text", "text", "text", "text", "text", 
    "image", "image", "text", "image", "text", "text", "text", 
    "image", "image", "text", "text","image", "image", "text", 
    "image","text", "text", "text"
]

contentLength = [
    1671167, 1671167, 11970, 11970, 1944, 247760, 247760, 
    66971, 66971, 3869, 4001, 71, 6897, 156983, 
    1671167, 1671167, 82628, 11971, 84760, 84760, 3869, 
    4001, 108, 71, 12779
]

# Preprocess contentLength and types
# combine multiple values into one
processed_contentLength = [np.mean(item) if isinstance(item, tuple) else item for item in contentLength]
# normalise value
# normalized_contentLength = [( x - min(processed_contentLength) / (max(processed_contentLength) - min(processed_contentLength))) for x in processed_contentLength]
normalized_contentLength = [( x  / 500000 ) for x in processed_contentLength]
# complement normalise value
complement_normalized_value = [(1 - x)  for x in normalized_contentLength]
# encode type 
encoder = OneHotEncoder(sparse_output=False)
types_encoded = encoder.fit_transform([[str(t)] for t in type])
# combine input features
# input_features = np.column_stack((normalized_contentLength, types_encoded))
input_features = np.column_stack((normalized_contentLength, complement_normalized_value,types_encoded))
print(input_features)
# Initialize and train Fuzzy ART
fuzzy_art = FuzzyART(input_size=input_features.shape[1], vigilance=0.9, learning_rate=0.1)

# Train Fuzzy ART on the processed features
for input_vec in input_features:
    category = fuzzy_art.train(input_vec)
    print(f"Input  assigned to category {category}")

# Predict category for with training data
for input_vec in input_features:
    predicted_category = fuzzy_art.predict(input_vec)
    print(f"Predicted category for input : {predicted_category}")