import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load your selected_df DataFrame (replace with actual data)
selected_df = pd.read_csv('selected_players.csv')

# Load the trained model
loaded_model = tf.keras.models.load_model('basketball_player_rating_model.h5')

# Define features (player characteristics) for team optimization
features = selected_df[['gp', 'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']].values

# Make predictions using the loaded model
predictions = loaded_model.predict(features)

# Define a threshold function to obtain one-hot representations
def apply_threshold(predictions, threshold):
    return np.where(predictions >= threshold, 1, 0)

# Set a threshold value based on your team selection criteria
threshold = 0.7  # Adjust as needed

# Apply the threshold function to obtain one-hot representations
one_hot_predictions = apply_threshold(predictions, threshold)

# Display the one-hot representations of selected players
selected_players = selected_df[one_hot_predictions.flatten().astype(bool)]
print(selected_players[['player_name', 'playerRating']])

# Interpret the output in the context of selecting an optimal basketball team based on your criteria
# For example, you can select the players with '1' in the one-hot representation as your optimal team members.
# Sort the selected_df DataFrame by 'playerRating' in descending order
top_players = selected_df.sort_values(by='playerRating', ascending=False)

# Select the top 5 players
top_5_players = top_players.head(5)

# Display the top 5 players
print("My team")
print(top_5_players[['player_name', 'playerRating']])
