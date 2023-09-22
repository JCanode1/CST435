
import pandas as pd
import random


# Specify the file path
file_path = "all_seasons.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Display the head (first few rows) of the DataFrame
print("Head of the DataFrame:")
print(df.head())

# Display the tail (last few rows) of the DataFrame
print("\nTail of the DataFrame:")
print(df.tail())


df['draft_year'] = df['season'].str.split('-').str[0].astype(int)

# Define the target year and the window size
target_year = 1996  # Replace with your desired target year
window_size = 5

# Calculate the start and end years of the window
start_year = target_year - window_size
end_year = target_year

# Filter the dataset to include only records within the 5-year window
filtered_df = df[(df['draft_year'] >= start_year) & (df['draft_year'] <= end_year)]

# Ensure the filtered dataset has at least 100 players
if len(filtered_df) < 100:
    print("There are not enough players within the specified window.")
else:
    # Randomly select 100 players from the filtered dataset
    random.seed(42)  # Set a random seed for reproducibility
    selected_players = random.sample(range(len(filtered_df)), 100)

    # Create a DataFrame containing the selected players
    selected_df = filtered_df.iloc[selected_players]

    # Display the selected players
    print(selected_df)



# Define the weights for each metric
weight_GP = 0.7
weight_PTS = 1.0
weight_REB = 0.8
weight_AST = 0.9
weight_Net_Rating = 0.4
weight_OREB_pct = 0.6
weight_DREB_pct = 0.5
weight_USG_pct = 0.3
weight_TS_pct = 0.2
weight_AST_pct = 0.1

# Calculate the player rating for each row in the selected_df DataFrame
selected_df['playerRating'] = (
    selected_df['gp'] * weight_GP +
    selected_df['pts'] * weight_PTS +
    selected_df['reb'] * weight_REB +
    selected_df['ast'] * weight_AST +
    selected_df['net_rating'] * weight_Net_Rating +
    selected_df['oreb_pct'] * weight_OREB_pct +
    selected_df['dreb_pct'] * weight_DREB_pct +
    selected_df['usg_pct'] * weight_USG_pct +
    selected_df['ts_pct'] * weight_TS_pct +
    selected_df['ast_pct'] * weight_AST_pct
)

# Display the DataFrame with player ratings
print(selected_df[['player_name', 'playerRating']])



import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Define features (player characteristics) and target (player ratings)
features = selected_df[['gp', 'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']].values
target = selected_df['playerRating'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a Sequential model
model = tf.keras.models.Sequential()

# Add an input layer with the number of features as input dimension
model.add(tf.keras.layers.Input(shape=(features.shape[1],)))

# Add one or more hidden layers with desired neurons and activation functions
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))

# Add an output layer with a single neuron (regression task)
model.add(tf.keras.layers.Dense(1))

# Compile the model with an appropriate optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {mae}')

# Save the trained model to a file
model.save('basketball_player_rating_model.h5')
