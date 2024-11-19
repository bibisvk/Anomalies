import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Connect to the Database
connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="postgres",
    password="password"
)

# Step 2: Retrieve the Data
query = "SELECT * FROM graph2p500000;"
df = pd.read_sql(query, connection)
connection.close()

# Step 3: Preprocessing (Scaling features for Autoencoder)
# Define features and target variable
features = df[['number_of_vertices', 'smallest_eigenvalue', 'matching_number',
               'second_largest_eigenvalue', 'radius', 'density', 'girth',
               'group_size', 'vertex_connectivity', 'diameter',
               'largest_eigenvalue', 'independence_number']]

# Standardize the feature values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Define and Train Autoencoder Model
input_dim = scaled_features.shape[1]  # Number of features

# Define the autoencoder model
autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(input_dim, activation='linear')  # Reconstruct the input
])

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 5: Predict Anomalies using Reconstruction Error
# Get the reconstruction of the input data
reconstructed = autoencoder.predict(scaled_features)

# Calculate the reconstruction error (MSE between input and reconstruction)
reconstruction_error = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)

# Add reconstruction error to the dataframe as 'anomaly_score'
df['anomaly_score'] = reconstruction_error

# Define threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 95)  # Top 5% most significant errors are anomalies

# Label anomalies based on the threshold
df['is_anomaly'] = df['anomaly_score'] > threshold

# Step 6: Visualization of feature distributions by anomaly status
for feature in features.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='is_anomaly', multiple='stack', kde=True)
    plt.title(f'Distribution of {feature} by Anomaly Status')
    plt.show()

# Step 7: Visualization of anomaly score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='anomaly_score', hue='is_anomaly', multiple='stack', kde=True, bins=50)
plt.title('Distribution of Anomaly Scores')
plt.show()

# Step 8: Create a New Table for Anomalies in the Database
connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="postgres",
    password="password"
)

cursor = connection.cursor()

create_table_query = '''
CREATE TABLE IF NOT EXISTS graph2p500000_autoencoder_anomalies (
    number_of_vertices INT,
    smallest_eigenvalue FLOAT8,
    matching_number INT,
    second_largest_eigenvalue FLOAT8,
    radius INT,
    density FLOAT8,
    girth INT,
    group_size INT,
    vertex_connectivity INT,
    diameter INT,
    largest_eigenvalue FLOAT8,
    independence_number INT,
    snark BOOLEAN,
    anomaly_score FLOAT8,
    is_anomaly BOOLEAN
);
'''
cursor.execute(create_table_query)
connection.commit()

# Step 9: Insert Data into the New Table
for index, row in df.iterrows():
    insert_query = '''
    INSERT INTO graph2p500000_autoencoder_anomalies (number_of_vertices, smallest_eigenvalue, matching_number, 
                                                     second_largest_eigenvalue, radius, density, girth, 
                                                     group_size, vertex_connectivity, diameter, largest_eigenvalue, 
                                                     independence_number, snark, anomaly_score, is_anomaly)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    cursor.execute(insert_query, (row['number_of_vertices'], row['smallest_eigenvalue'], row['matching_number'],
                                  row['second_largest_eigenvalue'], row['radius'], row['density'],
                                  row['girth'], row['group_size'], row['vertex_connectivity'],
                                  row['diameter'], row['largest_eigenvalue'], row['independence_number'],
                                  row['snark'], row['anomaly_score'], row['is_anomaly']))

connection.commit()

# Step 10: Close the Database Connection
cursor.close()
connection.close()
