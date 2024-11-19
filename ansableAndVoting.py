import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Step 3: Preprocessing (if needed)
features = df[['number_of_vertices', 'smallest_eigenvalue', 'matching_number',
               'second_largest_eigenvalue', 'radius', 'density', 'girth',
               'group_size', 'vertex_connectivity', 'diameter',
               'largest_eigenvalue', 'independence_number']]

# Target variable 'snark' indicates anomalies (1 for anomaly, 0 for normal)
target = df['snark'].map({True: 1, False: 0})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Define Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Train models in parallel
def train_model(model, X, y):
    model.fit(X, y)
    return model

models = Parallel(n_jobs=-1)(delayed(train_model)(model, X_train, y_train) for model in [rf_model, mlp_model, svm_model])

# Step 5: AND Voting Method for Prediction
def predict_and_vote(models, X):
    # Get predictions from each model
    predictions = np.array([model.predict(X) for model in models])
    # AND voting: All models must agree for an anomaly
    and_vote = np.all(predictions == 1, axis=0)
    return and_vote.astype(int)

# Predict using AND voting on the full dataset
df['is_anomaly'] = predict_and_vote(models, features)

# Step 6: Visualization of feature distributions by anomaly status
for feature in features.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='is_anomaly', multiple='stack', kde=True)
    plt.title(f'Distribution of {feature} by Anomaly Status (AND Voting)')
    plt.show()

# Visualization of anomaly score distribution (since AND voting doesn't give a direct score, use prediction count)
df['anomaly_score'] = predict_and_vote(models, features)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='anomaly_score', hue='is_anomaly', multiple='stack', kde=True, bins=50)
plt.title('Distribution of Anomaly Predictions (AND Voting)')
plt.show()

# Step 7: Create a New Table for Anomalies in the Database
connection = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="postgres",
    password="password"
)

cursor = connection.cursor()

create_table_query = '''
CREATE TABLE IF NOT EXISTS graph2p500000_and_voting_anomalies (
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
    anomaly_score INT,
    is_anomaly BOOLEAN
);
'''
cursor.execute(create_table_query)
connection.commit()

# Step 8: Insert Data into the New Table
for index, row in df.iterrows():
    # Convert any NaN values to None (for SQL NULL handling)
    row = row.where(pd.notnull(row), None)

    insert_query = '''
    INSERT INTO graph2p500000_and_voting_anomalies (number_of_vertices, smallest_eigenvalue, matching_number, 
                                      second_largest_eigenvalue, radius, density, girth, 
                                      group_size, vertex_connectivity, diameter, largest_eigenvalue, 
                                      independence_number, snark, anomaly_score, is_anomaly)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    try:
        # Ensure proper types are passed for each SQL argument
        cursor.execute(insert_query, (int(row['number_of_vertices']),
                                      float(row['smallest_eigenvalue']),
                                      int(row['matching_number']),
                                      float(row['second_largest_eigenvalue']),
                                      int(row['radius']),
                                      float(row['density']),
                                      int(row['girth']),
                                      int(row['group_size']),
                                      int(row['vertex_connectivity']),
                                      int(row['diameter']),
                                      float(row['largest_eigenvalue']),
                                      int(row['independence_number']),
                                      bool(row['snark']),
                                      float(row['anomaly_score']),
                                      bool(row['is_anomaly'])))
    except Exception as e:
        print(f"Error inserting row {index}: {e}")

connection.commit()

# Step 9: Close the Database Connection
cursor.close()
connection.close()
