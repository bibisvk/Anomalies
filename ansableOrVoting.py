import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import psycopg2
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
# Define features and target variable
features = df[['number_of_vertices', 'smallest_eigenvalue', 'matching_number',
               'second_largest_eigenvalue', 'radius', 'density', 'girth',
               'group_size', 'vertex_connectivity', 'diameter',
               'largest_eigenvalue', 'independence_number']]

# Assume 'snark' is the target variable indicating anomalies
target = df['snark'].map({True: 1, False: 0})  # Convert boolean to int

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train Models Individually
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Train the models
rf_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Step 5: OR Voting Prediction
# Predict using all models
rf_predictions = rf_model.predict(features)
mlp_predictions = mlp_model.predict(features)
svm_predictions = svm_model.predict(features)

# Apply OR logic: If any model predicts anomaly (1), the final result will be anomaly (True)
df['is_anomaly'] = (rf_predictions | mlp_predictions | svm_predictions).astype(bool)

# Calculate anomaly scores: Average of the probabilities from all models
rf_proba = rf_model.predict_proba(features)[:, 1]
mlp_proba = mlp_model.predict_proba(features)[:, 1]
svm_proba = svm_model.predict_proba(features)[:, 1]

df['anomaly_score'] = (rf_proba + mlp_proba + svm_proba) / 3

# Step 6: Visualization of feature distributions by anomaly status
for feature in features.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, hue='is_anomaly', multiple='stack', kde=True)
    plt.title(f'Distribution of {feature} by Anomaly Status')
    plt.show()

# Visualization of anomaly score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='anomaly_score', hue='is_anomaly', multiple='stack', kde=True, bins=50)
plt.title('Distribution of Anomaly Scores')
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
CREATE TABLE IF NOT EXISTS graph2p500000_or_voting_anomalies (
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

# Step 8: Insert Data into the New Table
for index, row in df.iterrows():
    # Convert NaN values to None for SQL NULL handling
    row = row.where(pd.notnull(row), None)

    insert_query = '''
    INSERT INTO graph2p500000_or_voting_anomalies (number_of_vertices, smallest_eigenvalue, matching_number, 
                                      second_largest_eigenvalue, radius, density, girth, 
                                      group_size, vertex_connectivity, diameter, largest_eigenvalue, 
                                      independence_number, snark, anomaly_score, is_anomaly)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
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

connection.commit()

# Step 9: Close the Database Connection
cursor.close()
connection.close()
