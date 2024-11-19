import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2

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

# Step 3: Preprocessing (Scaling features for SVM)
# Define features and target variable
features = df[['number_of_vertices', 'smallest_eigenvalue', 'matching_number',
               'second_largest_eigenvalue', 'radius', 'density', 'girth',
               'group_size', 'vertex_connectivity', 'diameter',
               'largest_eigenvalue', 'independence_number']]

target = df['snark'].map({True: 1, False: 0})  # Convert boolean to int

# Standardize the feature values (SVM performs better with standardized data)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Step 4: Train SVM Model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Step 5: Predict Anomalies
# Predict on the original dataset (full scaled dataset)
df['is_anomaly'] = svm_model.predict(scaler.transform(features))

# Use predicted probabilities to derive anomaly scores
df['anomaly_score'] = svm_model.predict_proba(scaler.transform(features))[:, 1]  # Probability of being a 'snark'

# Convert 'is_anomaly' back to boolean
df['is_anomaly'] = df['is_anomaly'].astype(bool)

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
CREATE TABLE IF NOT EXISTS graph2p500000_svm_anomalies (
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
    INSERT INTO graph2p500000_svm_anomalies (number_of_vertices, smallest_eigenvalue, matching_number, 
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
