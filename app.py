# In[1]:
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from google.cloud import aiplatform, storage
import joblib

# # Vertex SDK for Python
# Using os.system to run shell commands from within Python
os.system('pip3 install --upgrade --quiet google-cloud-aiplatform')

# ### Set Google Cloud project information
# Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

# In[3]:
PROJECT_ID = "valiant-epsilon-473305-s3"
LOCATION = "us-central1"

# ### Create a Cloud Storage bucket
# Create a storage bucket to store intermediate artifacts such as datasets.

# In[5]:
BUCKET_URI = "gs://mlops-week1-database"
MODEL_ARTIFACT_DIR = "iris_classifier/model"
ARTIFACT_BUCKET = "gs://mlops-week1-database"

# ### Initialize Vertex AI SDK for Python
# To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

# In[7]:
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

# ### Simple Decision Tree model
# Build a Decision Tree model on iris data

# In[11]:
print("--- Starting Model Training ---")
# ✅ MODIFIED LINE: Reading CSV from the local 'data' folder
data = pd.read_csv('data/data.csv')
print("Data loaded successfully from local folder.")
print(data.head(5))

# In[12]:
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train.species
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

# In[13]:
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)
prediction = mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is', "{:.3f}".format(metrics.accuracy_score(prediction, y_test)))

# ## MODEL SAVING TO BUCKET IN A TIMESTAMP FOLDER

# In[15]:
def upload_file_to_gcs(bucket_name, source_file_path, destination_blob_path):
    """
    Uploads a single file to GCS.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    blob.upload_from_filename(source_file_path)
    print(f"Uploaded to gs://{bucket_name}/{destination_blob_path}")

# ====== Configuration ======
bucket_name = BUCKET_URI.replace("gs://", "")
local_dir = "artifacts"
os.makedirs(local_dir, exist_ok=True)

# ====== Save the model locally ======
model_filename = "model.joblib"
local_model_path = os.path.join(local_dir, model_filename)

joblib.dump(mod_dt, local_model_path)
print(f"Model saved locally to: {local_model_path}")

# ====== Upload to GCS with timestamp ======
timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
gcs_blob_path = f"artifacts/{timestamp}/{model_filename}"

upload_file_to_gcs(bucket_name, local_model_path, gcs_blob_path)

# ### DOWNLOAD LATEST MODEL FROM BUCKET AND TEST IT

# In[17]:
print("\n--- Starting Inference ---")
# ====== 1. Configuration ======
# Bucket name without the "gs://" prefix
BUCKET_NAME = "mlops-week1-database"
# The main folder where all timestamped model folders are stored
ARTIFACTS_FOLDER = "artifacts/"

# Local directory to download the model to
LOCAL_DOWNLOAD_DIR = "downloaded_model"
os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
downloaded_model_path = os.path.join(LOCAL_DOWNLOAD_DIR, "model.joblib")

# ====== 2. Find the LATEST model in GCS ======
print(f"Searching for models in gs://{BUCKET_NAME}/{ARTIFACTS_FOLDER}...")

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# List all files in the artifacts folder that are a model.joblib file
# The timestamps are sortable, so the "last" one alphabetically is the newest
all_model_blobs = [
    blob for blob in bucket.list_blobs(prefix=ARTIFACTS_FOLDER)
    if blob.name.endswith("model.joblib")
]

if not all_model_blobs:
    raise ValueError("No models found in the specified bucket and folder.")

# Get the blob with the latest timestamp
latest_model_blob = max(all_model_blobs, key=lambda blob: blob.name)
LATEST_GCS_MODEL_PATH = latest_model_blob.name

print(f"Found latest model: gs://{BUCKET_NAME}/{LATEST_GCS_MODEL_PATH}")

# ====== 3. Download the latest model from GCS ======
print("Downloading model...")
latest_model_blob.download_to_filename(downloaded_model_path)
print(f"Model downloaded to {downloaded_model_path}")

# ====== 4. Load and run inference ======
print("Loading model and running inference...")
loaded_model = joblib.load(downloaded_model_path)
inference_prediction = loaded_model.predict(X_test)

# ====== 5. Evaluate and print the results ======
accuracy = metrics.accuracy_score(inference_prediction, y_test)
print("\n--- Inference Complete ---")
print(f"Accuracy of the LATEST model is: {'{:.3f}'.format(accuracy)}")