$env:GOOGLE_APPLICATION_CREDENTIALS="credentials\secret_key.json"
$env:BUCKET_NAME="cloud-ai-platform-3797bc57-6ed6-4e69-a850-2d18e0d363a2"
$env:REGION="eu-west4"
gsutil mb -l $REGION gs://$BUCKET_NAME

python setup.py sdist --formats=gztar

$env:CLOUD_STORAGE_DIRECTORY="gs://sdae-models"