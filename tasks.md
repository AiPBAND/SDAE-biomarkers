# Configuration
## Authentication

First paste the key in the JSON file. 

```shell
$env:GOOGLE_APPLICATION_CREDENTIALS="credentials\secret_key.json"
```
## Setting up

Set up the project directory, where the script and outpout should be.

```shell
$env:BUCKET_NAME="biomakers-autoencoder"

$env:REGION="eu-west4"

gsutil mb -l $REGION gs://$BUCKET_NAME
```

Package the Python module and upload to the bucket.