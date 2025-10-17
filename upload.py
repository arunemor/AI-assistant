import boto3
from botocore.exceptions import NoCredentialsError, ClientError


BUCKET_NAME = "uploadpdf2025"

# File to upload
file_path = "ubuntu/home/class5.pdf"  # replace with your PDF or any file
s3_key = "uploadpdf2025",       # name in S3

try:
    # Create S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id="AKIAVS4D7FKQHUSBOPFO",
        aws_secret_access_key="12Bv5Dj5w5nNgpKtfQ0tywku76zv0DRhgvfWeoA",
        region_name="ap-south-1"
     )

    # Upload file
    s3.upload_file(file_path, BUCKET_NAME, s3_key)
    print(f"✅ File '{file_path}' uploaded successfully to bucket '{BUCKET_NAME}' as '{s3_key}'.")

except FileNotFoundError:
    print("❌ The file was not found.")
except NoCredentialsError:
    print("❌ AWS credentials not available or incorrect.")
except ClientError as e:
    print(f"❌ Failed to upload: {e}")
