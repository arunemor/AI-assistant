import boto3
import PyPDF2
import io

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    print("✅ Lambda triggered with event:", event)

    # Extract bucket and key from the event
    record = event['Records'][0]
    source_bucket = record['s3']['bucket']['name']
    object_key = record['s3']['object']['key']
    destination_bucket = 'extractpdf202'  # your destination bucket

    print(f"📂 Source bucket: {source_bucket}, Key: {object_key}")

    try:
        s3_object = s3_client.get_object(Bucket=source_bucket, Key=object_key)
        print("📥 PDF downloaded successfully")

        pdf_content = s3_object['Body'].read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))

        extracted_text = ""
        for i, page in enumerate(pdf_reader.pages):
            extracted_text += page.extract_text() or ""
            print(f"📖 Extracted page {i+1}")

        # Upload text to destination bucket
        output_key = object_key.replace(".pdf", ".txt")
        s3_client.put_object(
            Bucket=destination_bucket,
            Key=output_key,
            Body=extracted_text.encode('utf-8')
        )
        print(f"✅ Extracted text uploaded as {output_key} to {destination_bucket}")

        return {
            'statusCode': 200,
            'body': f"Extracted text from {object_key} and saved to {destination_bucket}/{output_key}"
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise e
