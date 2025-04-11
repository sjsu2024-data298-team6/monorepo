import json
import boto3
import os
import re

sqs = boto3.client("sqs")

QUEUE_URL = os.environ.get("SQS_QUEUE_URL")


def lambda_handler(event, context):
    try:
        data = event.get("queryStringParameters", {}).get("data")
        task = event.get("queryStringParameters", {}).get("task")

        if not isinstance(data, dict):
            data = json.loads(data)

        if "tags" in data.keys() and len(data["tags"]) > 0:
            for tag in data["tags"]:
                if re.search(r"[^A-Za-z0-9_-]", tag):
                    return {"statusCode": 400, "body": json.dumps("Bad Request! Invalid characters in tags")}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps(f"Error reading form data: {str(e)}")}

    try:
        response = sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps({"data": data, "task": task}))

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "data sent to SQS", "sqs_message_id": response["MessageId"]}),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps(f"Error sending message to SQS: {str(e)}")}
