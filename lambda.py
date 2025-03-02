import json
import boto3
import os
import ast

sqs = boto3.client("sqs")

QUEUE_URL = os.environ.get("SQS_QUEUE_URL")


def lambda_handler(event, context):
    try:
        data = event.get("queryStringParameters", {}).get("data")
        task = event.get("queryStringParameters", {}).get("task")

        if not isinstance(data, dict):
            data = json.loads(data)

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error reading form data: {str(e)}"),
        }

    try:
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(
                {
                    "data": data,
                    "task": task,
                }
            ),
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "data sent to SQS",
                    "sqs_message_id": response["MessageId"],
                }
            ),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error sending message to SQS: {str(e)}"),
        }
