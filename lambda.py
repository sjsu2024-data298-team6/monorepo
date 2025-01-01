import json
import boto3
import os
import ast

sqs = boto3.client("sqs")

QUEUE_URL = os.environ.get(
    "SQS_QUEUE_URL"
)


def lambda_handler(event, context):
    try:
        url = event.get("queryStringParameters", {}).get("url")
        dtype = event.get("queryStringParameters", {}).get("dataset_type")
        names = event.get("queryStringParameters", {}).get("names")
        model = event.get("queryStringParameters", {}).get("model")
        params = event.get("queryStringParameters", {}).get("params")

        if not isinstance(params, dict):
            params = json.loads(ast.literal_eval(params))

        if not isinstance(names, list):
            names = [x.strip() for x in names.split(",")]

        print(type(url), type(dtype), type(names), type(model), type(params))

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
                    "url": url,
                    "dtype": dtype,
                    "names": names,
                    "model": model,
                    "params": params,
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
