import boto3
from keys import GeneralKeys
import logging
import traceback


class SNSHandler:
    def __init__(self, region_name="us-east-1", logger=None):
        assert isinstance(logger, logging.Logger)
        self.logger = logger
        self.sns = boto3.client("sns", region_name=region_name)

    def send(self, subject, message):
        try:
            self.sns.publish(
                TargetArn=GeneralKeys.SNS_ARN,
                Message=message,
                Subject=subject,
            )

        except Exception as e:
            if self.logger is not None:
                self.logger.info("Failed to send message")
                self.logger.debug(e)
                self.logger.debug(traceback.format_exc())
            pass
