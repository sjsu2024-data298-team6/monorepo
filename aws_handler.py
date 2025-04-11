import boto3
from keys import GeneralKeys
import logging
import traceback
import zipfile
import os


class SNSHandler:
    def __init__(self, region_name="us-east-1", logger=None):
        if logger is not None:
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
                self.logger.error("Failed to send message")
                self.logger.debug(e)
                self.logger.debug(traceback.format_exc())
            pass


class S3Handler:
    def __init__(self, bucket, logger=None):
        if logger is not None:
            assert isinstance(logger, logging.Logger)
        self.logger = logger
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def upload_folder_to_s3(self, folder_path, s3_key):
        if GeneralKeys.DEPLOYMENT == "dev":
            if self.logger is not None:
                self.logger.info("Not uploading in dev env")
            ret = ""
            s3_key = "test.s3"
        else:
            rets = []
            s3_key = os.path.join(s3_key, folder_path)
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    local = os.path.join(root, filename)
                    rel = os.path.relpath(local, folder_path)
                    upload_ret, _ = self.upload_file_to_s3(local, s3_key, rel)
                    rets.append(upload_ret)
            rets = "\n".join(rets)
            ret = f"Uploaded folder {folder_path} to s3://{self.bucket}/{s3_key}/\n\n{rets}"
            if self.logger is not None:
                self.logger.info(ret)

        return ret, s3_key

    def upload_file_to_s3(self, file_path, s3_path, file_name):
        if GeneralKeys.DEPLOYMENT == "dev":
            if self.logger is not None:
                self.logger.info("Not uploading in dev env")
            ret = ""
            s3_key = "test.s3"
        else:
            s3_key = os.path.join(s3_path, file_name)

            self.s3.upload_file(file_path, self.bucket, s3_key)
            ret = f"Uploaded {file_path} to s3://{self.bucket}/{s3_key}"
            if self.logger is not None:
                self.logger.info(ret)

        try:
            os.remove(file_path)
        except:
            pass

        return ret, s3_key

    def upload_zip_to_s3(self, local_path, s3_path, zip_name="upload.zip"):
        zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, local_path))

        if GeneralKeys.DEPLOYMENT == "dev":
            if self.logger is not None:
                self.logger.info("Not uploading in dev env")
            ret = ""
            s3_key = "test.s3"
        else:
            s3_key = os.path.join(s3_path, zip_name)

            self.s3.upload_file(zip_path, self.bucket, s3_key)
            ret = f"Uploaded {zip_path} to s3://{self.bucket}/{s3_key}"
            if self.logger is not None:
                self.logger.info(ret)

        try:
            os.remove(zip_path)
        except:
            pass

        return ret, s3_key

    def check_file_exists(self, key, logger=None):
        if logger is not None:
            logger.info(f"Checking if {key} exists in {self.bucket}")
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False

    def download_file(self, key, name):
        if self.logger is not None:
            self.logger.info(
                f"Downloading {key} from {self.bucket} to {os.getcwd()}/{name}"
            )
        try:
            self.s3.download_file(self.bucket, key, name)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(
                    f"Failed to download key {key} from {self.bucket} to {name}"
                )
                self.logger.debug(e)
                self.logger.debug(traceback.format_exc())
