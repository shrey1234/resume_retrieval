import boto3
import docx2txt
import helper as helper

s3_resource = boto3.resource('s3',aws_access_key_id = 'AKIA3RL2X62ZCFO46JO2',aws_secret_access_key = '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')
bucket = s3_resource.Bucket("jobresumes")


def get_list_of_resumes():
  arr_resume = []
  for my_bucket_object in bucket.objects.all():
    arr_resume.append(URL+my_bucket_object.key)
  return arr_resume