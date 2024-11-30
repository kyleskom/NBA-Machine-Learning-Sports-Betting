import boto3
import json
from botocore.exceptions import ClientError


# get proxy details
def _get_proxy_dict():
    secret_name = "proxy-details"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        print(f'Client error: {e}')
        raise e

    secret = get_secret_value_response['SecretString']
    secret_dict = json.loads(secret)

    PROXY_USERNAME = secret_dict.get("PROXY_USERNAME")
    PROXY_CITY_CODE = secret_dict.get("PROXY_CITY_CODE")
    PROXY_PASSWORD = secret_dict.get("PROXY_PASSWORD")
    PROXY_URL = secret_dict.get("PROXY_URL")
    PROXY_PORT = secret_dict.get("PROXY_PORT")

    proxy = f'https://customer-{PROXY_USERNAME}-{PROXY_CITY_CODE}:{PROXY_PASSWORD}@{PROXY_URL}:{PROXY_PORT}'

    proxy_dict = {
        "https": proxy
    }

    return proxy_dict
