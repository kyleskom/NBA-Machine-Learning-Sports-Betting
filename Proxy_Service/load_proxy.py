import os
from dotenv import load_dotenv


# get proxy details
def _get_proxy():
    load_dotenv(dotenv_path="/home/ubuntu/NBA-App/NBA-Machine-Learning-Sports-Betting/environment.env")
    PROXY_USERNAME = os.getenv('PROXY_USERNAME')
    PROXY_CITY_CODE = os.getenv('PROXY_CITY_CODE')
    PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
    PROXY_URL = os.getenv('PROXY_URL')
    PROXY_PORT = os.getenv('PROXY_PORT')
    proxy = f'https://customer-{PROXY_USERNAME}-{PROXY_CITY_CODE}:{PROXY_PASSWORD}@{PROXY_URL}:{PROXY_PORT}'

    return proxy
