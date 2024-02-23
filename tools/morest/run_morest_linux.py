import argparse
from config import DEV_SERVER_ADDRESS
from tools.morest.fuzzer import MorestFuzzer

def argument_parse() -> (str, str):
    service_urls = {
        'fdic': "http://0.0.0.0:9001",
        'genome-nexus': "http://0.0.0.0:9002",
        'language-tool': "http://0.0.0.0:9003",
        'ocvn': "http://0.0.0.0:9004",
        'ohsome': "http://0.0.0.0:9005",
        'omdb': "http://0.0.0.0:9006",
        'rest-countries': "http://0.0.0.0:9007",
        'spotify': "http://0.0.0.0:9008",
        'youtube': "http://0.0.0.0:9009"
    }
    parser = argparse.ArgumentParser(description='Generate requests based on API specification.')
    parser.add_argument('service', help='The service specification to use.')
    parser.add_argument('is_local', help='Whether the services are loaded locally or not.')
    args = parser.parse_args()
    api_url = service_urls.get(args.service)
    is_local = args.is_local
    if api_url is None:
        print(f"Service '{args.service}' not recognized. Available services are: {list(service_urls.keys())}")
        exit(1)
    if is_local not in {"true", "false"}:
        print(f"Invalid value for 'is_local'. Must be either 'true' or 'false'.")
        exit(1)
    if is_local == "false":
        api_url = api_url.replace("0.0.0.0", DEV_SERVER_ADDRESS)  # use config.py for DEV_SERVER_ADDRESS var
        api_url = api_url.replace(":9", ":8") # use public server proxy ports
    return args.service, api_url

if __name__ == "__main__":
    service_name, api_url = argument_parse()
    file_path = f"../../specs/original/oas/{service_name}.yaml"
    fuzzer = MorestFuzzer(file_path, api_url)
    fuzzer.run()
