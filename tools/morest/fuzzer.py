import argparse
import os

from prance import ResolvingParser

from build_graph import parse
from fuzzer.fuzzer import APIFuzzer
from utils.auth_util import get_token, SUT
from config import DEV_SERVER_ADDRESS

def default_reclimit_handler(limit, parsed_url, recursions=()):
    """Raise prance.util.url.ResolutionError."""
    return {
        "type": "object",
        "name": "Recursive Dependency",
        "properties": {}
    }

def main():
    # testing arguments format: swagger address, server address, system under test's name, *args to obtain token
    test_args = ['$SPEC_HERE', '$URL_HERE', SUT.BITBUCKET]
    parser = ResolvingParser(test_args[0],
                             recursion_limit_handler=default_reclimit_handler, backend='openapi-spec-validator', strict=False)
    apis, odg = parse(parser.specification)
    odg.draw()
    # return
    # headers = get_token(SUT.SPREE, "http://192.168.74.135:3000")
    # api_fuzzer = APIFuzzer(apis, parser.specification, odg, 'https://demo.traccar.org/api/')
    # api_fuzzer = APIFuzzer(apis, parser.specification, odg, 'http://192.168.74.135:3000', pre_defined_headers=headers)
    # token = get_token(*test_args[2:])
    api_fuzzer = APIFuzzer(apis, parser.specification, odg, test_args[1], time_budget=60)
    api_fuzzer.run()

def run_fuzzer(spec_path, server_url):
    test_args = [spec_path, server_url, SUT.BITBUCKET]
    parser = ResolvingParser(test_args[0],
                             recursion_limit_handler=default_reclimit_handler, backend='openapi-spec-validator',
                             strict=False)
    apis, odg = parse(parser.specification)
    odg.draw()
    api_fuzzer = APIFuzzer(apis, parser.specification, odg, test_args[1], time_budget=60) # changed time budget from 3600 to 60
    api_fuzzer.run()

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

class MorestFuzzer:
    def __init__(self, spec_path, server_url):
        self.spec_path = spec_path
        self.server_url = server_url

    def run(self):
        run_fuzzer(self.spec_path, self.server_url)

if __name__ == '__main__':
    service_name, api_url = argument_parse()
    file_path = f"../../specs/original/oas/{service_name}.yaml"
    fuzzer = MorestFuzzer(file_path, api_url)
    fuzzer.run()
    #main()
