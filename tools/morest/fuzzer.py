import os

from prance import ResolvingParser

from build_graph import parse
from fuzzer.fuzzer import APIFuzzer
from utils.auth_util import get_token, SUT


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

class MorestFuzzer:
    def __init__(self, spec_path, server_url):
        self.spec_path = spec_path
        self.server_url = server_url

    def run(self):
        run_fuzzer(self.spec_path, self.server_url)

def run_fuzzer(spec_path, server_url):
    test_args = [spec_path, server_url, SUT.BITBUCKET]
    parser = ResolvingParser(test_args[0],
                             recursion_limit_handler=default_reclimit_handler, backend='openapi-spec-validator',
                             strict=False)
    apis, odg = parse(parser.specification)
    odg.draw()
    api_fuzzer = APIFuzzer(apis, parser.specification, odg, test_args[1], time_budget=3600)
    api_fuzzer.run()

if __name__ == '__main__':
    main()
