import os

from prance import ResolvingParser

from build_graph import parse
from fuzzer.fuzzer import APIFuzzer
import requests
import json


def default_reclimit_handler(limit, parsed_url, recursions=()):
    """Raise prance.util.url.ResolutionError."""
    return {
        "type": "string",
        "name": "Recursive Dependency"
    }


def spree_login(ip_address):
    spree_payload = {
        "grant_type": "password",
        "username": "spree@example.com",
        "password": "spree123"
    }
    spree_login = requests.post('%s/spree_oauth/token'%str(ip_address), json = spree_payload)
    print(spree_login.text)
    token = "Bearer " + json.loads(spree_login.text)['access_token']
    # token = "Bearer 894fa3111b5a8c4ff8778f2e20f067a367b665918a6eac28"
    return token

def main():

    parser = ResolvingParser('./specs/magento.json', backend = 'openapi-spec-validator')
    apis, odg = parse(parser.specificati
    api_fuzzer = APIFuzzer(apis, parser.specification, odg, 'http://0.0.0.0', time_budget=300)


    # run
    api_fuzzer.run()


if __name__ == '__main__':
    main()
