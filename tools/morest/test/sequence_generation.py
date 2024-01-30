import os

from prance import ResolvingParser

from build_graph import parse
from fuzzer.fuzzer import APIFuzzer


def default_reclimit_handler(limit, parsed_url, recursions=()):
    """Raise prance.util.url.ResolutionError."""
    return {
        "type": "string",
        "name": "Recursive Dependency"
    }


def main():
    parser = ResolvingParser('../pet.json')#,backend = 'openapi-spec-validator')
    apis, odg = parse(parser.specification)
    odg.generate_sequence()
    # odg.draw()

if __name__ == '__main__':
    main()

