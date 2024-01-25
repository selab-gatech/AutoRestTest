import argparse
from traffic_fuzzer.fuzzer import Fuzzer

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-path', help='path to .cap path', required=True)
    args.add_argument('-time_budget', help='time budget to fuzzing (s)', required=True)
    args.add_argument('-server_address', help='server address e.g. http://foo.bar', required=True)
    args = args.parse_args()
    fuzzer = Fuzzer(cap_path=args.path, server_address=args.server_address, time_budget=float(args.time_budget))
    fuzzer.run()
