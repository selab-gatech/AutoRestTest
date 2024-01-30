import argparse
import os
import json
import time


def get_coverage_logs(path):
    files = os.listdir(path)

    return [path + "/" + f for f in files]


def read_data(path=""):
    with open(path, 'r', encoding="utf-8") as jsonfile:
        data = jsonfile.read()
        obj = json.loads(data)
        return obj


def run_single(folder, result={}):
    logs = (get_coverage_logs(folder))
    for log in logs:
        print("read", log)
        data = read_data(log)
        for file_name in data:
            if not (file_name in result):
                result[file_name] = list()
            for line in data[file_name]:
                if line in result[file_name]:
                    continue
                result[file_name].append(line)
        os.unlink(log)
    line_count = 0
    for file_name in result:
        if file_name == "line_count":
            continue
        line_count += len(result[file_name])
    result["line_count"] = line_count
    print("total line count", line_count)


def main(args):
    coverage_folder = args.c
    output_file = args.o
    result = {}
    while True:
        print("read from", coverage_folder)
        print("write to", output_file)
        run_single(coverage_folder, result)
        with open(output_file, 'w') as data:
            json.dump(result, data)
        time.sleep(3)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', default="coverage", help="coverage folder")
    args.add_argument('-o', default="coverage.json", help="output file, default coverage.json")
    main(args.parse_args())
