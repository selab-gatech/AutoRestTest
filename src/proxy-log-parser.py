import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LogEntry:
    request_type: Optional[str] = ""
    request_text: Optional[str] = ""
    status_code: Optional[int] = ""
    response_text: Optional[str] = ""

@dataclass
class StatusCodes:
    status_code: int
    count: int
    log_entries: List[LogEntry]

def parse_file(service_name: str, log_type: str):
    file_path = f"./{log_type}-logs/log-{service_name}.txt"
    log_object = {}
    try:
        with open(file_path, "r") as file:
            state = "searching"
            log_entry = LogEntry()
            for line in file:
                if line is None:
                    continue
                line = line.strip()
                if state == "searching":
                    if line == "========REQUEST========":
                        state = "request_type"
                elif state == "request_type":
                    log_entry.request_type = line
                    state = "request_text"
                elif state == "request_text":
                    if line == "========RESPONSE========":
                        state = "skip_line"
                    else:
                        log_entry.request_text += line
                elif state == "skip_line":
                    state = "status_code"
                elif state == "status_code":
                    log_entry.status_code = int(line)
                    state = "response_text"
                elif state == "response_text":
                    if line == "========REQUEST========":
                        state = "request_type"
                        if log_entry.status_code not in log_object:
                            log_object[log_entry.status_code] = StatusCodes(count=1, status_code=log_entry.status_code,
                                                                            log_entries=[log_entry])
                        else:
                            log_object[log_entry.status_code].count += 1
                            log_object[log_entry.status_code].log_entries.append(log_entry)
                        log_entry = LogEntry()
                    else:
                        log_entry.response_text += line
        return log_object
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def output_file(log_entries: dict, service_name: str, log_type: str):
    directory = f"./parsed-proxy-logs/{log_type}-logs/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"./parsed-proxy-logs/{log_type}-logs/{service_name}.txt", "w") as file:
        file.write(f"MOREST OUTPUT FOR {service_name}\n")
        for status_code, status_data in log_entries.items():
            file.write("========================================\n")
            file.write(f"Status Code Category: {status_code}\n")
            file.write(f"Count: {status_data.count}\n")
            file.write("----------------------------------------\n")
            for entry in status_data.log_entries:
                file.write(f"Request Type: {entry.request_type}\n")
                file.write(f"Request Text: {entry.request_text}\n")
                file.write(f"Status Code: {entry.status_code}\n")
                file.write(f"Response Text: {entry.response_text}\n")
                file.write("\n")

def main():
    services = ["language-tool", "ocvn", "ohsome", "omdb", "rest-countries", "youtube"]
    testing_services = ["omdb"]
    log_types = ["morest"]
    for log_type in log_types:
        print("Parsing log type: ", log_type)
        for service in services:
            print("Parsing service: ", service)
            log_entries = parse_file(service, log_type)
            if log_entries:
                output_file(log_entries, service, log_type)

if __name__ == "__main__":
    main()

