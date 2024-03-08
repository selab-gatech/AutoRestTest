import time
import os

class Counter:
    def __init__(self):
        self.directory = "./service_logs/"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.filename = self.directory + os.environ.get('LOG_FILE', 'default.log')

    def request(self, flow):
        with open(self.filename, "a") as f:
            f.write("========REQUEST========\n")
            f.write(flow.request.method + "\n")
            f.write(flow.request.pretty_url + "\n")
            f.write(flow.request.text + "\n")

    def response(self, flow):
        with open(self.filename, "a") as f:
            f.write("========RESPONSE========\n")
            f.write(str(time.time()) + "\n")
            f.write(str(flow.response.status_code) + "\n")
            f.write(flow.response.text + "\n")

addons = [Counter()]