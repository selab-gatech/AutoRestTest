import time
import os

class Counter:
    def __init__(self):
        self.filename = os.environ.get('LOG_FILE', 'default.log')

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