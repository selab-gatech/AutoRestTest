import time
import os

class Counter:
    def __init__(self):
        self.status_total = 0
        self.status_2xx = 0
        self.status_4xx = 0
        self.status_500 = 0
        self.directory = "./service_logs/"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.file_name = self.directory + "log-spotify.txt"

    def request(self, flow):
        # https://developer.spotify.com/console/get-playlists/
        flow.request.headers["Authorization"] = "Bearer no_token"
        with open(self.file_name, "a") as f:
            f.write("========REQUEST========\n")
            f.write(flow.request.method + "\n")
            f.write(flow.request.pretty_url + "\n")
            f.write(flow.request.text + "\n")
    def response(self, flow):
        with open(self.file_name, "a") as f:
            f.write("========RESPONSE========\n")
            f.write(str(time.time()) + "\n")
            f.write(str(flow.response.status_code) + "\n")
            f.write(flow.response.text + "\n")


addons = [Counter()]
