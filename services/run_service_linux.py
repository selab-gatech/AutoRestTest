import os
import sys
import time
import subprocess


# This function is used to run a service given the service path, class name, and coverage collecting port number
def run_service(service_path, class_name, port_number):
    # Read classpath from cp.txt file in the service path
    with open(service_path + "/cp.txt", 'r') as f:
        cp = f.read()
    if "languagetool" in service_path:
        with open(service_path + "/run.sh", 'w') as f:
            f.write(
                "java " + cov + port_number + ".exec" + " -cp target/classes:target/test-classes:" + cp + ' ' + class_name)
        subprocess.run(
            ". ./java8.env && cd " + service_path + " && tmux new-session -d -s language-tool-server 'sh run.sh'",
            shell=True)
    elif "ocvn" in service_path:
        with open(service_path + "/run.sh", 'w') as f:
            f.write(
                "java " + cov + port_number + ".exec" + " -cp target/classes:target/test-classes:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/rt.jar:" + base + "/emb/cs/rest-gui/ocvn/web/target/classes:" + cp + ' ' + class_name)
        subprocess.run(
            ". ./java8.env && cd " + service_path + " && tmux new-session -d -s ocvn-server 'sudo sh run.sh'",
            shell=True)
    else:
        with open(service_path + "/run.sh", 'w') as f:
            f.write(
                "java " + cov + port_number + ".exec" + " -cp target/classes:target/test-classes:" + cp + ' ' + class_name)
        subprocess.run(
            ". ./java11.env && cd " + service_path + " && tmux new-session -d -s youtube-server 'sh run.sh'",
            shell=True)


if __name__ == "__main__":
    name = sys.argv[1]
    token = sys.argv[2]
    base = os.getcwd()

    cov = "-javaagent:" + base + "/../org.jacoco.agent-0.8.7-runtime.jar=destfile=jacoco"

    if name == "fdic":
        subprocess.run("tmux new -d -s fdic-proxy 'LOG_FILE=log-fdic.txt mitmproxy --mode reverse:https://banks.data.fdic.gov -p 9001 -s proxy.py'", shell=True)
    elif name == "genome-nexus":
        subprocess.run("sudo docker stop gn-mongo", shell=True)
        subprocess.run("sudo docker rm gn-mongo", shell=True)
        subprocess.run("sudo docker run --name=gn-mongo --restart=always -p 27018:27017 -d genomenexus/gn-mongo:latest", shell=True)
        time.sleep(180)
        subprocess.run(
            "tmux new -d -s genome-nexus-server '. ./java8.env && java " + cov + "9002.exec" + " -jar ./genome-nexus/web/target/web-0-unknown-version-SNAPSHOT.war'",
            shell=True)
        subprocess.run(
            "tmux new -d -s genome-nexus-proxy 'LOG_FILE=log-genome-nexus.txt mitmproxy --mode reverse:http://0.0.0.0:50110 -p 9002 -s proxy.py'",
            shell=True)
    elif name == "language-tool":
        run_service("emb/em/embedded/rest/languagetool", "em.embedded.org.languagetool.RunServer",
                    "9003")
        subprocess.run(
            "tmux new -d -s language-tool-proxy 'LOG_FILE=log-language-tool.txt mitmproxy --mode reverse:http://0.0.0.0:50111 -p 9003 -s proxy.py'",
            shell=True)
    elif name == "ocvn":
        run_service("emb/em/embedded/rest/ocvn", "em.embedded.org.devgateway.ocvn.RunServer", "9004")
        subprocess.run("tmux new -d -s ocvn-proxy 'LOG_FILE=log-ocvn.txt mitmproxy --mode reverse:http://0.0.0.0:50112 -p 9004 -s proxy.py'",
                       shell=True)
    elif name == "ohsome":
        subprocess.run(
            "tmux new -d -s ohsome-proxy 'LOG_FILE=log-ohsome.txt mitmproxy --mode reverse:https://api.ohsome.org -p 9005 -s proxy.py'",
            shell=True)
    elif name == "omdb":
        subprocess.run("tmux new -d -s omdb-proxy 'LOG_FILE=log-omdb.txt mitmproxy --mode reverse:https://omdbapi.com/ -p 9006 -s omdb.py'", shell=True)
    elif name == "rest-countries":
        subprocess.run("tmux new -d -s rest-countries-proxy 'LOG_FILE=log-rest-countries.txt mitmproxy --mode reverse:https://restcountries.com -p 9007 -s proxy.py'", shell=True)
    elif name == "spotify":
        with open("spotify.py", "r") as file:
            content = file.read()

        content = content.replace("TOKEN_HERE", token)

        with open("spotify.py", "w") as file:
            file.write(content)
        subprocess.run("tmux new -d -s spotify-proxy 'LOG_FILE=log-spotify.txt mitmproxy --mode reverse:https://api.spotify.com -p 9008 -s spotify.py'", shell=True)
#        subprocess.run("python3 ./spotify_setting.py " + token, shell=True)
    elif name == "youtube":
        run_service("youtube", "youtube.api.Application", "9009")
        subprocess.run(
            "tmux new -d -s youtube-proxy 'LOG_FILE=log-youtube.txt mitmproxy --mode reverse:http://0.0.0.0:8080 -p 9009 -s proxy.py'",
            shell=True)
    elif name == "all":
        subprocess.run(
            "tmux new -d -s fdic-proxy 'LOG_FILE=log-fdic.txt mitmproxy --mode reverse:https://banks.data.fdic.gov -p 9001 -s proxy.py'",
            shell=True)
        subprocess.run("sudo docker stop gn-mongo", shell=True)
        subprocess.run("sudo docker rm gn-mongo", shell=True)
        subprocess.run("sudo docker run --name=gn-mongo --restart=always -p 27018:27017 -d genomenexus/gn-mongo:latest",
                       shell=True)
        time.sleep(30)
        subprocess.run(
            "tmux new -d -s genome-nexus-server '. ./java8.env && java " + cov + "9002.exec" + " -jar ./genome-nexus/web/target/web-0-unknown-version-SNAPSHOT.war'",
            shell=True)
        subprocess.run(
            "tmux new -d -s genome-nexus-proxy 'LOG_FILE=log-genome-nexus.txt mitmproxy --mode reverse:http://0.0.0.0:50110 -p 9002 -s proxy.py'",
            shell=True)
        run_service("emb/em/embedded/rest/languagetool",
                    "em.embedded.org.languagetool.RunServer",
                    "9003")
        subprocess.run(
            "tmux new -d -s language-tool-proxy 'LOG_FILE=log-language-tool.txt mitmproxy --mode reverse:http://0.0.0.0:50111 -p 9003 -s proxy.py'",
            shell=True)
        run_service("emb/em/embedded/rest/ocvn",
                    "em.embedded.org.devgateway.ocvn.RunServer", "9004")
        subprocess.run(
            "tmux new -d -s ocvn-proxy 'LOG_FILE=log-ocvn.txt mitmproxy --mode reverse:http://0.0.0.0:50112 -p 9004 -s proxy.py'",
            shell=True)
        subprocess.run(
            "tmux new -d -s ohsome-proxy 'LOG_FILE=log-ohsome.txt mitmproxy --mode reverse:https://api.ohsome.org -p 9005 -s proxy.py'",
            shell=True)
        subprocess.run(
            "tmux new -d -s omdb-proxy 'LOG_FILE=log-omdb.txt mitmproxy --mode reverse:https://omdbapi.com/ -p 9006 -s omdb.py'",
            shell=True)
        subprocess.run(
            "tmux new -d -s rest-countries-proxy 'LOG_FILE=log-rest-countries.txt mitmproxy --mode reverse:https://restcountries.com -p 9007 -s proxy.py'",
            shell=True)
        with open("spotify.py", "r") as file:
            content = file.read()

        content = content.replace("TOKEN_HERE", token)

        with open("spotify.py", "w") as file:
            file.write(content)
        subprocess.run(
            "tmux new -d -s spotify-proxy 'LOG_FILE=log-spotify.txt mitmproxy --mode reverse:https://api.spotify.com -p 9008 -s spotify.py'",
            shell=True)
#        subprocess.run("python3 ./spotify_setting.py " + token, shell=True)
        run_service("youtube", "youtube.api.Application", "9009")
        subprocess.run(
            "tmux new -d -s youtube-proxy 'LOG_FILE=log-youtube.txt mitmproxy --mode reverse:http://0.0.0.0:8080 -p 9009 -s proxy.py'",
            shell=True)
