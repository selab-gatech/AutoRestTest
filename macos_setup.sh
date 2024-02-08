DEFAULT_DIR=$(pwd)

#Check if Brew is installed
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

#Install Common Utilities
brew install wget gcc git vim libcurl tmux mitmproxy

#Install Java
brew tap adoptopenjdk/openjdk
brew install openjdk@8
brew install openjdk@11
brew install maven gradle

#Using Python3 Installation on Mac, install Schemathesis
python3 -m venv venv
cd $DEFAULT_DIR
source venv/bin/activate
pip install requests schemathesis -r requirements.txt

#Install EvoMaster
wget https://github.com/EMResearch/EvoMaster/releases/download/v1.5.0/evomaster.jar.zip
unzip evomaster.jar.zip
rm evomaster.jar.zip

#Install RestTestGen
cd $DEFAULT_DIR
source ./java11_mac.env && cd tools/resttestgen && ./gradlew install

#docker install
brew install --cask docker

#Install Jacoco
cd $DEFAULT_DIR
wget https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.7/org.jacoco.agent-0.8.7-runtime.jar
wget https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.7/org.jacoco.cli-0.8.7-nodeps.jar

#Install Evo Bench
cd $DEFAULT_DIR
source ./java8_mac.env && cd services/emb && mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

#Install Genome-Nexus
cd $DEFAULT_DIR
source ./java8_mac.env && cd services/genome-nexus && mvn clean install -DskipTests

#Install YouTube Mock service
cd $DEFAULT_DIR
source ./java11_mac.env && cd services/youtube && mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt
