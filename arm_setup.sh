DEFAULT_DIR=$(pwd)

# Update Homebrew and Check if Brew is installed
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    brew update # quick check for brew updates
fi

# Install Common Utilities
brew install wget gcc git vim curl tmux mitmproxy

# Install Java (uses Temurin since AdoptOpenJDK is deprecated) and jenv to manage jdk
brew tap homebrew/cask-versions
brew install --cask temurin8
brew install --cask temurin11
brew install maven gradle

# Using Python3 Installation on Mac, install Schemathesis
python3 -m venv venv
cd $DEFAULT_DIR
source venv/bin/activate
pip install requests schemathesis
if [ -f "requirements.txt" ]; then # check for requirements.txt
    pip install -r requirements.txt
fi

#Install EvoMaster
wget https://github.com/EMResearch/EvoMaster/releases/download/v1.5.0/evomaster.jar.zip
unzip evomaster.jar.zip
rm evomaster.jar.zip

#Install RestTestGen
cd $DEFAULT_DIR
source ./env/java11_arm.env && cd tools/resttestgen && ./gradlew install

#docker installa
brew install --cask docker

#Install Jacoco
cd $DEFAULT_DIR
wget https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.7/org.jacoco.agent-0.8.7-runtime.jar
wget https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.7/org.jacoco.cli-0.8.7-nodeps.jar

#Install Evo Bench
cd $DEFAULT_DIR
source ./env/java8_arm.env && cd services/emb && mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

#Install Genome-Nexus
cd $DEFAULT_DIR
source ./env/java8_arm.env && cd services/genome-nexus && mvn clean install -DskipTests

#Install YouTube Mock service
cd $DEFAULT_DIR
source ./env/java11_arm.env && cd services/youtube && mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

