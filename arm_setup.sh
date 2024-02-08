DEFAULT_DIR=$(pwd)

# Update Homebrew and Check if Brew is installed
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    brew update # quick check for brew updates
fi

# Install Common Utilities
brew install wget gcc git vim curl tmux mitmproxy

# Install Java (uses Temurin since AdoptOpenJDK is deprecated)
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

# Install EvoMaster
curl -L https://github.com/EMResearch/EvoMaster/releases/download/v1.5.0/evomaster.jar.zip -o evomaster.jar.zip # use curl for redirects
unzip evomaster.jar.zip
rm evomaster.jar.zip

# Prepare and Install RestTestGen
cd $DEFAULT_DIR
if [ -f "./java11_mac.env" ]; then # check if java file exists then executes environment commands
    source ./java11_arm.env
fi
cd tools/resttestgen && ./gradlew install

# Install Docker Desktop application
brew install --cask docker

# Install JaCoCo
cd $DEFAULT_DIR
curl -L https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.7/org.jacoco.agent-0.8.7-runtime.jar -o org.jacoco.agent-0.8.7-runtime.jar
curl -L https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.7/org.jacoco.cli-0.8.7-nodeps.jar -o org.jacoco.cli-0.8.7-nodeps.jar

# Install Evo Bench
cd $DEFAULT_DIR
if [ -f "./java8_mac.env" ]; then # check if java file exists
    source ./java8_arm.env
fi
cd services/emb && mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

# Install Genome-Nexus
cd $DEFAULT_DIR
if [ -f "./java8_mac.env" ]; then
    source ./java8_arm.env
fi
cd services/genome-nexus && mvn clean install -DskipTests

# Install YouTube Mock service
cd $DEFAULT_DIR
if [ -f "./java11_mac.env" ]; then
    source ./java11_arm.env
fi
cd services/youtube && mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt
