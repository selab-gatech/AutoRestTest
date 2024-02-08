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
brew install jenv

# Initialize jenv
export PATH="$HOME/.jenv/bin:$PATH"
eval "$(jenv init -)"
jenv enable-plugin export
# Setup jenv for managing Java JDKs (eval needed)
# PATH assignment is just to keep changes across sessions

# Add Temurin JDKs to jenv
# ENSURE THAT THESE FILES EXIST WITH YOUR JDK INSTALLATION (temurin should auto-install here but I'm not certain)
jenv add /Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
jenv add /Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home

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

# Install Docker Desktop application
brew install --cask docker

# Install JaCoCo
cd $DEFAULT_DIR
curl -L https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.7/org.jacoco.agent-0.8.7-runtime.jar -o org.jacoco.agent-0.8.7-runtime.jar
curl -L https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.7/org.jacoco.cli-0.8.7-nodeps.jar -o org.jacoco.cli-0.8.7-nodeps.jar

# MAKE SURE JENV VERSIONS ALIGN HERE (use jenv versions)

# Prepare and Install RestTestGen (Java 11)
cd $DEFAULT_DIR/tools/resttestgen
jenv local 11.0.22
./gradlew install

# Install Genome-Nexus (Java 8)
cd $DEFAULT_DIR/services/genome-nexus
jenv local 1.8.0.402
mvn clean install -DskipTests

# Install YouTube Mock service (Java 11)
cd $DEFAULT_DIR/services/youtube
jenv local 11.0.22
mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt

# Install Evo Bench (Java 8)
cd $DEFAULT_DIR/services/emb
jenv local 1.8.0.402
mvn clean install -DskipTests && mvn dependency:build-classpath -Dmdep.outputFile=cp.txt
