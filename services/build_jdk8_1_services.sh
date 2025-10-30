#!/bin/bash

# Build script for JDK 8_1 services
# This script builds all services in the aratrl-service/jdk8_1 directory

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
JDK8_1_DIR="$PROJECT_ROOT/aratrl-service/jdk8_1"

echo "======================================"
echo "Building JDK 8_1 Services"
echo "======================================"
echo ""

# Check if directory exists
if [ ! -d "$JDK8_1_DIR" ]; then
    echo "Error: JDK 8_1 directory not found at $JDK8_1_DIR"
    exit 1
fi

# Navigate to JDK 8_1 directory
cd "$JDK8_1_DIR"

# Source Java 8 environment
if [ -f "$SCRIPT_DIR/java8_mac.env" ]; then
    echo "Loading Java 8 environment..."
    source "$SCRIPT_DIR/java8_mac.env"
elif [ -f "$SCRIPT_DIR/java8.env" ]; then
    echo "Loading Java 8 environment..."
    source "$SCRIPT_DIR/java8.env"
else
    echo "Warning: Java 8 environment file not found. Proceeding with system default Java."
fi

# Display Java version
echo "Using Java version:"
java -version

echo ""
echo "Building all JDK 8_1 services..."
echo "This may take several minutes..."
echo ""

# Build with Maven
mvn clean install -DskipTests

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Build completed successfully!"
    echo "======================================"
    echo ""
    echo "Built services:"
    echo "  - features-service (cs/rest/original/features-service)"
    echo "  - ncs (cs/rest/artificial/ncs)"
    echo "  - scs (cs/rest/artificial/scs)"
    echo ""
    echo "You can now run these services using:"
    echo "  python services/run_service_mac.py features-service no_token"
    echo "  python services/run_service_mac.py ncs no_token"
    echo "  python services/run_service_mac.py scs no_token"
else
    echo ""
    echo "======================================"
    echo "Build failed!"
    echo "======================================"
    exit 1
fi
