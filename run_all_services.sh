#!/bin/bash

# Define list of services
declare -a services=(
  "fdic"
  "genome-nexus"
#   "language-tool"
#   "ocvn"
  "ohsome"
  "omdb"
  "rest-countries"
  "spotify"
  "youtube"
)

# Loop over the services array
for service in "${services[@]}"
do
  echo "Starting service: $service"
  
  # Navigate to the services directory and start the service
  cd services
  if python3 run_service_arm.py "$service" no_token; then
    echo "Service $service started successfully."
  else
    echo "Failed to start service $service."
    cd ..
    continue  # Skip the rest of the loop and proceed with the next service
  fi
  cd ..

  # Give some time for the service to start up properly
  sleep 5  # Adjust this time as needed

  echo "Running request generator for: $service"
  
  # Navigate to the src directory and run the request generator
  cd src
  if python3 request_generator.py "$service"; then
    echo "Request generator for $service completed successfully."
  else
    echo "Request generator for $service failed."
    # No need to continue here as it's the end of the loop body
  fi
  cd ..

  # Give some time for the request generator to finish
  sleep 10  # Adjust this time as needed


done

echo "All services have been processed."
