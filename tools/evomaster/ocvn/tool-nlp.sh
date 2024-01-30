#! /bin/bash
end=$((SECONDS+3600))
current_path=$(pwd)
path_to_file="$current_path/../../../specs/enhanced"

while [ $SECONDS -lt $end ]; do
    java -jar ../../../evomaster.jar --blackBox true --bbSwaggerUrl file://$path_to_file/ocvn3.yaml --maxTime 1h --outputFormat JAVA_JUNIT_4
done
