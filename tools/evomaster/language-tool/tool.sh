#! /bin/bash
end=$((SECONDS+3600))
current_path=$(pwd)
path_to_file="$current_path/../../../specs/openapi_json"

while [ $SECONDS -lt $end ]; do
    java -jar ../../../evomaster.jar --blackBox true --bbSwaggerUrl file://$path_to_file/language-tool.json --maxTime 1h --outputFormat JAVA_JUNIT_4
done
