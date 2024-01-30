#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    schemathesis run ../../../specs/enhanced/fdic.json --hypothesis-database=none --checks all --stateful=links --max-response-time=30000 --validate-schema False --base-url http://localhost:9001/api
    schemathesis run ../../../specs/enhanced/fdic.json --hypothesis-database=none --stateful=links --max-response-time=30000 --validate-schema False --base-url http://localhost:9001/api
done
