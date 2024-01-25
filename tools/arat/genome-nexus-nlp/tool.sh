#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    python ../../../arat.py ../../../specs/eswagger/genome-nexus.yaml http://localhost:9002
done
