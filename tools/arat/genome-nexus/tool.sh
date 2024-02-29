#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    python ../../arat.py ../../../../specs/original/swagger/genome-nexus.yaml http://localhost:9002
done
