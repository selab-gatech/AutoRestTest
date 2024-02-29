#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    python ../../arat.py ../../../../specs/nlp2rest/swagger/ohsome.yaml http://localhost:9005/v1
done
