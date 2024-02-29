#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    python ../../arat.py ../../../../specs/nlp2rest/swagger/omdb.yaml http://localhost:9006
done
