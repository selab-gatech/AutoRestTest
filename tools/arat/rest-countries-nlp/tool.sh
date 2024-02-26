#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    python ../../arat.py ../../../../specs/nlp2rest/swagger/rest-countries.yaml http://localhost:9007
done
