#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    python ../../../arat.py ../../../specs/eswagger/spotify.yaml http://localhost:9008/v1
done
