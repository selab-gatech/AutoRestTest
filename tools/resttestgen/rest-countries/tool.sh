#! /bin/bash
end=$((SECONDS+3600))

while [ $SECONDS -lt $end ]; do
    ../build/install/resttestgen-framework/bin/resttestgen-framework
done
