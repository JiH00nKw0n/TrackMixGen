#!/bin/bash
# Stops the highest-numbered running shard of a run when the host's available memory drops below FLOOR_GB (default 30)
# and logs it; the stopped shard resumes later with run_shards.sh. Exits when no shard of the run is running.
#   scripts/mixval/ramguard.sh <run name> [floor GB]
RUN=$1; FLOOR=${2:-30}
while :; do
  names=$(docker ps --format '{{.Names}}' | grep "^mixval_${RUN}_s" | sort)
  [ -z "$names" ] && { echo "$(date -u +%FT%TZ) no shard running, guard exits"; exit 0; }
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$FLOOR" ]; then
    last=$(echo "$names" | tail -1); docker stop $last >/dev/null
    echo "$(date -u +%FT%TZ) RAM guard: $avail GB available < $FLOOR GB, stopped $last"
  fi
  sleep 30
done
