REMOTE_HOST="devd@login.genome.au.dk"
REMOTE_BASE="/home/devd/worm-species/source"
LOCAL_BASE="/home/devd/worm-species/"

ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && find . -type f -print0" |
parallel --ungroup -0 -j 4 --pipe --block 10M \
  rsync -avh --info=progress2 --ignore-existing \
  --bwlimit=6500K \
  --from0 --files-from=- \
  "$REMOTE_HOST:$REMOTE_BASE/" \
  "$LOCAL_BASE/"