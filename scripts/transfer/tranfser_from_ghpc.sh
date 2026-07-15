REMOTE_HOST="mehrot@console2.ghpc.au.dk"
REMOTE_BASE="/usr/home/qgg/mehrot/worm-species"
LOCAL_BASE="/home/devd/worm-species/"

ssh "$REMOTE_HOST" "cd '$REMOTE_BASE' && find . -type f -print0" |
parallel --ungroup -0 -j 8 --pipe --block 10M \
  rsync -avh --info=progress2 --ignore-existing \
  --from0 --files-from=- \
  "$REMOTE_HOST:$REMOTE_BASE/" \
  "$LOCAL_BASE/"