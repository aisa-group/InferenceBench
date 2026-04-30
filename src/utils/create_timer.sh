#!/bin/bash

NUM_HOURS=$1
TARGET_PATH=$2
CREATION_DATE=$(date +%s)
# Fractional NUM_HOURS (e.g. 0.17) would break bash $((...)) arithmetic in the
# generated timer, so convert to whole seconds here via awk.
NUM_SECONDS="$(awk -v h="${NUM_HOURS}" 'BEGIN{printf "%d", h*3600}')"

cat > "$TARGET_PATH" << TIMER
#!/bin/bash

NUM_HOURS=${NUM_HOURS}
CREATION_DATE=${CREATION_DATE}
NUM_SECONDS=${NUM_SECONDS}

DEADLINE=\$((CREATION_DATE + NUM_SECONDS))
NOW=\$(date +%s)
REMAINING=\$((DEADLINE - NOW))

if [ \$REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=\$((REMAINING / 3600))
    MINUTES=\$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" \$HOURS \$MINUTES
fi
TIMER

chmod +x "$TARGET_PATH"