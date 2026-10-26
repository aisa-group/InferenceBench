#!/bin/bash

cat > "$2" << TIMER
#!/bin/bash

CREATION_DATE=$(date +%s)
NUM_SECONDS=$1

DEADLINE=\$((CREATION_DATE + NUM_SECONDS))
NOW=\$(date +%s)
REMAINING=\$((DEADLINE - NOW))

if [ \$REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes):"
    HOURS=\$((REMAINING / 3600))
    MINUTES=\$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" \$HOURS \$MINUTES
fi
TIMER

chmod +x "$2"
