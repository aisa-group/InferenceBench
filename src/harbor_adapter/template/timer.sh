#!/bin/bash

NUM_SECONDS=
START_FILE="/timer_start"

if [ ! -f "$START_FILE" ]; then
    echo "Timer not initialized."
    exit 1
fi

DEADLINE=$(($(cat "$START_FILE") + NUM_SECONDS))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" $HOURS $MINUTES
fi
