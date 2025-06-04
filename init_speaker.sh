#!/bin/bash

if [ -z "$1" ]; then
    pactl set-card-profile bluez_card.50_C2_75_A9_1F_EE handsfree_head_unit
    pactl list cards | grep 'Active Profile'
elif [ "$1" == "a2dp" ]; then
    pactl set-card-profile bluez_card.50_C2_75_A9_1F_EE a2dp_sink
    pactl list cards | grep 'Active Profile'
elif [ "$1" == "show" ]; then
    pactl list cards | grep 'Active Profile'
else
    echo "Usage: $0 [a2dp|show]"
fi