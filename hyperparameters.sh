#!/bin/bash

# epochs=("100" "125" "150")
epochs=("25" "40" "50")
# hidden1=("1700" "2000")
# hidden1=("12" "28")
hidden1=("512" "1024")
hidden2=("2048")
# hidden2=("12" "28")
# hidden2=("28" "196" "392")

for hid1 in "${hidden1[@]}"; do
    for hid2 in "${hidden2[@]}"; do
        echo "l1: $hid1, l2: $hid2, epoch: ${epochs[0]}"
        (./main ${epochs[0]} $hid1 $hid2 -1  1 && echo "n1: $hid1, n2: $hid2, epochs: ${epochs[0]}") >> hyperparemeters.txt &

        echo "l1: $hid1, l2: $hid2, epoch: ${epochs[1]}"
        (./main ${epochs[1]} $hid1 $hid2 -1  3 && echo "n1: $hid1, n2: $hid2, epochs: ${epochs[1]}") >> hyperparemeters.txt &

        echo "l1: $hid1, l2: $hid2, epoch: ${epochs[2]}"
        (./main ${epochs[2]} $hid1 $hid2 -1  5 && echo "n1: $hid1, n2: $hid2, epochs: ${epochs[2]}") >> hyperparemeters.txt
        tail -6 hyperparemeters.txt
    done
done

