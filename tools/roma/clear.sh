#! /bin/bash

pkill -9 -i vllm
sleep 1s
pkill -9 -i python
sleep 1s
ray stop
echo "All processes killed"
