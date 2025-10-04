#!/bin/bash

# output wav file time duration $1
ffmpeg -i "$1" 2>&1 | grep "Duration" | awk '{print $2}' | tr -d ,

