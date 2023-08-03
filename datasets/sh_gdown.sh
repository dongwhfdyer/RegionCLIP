#!/bin/bash

# Define an array of links
links=(
  "https://drive.google.com/file/d/19-ADeGcDe_i-TFMkzkI3Kgm6uE9W1vm2/view?usp=drive_link"
  "https://drive.google.com/file/d/1TgmE-VGs1zweRwncqWmz-1UzGUaewNkm/view?usp=drive_link"
  "https://drive.google.com/file/d/1b49BpZQux7W3h8umzPrBSlWdaFBGXqiC/view?usp=drive_link"
  "https://drive.google.com/file/d/1UKFRgTTatOanM7cX4KVwpoMokmaabbDk/view?usp=drive_link"
  "https://drive.google.com/file/d/1-OxYpKvDjG7m9i5GmS9mxwXmd6NuteYz/view?usp=drive_link"
  "https://drive.google.com/file/d/1DR5WbtAawrUhoOdQaN-_ua6xY04eLEKx/view?usp=drive_link"
)

# Iterate over the links
for link in "${links[@]}"
do
  # Extract the file code from the link
  file_code=$(echo "$link" | awk -F'/' '/file\/d/{print $NF}' | awk -F'/' '{print $1}')

  # Download the file using gdown
  gdown "$file_code"
done
