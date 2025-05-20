#!/bin/bash

# Check if http_req_duration.csv exists
if [ ! -f "http_req_duration.csv" ]; then
  echo "Error: http_req_duration.csv not found."
  exit 1
fi

# Check if summary.csv exists
if [ ! -f "summary.csv" ]; then
  echo "Error: summary.csv not found."
  exit 1
fi

# Extract the second line (values) from http_req_duration.csv
values=$(sed -n '2p' http_req_duration.csv)

# Check if values are empty
if [ -z "$values" ]; then
  echo "Error: No data found in the second line of http_req_duration.csv."
  exit 1
fi

# Create epoch timestamp
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")

# Append the values to summary.csv
echo "$timestamp,$values" >> summary.csv

echo "Successfully appended data to summary.csv"

exit 0