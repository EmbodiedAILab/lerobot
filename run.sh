#!/bin/bash

policy_type="$1"
policy_url="$2"

# Validate parameters
if [[ -z "$policy_type" || -z "$policy_url" ]]; then
    echo "Usage: $0 <policy_type> <policy_url>"
    echo "Example: $0 act obs://bucket/path/to/prefix"
    exit 1
fi

# Print the received parameters
echo "Received policy_type: $policy_type"
echo "Received policy_url : $policy_url"

python lerobot/scripts/serving_policy.py --policy.type=$policy_type --policy.path=$policy_url
