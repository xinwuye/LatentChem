#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting full latentchem14b pipeline..."

"${script_dir}/train_latentchem14b_stage1.sh"
"${script_dir}/train_latentchem14b_stage2.sh"
"${script_dir}/train_latentchem14b_stage3.sh"
"${script_dir}/train_latentchem14b_stage4.sh"
