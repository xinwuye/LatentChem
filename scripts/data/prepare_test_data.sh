#!/bin/bash

# Download test datasets
echo "Downloading test datasets..."
huggingface-cli download --repo-type dataset anonymousssss22321/latentchem --include "ChemCoTBench/**" --local-dir ./test-data/
huggingface-cli download --repo-type dataset anonymousssss22321/latentchem --include "ChemLLMBench/**" --local-dir ./test-data/
huggingface-cli download --repo-type dataset anonymousssss22321/latentchem --include "Mol-Instructions/**" --local-dir ./test-data/
huggingface-cli download --repo-type dataset anonymousssss22321/latentchem --include "ChEBI-20/**" --local-dir ./test-data/

