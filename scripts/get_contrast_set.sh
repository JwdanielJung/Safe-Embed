#!/usr/env/bin bash

OPENAI_API_KEY=YOUR_KEY_HERE

python -m Safety_Contrast.get_contrast_data $OPENAI_API_KEY
python -m Safety_Contrast.parse_contrast_data