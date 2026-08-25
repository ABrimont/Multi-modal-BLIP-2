#!/bin/bash

set -e

echo "🐍 Creating and activating the virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "📥 Cloning the microsoft/unilm repository..."
cd lavis/models/blip2_models
if [ ! -d unilm ]; then
    git clone https://github.com/microsoft/unilm.git
else
    echo "unilm already present, skipping clone."
fi
cd ../../../

echo "🛠️ Applying patches to the BEATs files..."
BEATS_FILE="lavis/models/blip2_models/unilm/beats/BEATs.py"
BACKBONE_FILE="lavis/models/blip2_models/unilm/beats/backbone.py"

# Patch 1: fix relative import in BEATs.py
grep -q "^from .backbone import" "$BEATS_FILE" || \
    sed -i 's/^from backbone import/from .backbone import/' "$BEATS_FILE"

# Patch 2: add noise injection in the preprocess function of BEATs.py
grep -q "noise = torch.randn_like" "$BEATS_FILE" || \
    sed -i '/fbank = (fbank - fbank_mean) \/ (2 \* fbank_std)/a \        noise = torch.randn_like(fbank) * 1e-6\n        fbank = fbank + noise' "$BEATS_FILE"

# Patch 3: fix relative import in backbone.py
grep -q "^from .modules import" "$BACKBONE_FILE" || \
    sed -i 's/^from modules import/from .modules import/' "$BACKBONE_FILE"

echo "✅ Installation completed successfully!"
echo "👉 To get started, run: cd Multi-modal-BLIP-2 && source venv/bin/activate"
