#!/usr/bin/env bash
# serenity_run.sh — launch the Serenity trainer UI.
#
# Usage:
#   ./serenity_run.sh            # launch the UI (uses the existing binary)
#   ./serenity_run.sh --build    # rebuild the UI first (pixi run trainer-ui-build)
#
# The runner binaries in target/ carry their own rpaths (cshim/cudnn/.pixi);
# the UI itself only needs the MojoUI floor library on LD_LIBRARY_PATH.

set -euo pipefail
cd "$(dirname "$0")"

UI_BIN=target/serenity_trainer_ui
MOJOUI_DIR=/home/alex/MojoUI

if [[ "${1:-}" == "--build" ]]; then
    pixi run trainer-ui-build
fi

if [[ ! -x "$UI_BIN" ]]; then
    echo "serenity_run.sh: $UI_BIN missing — building it (pixi run trainer-ui-build)" >&2
    pixi run trainer-ui-build
fi

export LD_LIBRARY_PATH="$MOJOUI_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$UI_BIN"
