#!/bin/bash
# Launch Nyaruru via the nwjs SDK nw.exe under Proton, with CDP enabled.
# The stock game ignores Steam launch options (protection), so we point the
# SDK runtime at the game directory instead - the package.json main/index.html
# loads as usual, and --remote-debugging-port reaches Chromium.
#
# Everything machine-specific is overridable via environment variables:
#   NYARURU_GAME_DIR   game install directory (default: Steam library Nyaruru)
#   NYARURU_NWSDK_DIR  nwjs SDK directory containing nw.exe
#                      (default: <repo>/.nwjs-sdk/nwjs-sdk-v0.64.1-win-x64;
#                      symlink your SDK there or point the variable elsewhere)
#   NYARURU_PROTON     proton binary (default: Proton - Experimental)
#   NYARURU_CDP_PORT   CDP port (default: 9222)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STEAM_ROOT="${STEAM_ROOT:-$HOME/.steam/steam}"
GAME_DIR="${NYARURU_GAME_DIR:-$STEAM_ROOT/steamapps/common/Nyaruru}"
NWSDK_DIR="${NYARURU_NWSDK_DIR:-}"
if [ -z "$NWSDK_DIR" ]; then
  # Accept both layouts: .nwjs-sdk/ IS the SDK dir, or contains it.
  if [ -f "$REPO_ROOT/.nwjs-sdk/nw.exe" ]; then
    NWSDK_DIR="$REPO_ROOT/.nwjs-sdk"
  else
    NWSDK_DIR="$REPO_ROOT/.nwjs-sdk/nwjs-sdk-v0.64.1-win-x64"
  fi
fi
PROTON="${NYARURU_PROTON:-$STEAM_ROOT/steamapps/common/Proton - Experimental/proton}"
CDP_PORT="${NYARURU_CDP_PORT:-9222}"

# Proton consumes Windows-style paths; the Z: drive maps to the POSIX root.
to_win() { printf 'Z:%s' "$1"; }

if [ ! -f "$NWSDK_DIR/nw.exe" ]; then
  echo "error: nw.exe not found under $NWSDK_DIR" >&2
  echo "       set NYARURU_NWSDK_DIR or create the .nwjs-sdk symlink" >&2
  exit 1
fi

export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT"
export STEAM_COMPAT_DATA_PATH="$STEAM_ROOT/steamapps/compatdata/1478160"

PROFILE="$REPO_ROOT/.nw-profile"
mkdir -p "$PROFILE"
exec "$PROTON" run "$(to_win "$NWSDK_DIR")/nw.exe" "$(to_win "$GAME_DIR")" \
  --remote-debugging-port="$CDP_PORT" \
  --user-data-dir="$(to_win "$PROFILE")"
