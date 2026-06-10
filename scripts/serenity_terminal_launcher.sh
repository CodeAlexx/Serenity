#!/usr/bin/env bash
set -euo pipefail

root="/home/alex/serenity-trainer"
terminal_script="$root/target/serenity_terminal_run.sh"

if [ "$#" -lt 4 ]; then
    echo "usage: $0 backend_label runner pidfile logfile [runner args...]" >&2
    exit 2
fi

backend_label="$1"
runner="$2"
pidfile="$3"
logfile="$4"
shift 4

mkdir -p "$root/target" "$(dirname "$logfile")"
rm -f "$pidfile" "$logfile"

cat > "$terminal_script" <<'EOS'
#!/usr/bin/env bash
set -u

root="$1"
pidfile="$2"
logfile="$3"
backend_label="$4"
runner="$5"
shift 5

cd "$root"
rm -f "$pidfile"

echo "Serenity ${backend_label} trainer"
echo "Working dir: $root"
echo "Runner: $runner"
echo "Log: $logfile"
echo

"$runner" "$@" > >(tee "$logfile") 2>&1 &
child="$!"
echo "$child" > "$pidfile"

stop_child() {
    if [ -n "${child:-}" ]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    rm -f "$pidfile"
    exit 143
}

trap stop_child TERM INT
wait "$child"
status="$?"
rm -f "$pidfile"

echo
echo "Serenity ${backend_label} trainer exited with status $status"
read -r -p "Press Enter to close terminal..."
exit "$status"
EOS
chmod +x "$terminal_script"

env -i \
    HOME=/home/alex \
    USER=alex \
    LOGNAME=alex \
    SHELL=/bin/bash \
    LANG="${LANG:-en_US.UTF-8}" \
    DISPLAY="${DISPLAY:-:1}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}" \
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}" \
    XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-ubuntu:GNOME}" \
    DESKTOP_SESSION="${DESKTOP_SESSION:-ubuntu}" \
    GDMSESSION="${GDMSESSION:-ubuntu}" \
    PATH="/home/alex/.local/bin:/usr/local/cuda-12.6/bin:/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:/usr/local/cuda-11.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin" \
    LD_LIBRARY_PATH="/home/alex/.local/lib:/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/usr/local/cuda-11.8/lib64" \
    /usr/bin/gnome-terminal \
        --title="Serenity ${backend_label} Trainer" \
        -- bash "$terminal_script" \
            "$root" \
            "$pidfile" \
            "$logfile" \
            "$backend_label" \
            "$runner" \
            "$@"
