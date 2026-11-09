#!/usr/bin/env bash
# voidtrackr — deploy to the LAN host.
#
# Usage:
#   ./deploy.sh              # deploy to the registered host
#   ./deploy.sh <host>       # deploy to a specific host (key auth)
#
# What it does:
#   1. builds the site (and refuses to ship without the model weights)
#   2. rsyncs the built files into the webroot
#   3. installs the nginx vhost and this site's security headers
#   4. disables the template's own site, so there is one site per container
#   5. validates the nginx config and reloads
#   6. smoke tests the result
#
# The site is static: no service to start, no runtime to install. Only the
# files and the nginx vhost change. The heavy lifting happens in the
# browser, so the box itself needs nothing but nginx.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The umbrella repo holds the shared deploy key. This project is standalone:
# it is a SIBLING of the umbrella on the Desktop (~/Desktop/voidtrackr and
# ~/Desktop/TheVoidCo), not a child of it. So the umbrella is found by
# walking up the tree, and then by looking at the siblings of this folder.
# Set REPO_ROOT to override both.
find_repo_root() {
  local dir="$1"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/ops/keys/thevoidco-lxc-id_ed25519" ]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname -- "$dir")"
  done

  # A sibling of this folder — the usual layout for a standalone project.
  local parent sibling
  parent="$(dirname -- "$1")"
  for sibling in "$parent"/*/; do
    [ -d "$sibling" ] || continue
    if [ -f "$sibling/ops/keys/thevoidco-lxc-id_ed25519" ]; then
      echo "${sibling%/}"
      return 0
    fi
  done

  return 1
}

REPO_ROOT="$(find_repo_root "$SCRIPT_DIR" || true)"
if [ -z "${REPO_ROOT:-}" ]; then
  echo "error: could not find the umbrella repo (ops/keys/thevoidco-lxc-id_ed25519)." >&2
  echo "       It normally sits next to this folder on the Desktop." >&2
  echo "       Point REPO_ROOT at it, or set VOIDTRACKR_KEY to a key of your own." >&2
  exit 1
fi

# The host address is reserved in the router at provisioning time and
# registered in the umbrella's ops/vm.sh. Override it here if it changes.
#
# 192.0.2.0 is the RFC 5737 documentation address. It is deliberately NOT a
# real address: the deploy refuses to run against it, so a half-provisioned
# host can never be overwritten by accident.
HOST="${1:-${VOIDTRACKR_HOST:-192.0.2.0}}"
USER="${VOIDTRACKR_USER:-root}"

# There is ONE deploy key for TheVoidCo's infrastructure
# (ops/keys/thevoidco-lxc-id_ed25519). A container cloned from the template
# inherits it in /root/.ssh/authorized_keys, so this is the same key every
# other Void* host uses — not a per-project key.
KEY="${VOIDTRACKR_KEY:-$REPO_ROOT/ops/keys/thevoidco-lxc-id_ed25519}"
WEBROOT="/var/www/voidtrackr"

# The token in the checked-in vhost that stands in for the real address.
PLACEHOLDER_IP="192.0.2.0"

if [ "$HOST" = "$PLACEHOLDER_IP" ]; then
  echo "error: no host address is configured." >&2
  echo "" >&2
  echo "  The address is reserved in the router by the container's MAC," >&2
  echo "  so it is only known once the container is provisioned." >&2
  echo "" >&2
  echo "  Pass it explicitly:   ./deploy.sh 192.168.100.xxx" >&2
  echo "  or set it for a run:  VOIDTRACKR_HOST=192.168.100.xxx ./deploy.sh" >&2
  echo "" >&2
  echo "  The placeholder is ${PLACEHOLDER_IP} (RFC 5737 documentation range)." >&2
  echo "  Deploying to it would be meaningless; refusing." >&2
  exit 1
fi

case "$HOST" in
  *[!0-9.]*|"")
    echo "error: the host address '${HOST}' is not an IP address." >&2
    exit 1
    ;;
esac

if [ ! -f "$KEY" ]; then
  echo "error: the ssh key ${KEY} is missing." >&2
  exit 1
fi

SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ConnectTimeout=8 -o LogLevel=ERROR)
SCP=(scp -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ConnectTimeout=8 -o LogLevel=ERROR)
RSYNC_SSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
RSYNC=(rsync -az --delete --chmod=D755,F644
       -e "$RSYNC_SSH")

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# 1. Build.
#
#    The build copies the model weights into dist/, so the site can fetch them
#    from its own origin. Without them the browser has no model to load and
#    the app is a dead page — so this is checked, not assumed.
say "Building the site"

# Which model is the default comes from the registry, so the deploy never has
# to be told what the default is and cannot drift away from it.
DEFAULT_MODEL="$(sed -n "s/.*DEFAULT_MODEL: ModelId = '\([^']*\)'.*/\1/p" \
  "$SCRIPT_DIR/src/core/registry.ts")"
if [ -z "$DEFAULT_MODEL" ]; then
  echo "error: could not read DEFAULT_MODEL from src/core/registry.ts." >&2
  exit 1
fi

case "$DEFAULT_MODEL" in
  basic-pitch)
    DEFAULT_WEIGHTS="$SCRIPT_DIR/public/model/model.json"
    DEFAULT_COMMAND="npm run model:basic-pitch"
    ;;
  okarina)
    DEFAULT_WEIGHTS="$SCRIPT_DIR/models/onnx/okarina_m.onnx"
    DEFAULT_COMMAND="npm run model:okarina"
    ;;
  *)
    DEFAULT_WEIGHTS="$SCRIPT_DIR/models/onnx/$DEFAULT_MODEL"
    DEFAULT_COMMAND="npm run model:$DEFAULT_MODEL"
    ;;
esac

# The default model's weights are not optional: the app loads that model the
# moment a file is dropped in, so a site without them is a page that does
# nothing.
if [ ! -e "$DEFAULT_WEIGHTS" ]; then
  echo "error: the weights of the default model ($DEFAULT_MODEL) are missing." >&2
  echo "       Run: $DEFAULT_COMMAND" >&2
  exit 1
fi

(cd "$SCRIPT_DIR" && npm run build)

# Okarina's weights sit at the root of model-onnx/; the others have a folder
# of their own. Each model is checked by the path the browser actually asks
# for, so a missing model is reported rather than silently broken.
for model in basic-pitch okarina tsumugi yourmt3 piano; do
  case "$model" in
    basic-pitch) present="$SCRIPT_DIR/dist/model/model.json" ;;
    okarina)     present="$SCRIPT_DIR/dist/model-onnx/okarina_m.onnx" ;;
    *)           present="$SCRIPT_DIR/dist/model-onnx/$model" ;;
  esac
  if [ ! -e "$present" ]; then
    echo "" >&2
    echo "warning: the $model weights are not present, so that model will not" >&2
    echo "         work in the browser. Run: npm run model:$model" >&2
  fi
done

# 2. Ship the site files.
say "Uploading site files to ${HOST}:${WEBROOT}"
"${SSH[@]}" "$USER@$HOST" "mkdir -p '$WEBROOT' /var/log/webapps"
"${RSYNC[@]}" --rsync-path="mkdir -p '$WEBROOT' && rsync" \
  "$SCRIPT_DIR/dist/" "$USER@$HOST:$WEBROOT/"

# 3. Install the nginx vhost and this site's headers.
#    The source files live here, so they are copied across first — `install`
#    runs on the remote box and cannot see local paths.
say "Installing nginx configuration"
"${SSH[@]}" "$USER@$HOST" "install -d -m 755 /etc/nginx/snippets /etc/nginx/sites-available"

"${SCP[@]}" "$SCRIPT_DIR/ops/nginx/voidtrackr-headers.conf" \
  "$USER@$HOST:/tmp/voidtrackr-headers.conf"
"${SCP[@]}" "$SCRIPT_DIR/ops/nginx/voidtrackr.conf" \
  "$USER@$HOST:/tmp/voidtrackr.conf"

"${SSH[@]}" "$USER@$HOST" \
  "sed 's|${PLACEHOLDER_IP//./\\.}|${HOST}|g' /tmp/voidtrackr.conf > /tmp/voidtrackr.conf.new && \
   install -m 644 /tmp/voidtrackr-headers.conf /etc/nginx/snippets/voidtrackr-headers.conf && \
   install -m 644 /tmp/voidtrackr.conf.new /etc/nginx/sites-available/voidtrackr.conf && \
   ln -sf /etc/nginx/sites-available/voidtrackr.conf /etc/nginx/sites-enabled/voidtrackr.conf && \
   rm -f /tmp/voidtrackr-headers.conf /tmp/voidtrackr.conf /tmp/voidtrackr.conf.new"

# 4. One container, one site.
#
#    The template the container was cloned from carries its own base landing
#    page, and that vhost is `listen 80 default_server` — exactly like
#    voidtrackr's. Two default servers on one port is a hard nginx error:
#
#      [emerg] a duplicate default server for 0.0.0.0:80
#
#    so nginx would refuse to start with the template's site still enabled.
#    Only the *link* is removed, never the file in sites-available, so the
#    template's page is one `ln -sf` away from coming back.
say "Disabling the template's own site"
"${SSH[@]}" "$USER@$HOST" \
  "find /etc/nginx/sites-enabled -mindepth 1 -maxdepth 1 ! -name 'voidtrackr.conf' -print -delete"

# 5. Validate and reload. nginx -t runs first, so a broken config never
#    takes the running site down.
say "Validating and reloading nginx"
"${SSH[@]}" "$USER@$HOST" "nginx -t && systemctl reload nginx"

# 6. Smoke test.
say "Smoke test"
for path in / /404.html /robots.txt /model/model.json /model-onnx/okarina_m.onnx \
            /model-onnx/tsumugi/tsumugi_other_v1_5.pos32i_i32.onnx \
            /model-onnx/yourmt3/encoder.onnx \
            /model-onnx/piano/piano_kong.onnx \
            /ort/ort-wasm-simd-threaded.wasm; do
  code="$("${SSH[@]}" "$USER@$HOST" \
    "curl -s -o /dev/null -w '%{http_code}' -H 'Host: ${HOST}' http://127.0.0.1$path")"
  printf '  %-58s %s\n' "$path" "$code"
done

say "Live at http://${HOST}/"
