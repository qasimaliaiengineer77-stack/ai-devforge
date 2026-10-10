# AI Software Company — Lighthouse runner image (Step 13)
# Architecture reference: Section 36 (Lighthouse Docker)
#
# Rebuilt to fix "waiting for dynamic debugging port" errors.
# Chrome needs a writable /dev/shm (large enough), a proper user-data-dir,
# and a stable sandbox configuration inside containers.

FROM node:22-slim

# 1. Install Chromium, all necessary shared libraries, and Xvfb (for headful mode if ever needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        ca-certificates \
        fonts-liberation \
        libnss3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libgbm1 \
        libgtk-3-0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libxkbcommon0 \
        libasound2 \
        libpangocairo-1.0-0 \
        libx11-xcb1 \
        libxcursor1 \
        libxi6 \
        libxtst6 \
        xvfb \
        dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# 2. Point Lighthouse to the Chromium binary
ENV CHROME_PATH=/usr/bin/chromium
ENV LIGHTHOUSE_CHROMIUM_PATH=/usr/bin/chromium

# 3. Install a recent, stable Lighthouse version
RUN npm install -g lighthouse@12.2.1

# 4. Create a non-root user and a writable profile directory
RUN useradd -m -u 1001 lighthouse
RUN mkdir -p /home/lighthouse/chrome-profile \
    && chown -R lighthouse:lighthouse /home/lighthouse

USER lighthouse
WORKDIR /home/lighthouse

# 5. Add a tiny entrypoint wrapper to ensure the profile dir is fresh and writable.
#    This prevents stale DevTools port logs from confusing chrome-launcher.
RUN echo '#!/bin/sh\n\
set -e\n\
PROFILE=/home/lighthouse/chrome-profile\n\
rm -rf "$PROFILE"\n\
mkdir -p "$PROFILE"\n\
exec lighthouse "$@" --chrome-flags="--headless=new --no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage --disable-gpu --remote-debugging-port=9222 --user-data-dir=$PROFILE" --max-wait-for-load=45000' \
    > /home/lighthouse/entrypoint.sh \
    && chmod +x /home/lighthouse/entrypoint.sh

ENTRYPOINT ["/home/lighthouse/entrypoint.sh"]