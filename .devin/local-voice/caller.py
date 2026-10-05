#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

r"""Place one call to the local voice bot from a browser with a WAV as its mic.

Opens the Pipecat Playground, clicks Connect, lets the call run, saves
screenshots, prints the conversation shown in the UI, then hangs up.

Run (``headed`` is useful when the screen is being recorded)::

    uv run --with playwright python .devin/local-voice/caller.py \\
        --audio /tmp/caller.wav --seconds 45 --headed
"""

import argparse
import asyncio
import os
import shutil

from playwright.async_api import async_playwright


def _chrome() -> str | None:
    return os.getenv("CHROME_PATH") or shutil.which("google-chrome") or shutil.which("chromium")


async def main():
    """Run the call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="WAV file used as the microphone.")
    parser.add_argument("--url", default="http://localhost:7860/client")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--screenshots", default="/tmp/local-voice-call")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.screenshots, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=_chrome(),
            headless=not args.headed,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={os.path.abspath(args.audio)}",
                "--autoplay-policy=no-user-gesture-required",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context(
            permissions=["microphone"], viewport={"width": 1280, "height": 720}
        )
        page = await context.new_page()
        await page.goto(args.url, wait_until="networkidle")
        await page.locator("button", has_text="Connect").first.click(timeout=15000)

        elapsed = 0.0
        while elapsed < args.seconds:
            await asyncio.sleep(5)
            elapsed += 5
            await page.screenshot(path=f"{args.screenshots}/call_{int(elapsed):03d}s.png")

        print(await page.locator("body").inner_text(timeout=5000))
        await page.locator("button", has_text="Disconnect").first.click(timeout=5000)
        await asyncio.sleep(2)
        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
