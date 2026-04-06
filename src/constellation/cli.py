import argparse
import asyncio
from pathlib import Path
import select
import signal
import sys
import termios
import threading
import tty
from typing import Any

from constellation.core.agent import Agent
from constellation.core.session import VoiceSession
from constellation.loader import load_agent_config
from constellation.logger import LOGGER


def _setup_keyboard_listener(session: VoiceSession, shutdown_event: asyncio.Event) -> threading.Thread:
    def keyboard_thread() -> None:
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not shutdown_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char.lower() == "m":
                        is_muted = session.toggle_mute()
                        status = "MUTED" if is_muted else "UNMUTED"
                        print(f"\n[{status}]")
        except Exception:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    thread = threading.Thread(target=keyboard_thread, daemon=True)
    thread.start()
    return thread


async def run_session(config_path: str) -> None:
    config = load_agent_config(config_path)
    agent = Agent(config)
    session = VoiceSession(agent)

    shutdown_event = asyncio.Event()

    def handle_signal(sig: int, frame: Any) -> None:
        LOGGER.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    old_settings = None
    try:
        await session.start()

        print("\n" + "=" * 50)
        print("Constellation Voice Agent")
        print("=" * 50)
        print("Listening... (Press Ctrl+C to stop)")
        print("Press 'm' to toggle mute")
        print("=" * 50 + "\n")

        old_settings = termios.tcgetattr(sys.stdin)
        _setup_keyboard_listener(session, shutdown_event)

        session.run_initiation()

        while session.is_running() and not shutdown_event.is_set():
            await asyncio.sleep(0.1)

    except Exception as e:
        LOGGER.error(f"Session error: {e}", exc_info=True)
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        await session.stop()
        print("\nSession ended.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constellation Voice Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Run a voice agent session")
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to agent config YAML file",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    import logging

    if hasattr(args, "verbose") and args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("deepgram").setLevel(logging.WARNING)

    if args.command == "run":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        asyncio.run(run_session(str(config_path)))


if __name__ == "__main__":
    main()
