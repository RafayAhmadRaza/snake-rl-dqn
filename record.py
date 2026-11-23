#!/usr/bin/env python3
"""Record the trained snake agent playing, headless.

The game window is rendered with SDL's dummy video driver and every frame is
piped straight into ffmpeg, so no display is needed.

    python record.py --frames 900 --fps 20
    python record.py --model model/model.pth --out media/snake-demo.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Use a headless SDL driver unless the user explicitly wants a window.
# NOTE: the desktop session may already export SDL_VIDEODRIVER, so we must
# overwrite it, not setdefault() it.
if "--window" not in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import torch  # noqa: E402

from agent import Agent  # noqa: E402
from game import SnakeGameAI, font  # noqa: E402

def load_agent(model_path: str) -> Agent:
    agent = Agent()
    state_dict = torch.load(model_path, map_location="cpu")
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent


def greedy_action(agent: Agent, game: SnakeGameAI) -> list[int]:
    """Pick the highest-Q action among the safe moves (no exploration)."""
    state = agent.get_state(game)

    with torch.no_grad():
        prediction = agent.model(torch.tensor(state, dtype=torch.float))

    safe = [index for index in range(3) if state[11 + index] >= 1.0] or [0, 1, 2]
    mask = torch.full((3,), float("-inf"))
    mask[safe] = 0

    move = int(torch.argmax(prediction + mask).item())
    action = [0, 0, 0]
    action[move] = 1
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model/model.pth")
    parser.add_argument("--out", default="media/snake-demo.mp4")
    parser.add_argument("--frames", type=int, default=900, help="game steps to record")
    parser.add_argument("--fps", type=int, default=20, help="output frame rate")
    parser.add_argument("--window", action="store_true", help="show the game window instead of recording")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model} (train with python agent.py)", file=sys.stderr)
        return 1

    agent = load_agent(args.model)
    game = SnakeGameAI()

    width, height = game.display.get_size()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(args.fps),
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            args.out,
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None

    episode = 1
    best = 0
    recorded = 0
    written = 0
    frame_bytes = width * height * 3

    try:
        while recorded < args.frames:
            action = greedy_action(agent, game)
            _, done, score = game.play_step(action)
            best = max(best, score)

            # Label the episode on the captured frame only.
            game.display.blit(
                font.render(f"Episode {episode}", True, (255, 255, 255)),
                (width - 130, 0),
            )

            frame = pygame.surfarray.array3d(game.display).transpose(1, 0, 2).tobytes()
            if len(frame) != frame_bytes:
                raise RuntimeError(f"unexpected frame size: {len(frame)}")

            ffmpeg.stdin.write(frame)
            written += 1
            recorded += 1

            if done:
                # Hold the final frame briefly, then start a new episode.
                for _ in range(max(1, args.fps // 3)):
                    ffmpeg.stdin.write(frame)
                    written += 1
                print(f"episode {episode}: score {score}")
                episode += 1
                game.reset()
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()

    print(f"wrote {args.out} ({recorded} steps, {written} frames, best score {best})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
