# main.py
#
# Simple CLI loop for LocalWhisper Step 1.
# Run:  python main.py
#       python main.py --verbose    ← shows the ReAct reasoning steps

import argparse
import sys

from agent import run, run_verbose

def main():
    parser = argparse.ArgumentParser(description="LocalWhisper CLI")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the agent's reasoning steps (tool calls and observations)."
    )
    args = parser.parse_args()

    respond = run_verbose if args.verbose else run

    print("Ivy 🌿  Assistente de lazer urbano")
    print("Digite sua mensagem. Ctrl+C para sair.\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAté logo! 🌿")
            sys.exit(0)

        if not user_input:
            continue

        print("Ivy: ", end="", flush=True)
        reply = respond(user_input, history)
        print(reply)
        print()

        # Keep history for multi-turn context
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
