import json
import time
import os
import sys
from dance_discord_utils import DiscordAgentBridge

def main():
    # User Configuration
    WEBHOOK_URL = "..."
    RESULTS_FILE = "ballet_multi_agent_results.json"

    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found! Please run the '08_multi_agents_ballet.ipynb' notebook first to generate results.")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    bridge = DiscordAgentBridge(WEBHOOK_URL)
    
    print("="*40)
    print("🩰 Ballet Multi-Agent Discord Bridge 🩰")
    print("="*40)

    # Handle Command Line Arguments or Interactive Input
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        print(f"Found {len(data)} segments in {RESULTS_FILE}.")
        choice = input("Do you want to push ALL segments or just the FIRST one? (all/first): ").strip().lower()

    if choice == "all":
        bridge.send_status("Starting Ballet Multi-Agent Stream...")
        for i, segment in enumerate(data):
            print(f"[{i+1}/{len(data)}] Pushing segment at {segment['timestamp']}s...")
            bridge.push_segment(segment, mode="ballet")
            time.sleep(3) # Delay between segments to avoid rate limiting or flooding
        bridge.send_status("Ballet Analysis Stream Completed.")
    elif choice == "first":
        print(f"Pushing the first segment...")
        bridge.push_segment(data[0], mode="ballet")
    else:
        print("Invalid choice. Use 'all' or 'first'.")

if __name__ == "__main__":
    main()
