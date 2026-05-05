import nbformat as nbf
import os

NOTEBOOK_PATH = "08_multi_agents_streetdance.ipynb"

# 1. Load the notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 2. Define the new Setup Cell
discord_setup_code = """# ========================
# Discord Multi-Agent Bridge Setup
# ========================
from dance_discord_utils import DiscordAgentBridge

DISCORD_WEBHOOK_URL = "..."
discord_bridge = DiscordAgentBridge(DISCORD_WEBHOOK_URL)

print("✅ Discord 即時串流橋接器準備就緒！")"""

discord_setup_cell = nbf.v4.new_code_cell(discord_setup_code)

# 3. Find where to insert (after cell 0 usually)
nb.cells.insert(1, discord_setup_cell)

# 4. Modify the Main Loop cell
# We'll look for a cell containing "ai_sees(row, cultural_library)"
for cell in nb.cells:
    if "ai_sees(row, cultural_library)" in cell.source:
        print("Found main loop cell, modifying...")
        
        # Replace the loop logic to include Discord calls
        old_source = cell.source
        
        # Insert "Analyzing..." status
        new_source = old_source.replace(
            "for i, row in segments_df.iterrows():",
            "discord_bridge.send_status(\"🚀 開始全片即時分析及點評...\")\n\nfor i, row in segments_df.iterrows():"
        )
        
        # Insert "Analyzing segment..." before AI generation
        new_source = new_source.replace(
            "sees_content, keywords = ai_sees(row, cultural_library)",
            "discord_bridge.send_status(f\"正在分析片段 {i+1}/{len(segments_df)} (T={row['timestamp_sec']}s)......\")\n    sees_content, keywords = ai_sees(row, cultural_library)"
        )
        
        # Insert "push_segment" after segment creation
        new_source = new_source.replace(
            "results.append(res)",
            "results.append(res)\n    \n    # 4. 即時推送到 Discord\n    discord_bridge.push_segment(res)"
        )
        
        # Add "Analysis Complete" at end of loop (after loop finished)
        new_source += "\n\ndiscord_bridge.send_status(\"✨ 全部分析完成！檔案已自動儲存。\")"
        
        cell.source = new_source
        break

# 5. Save the notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully injected real-time Discord logic into {NOTEBOOK_PATH}")
