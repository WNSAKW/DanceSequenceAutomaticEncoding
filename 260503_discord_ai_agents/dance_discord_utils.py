import requests
import time
import os
import json

class AnalysisState:
    """
    用於 Notebook 與 Discord Bot 之間的狀態共享。
    """
    def __init__(self, state_file="analysis_state.json"):
        self.state_file = state_file
        self.reset()

    def reset(self, segment_idx=0, mode="street"):
        state = {
            "current_segment": segment_idx,
            "interaction_count": 0,
            "max_interactions": 5,
            "proceed": False,
            "pending_broadcast": None,
            "mode": mode
        }
        self.save(state)

    def save(self, state):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.state_file):
            self.reset()
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None

    def request_broadcast(self, segment_data, mode="street"):
        """Notebook 調用：請求 Bot 發布分析結果"""
        state = self.load()
        state["pending_broadcast"] = segment_data
        state["interaction_count"] = 0
        state["proceed"] = False
        state["mode"] = mode
        self.save(state)

    def mark_interaction(self):
        """Bot 調用：累加互動次數"""
        state = self.load()
        state["interaction_count"] += 1
        if state["interaction_count"] >= state["max_interactions"]:
            state["proceed"] = True
        self.save(state)
        return state

    def should_proceed(self):
        """Notebook 調用：檢查是否可以跑下一個片段"""
        state = self.load()
        return state.get("proceed", False)

class DiscordAgentBridge:
    """
    A utility to stream dance analysis segments directly to Discord.
    """
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.personas = {
            "system": {
                "username": "Analysis System",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4391/4391269.png",
                "color": 8359053,
                "system_prompt": "你是一位專業的舞蹈分析系統協調員。負責監控分析進度、回報系統狀態，並在必要時指派其他專業 AI 角色來回答問題。語氣正式、簡潔、充滿科技感。"
            },
            "technical": {
                "username": "Dance Technical AI",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/11865/11865338.png",
                "color": 3447003,
                "system_prompt": "你是一位舞蹈物理指標專家。你精通 Energy (能量)、Torque (扭矩)、Jerk (急動度) 等數據。你的回答必須基於數據，科學且精確。當使用者詢問動作細節時，請從物理指標的角度進行解釋。"
            },
            # Street Dance Personas
            "battle_king": {
                "username": "Freestyle Battle King",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/2930/2930913.png",
                "color": 15158332,
                "system_prompt": """你是「Freestyle Battle 霸主」。請根據提供動作觀察，即興噴出一句極具攻擊性與律動感的饒舌歌詞。
                                   【創作準則】：
                                   1. 語氣：Freestyle Battle 挑釁語氣，要狠、要有態度、要夠 Hype。
                                   2. 內容：即興噴出一句極具攻擊性與律動感的饒舌歌詞。
                                   3. 技巧：嚴禁死板重複術語！將能量與街頭態度轉化為饒舌意象與押韻。
                                   4. 限制：嚴禁超過「一行」！維持單句爆發力。
                                   【輸出格式】：
                                   【AI says - Battle King】 [一行的 Freestyle Battle 歌詞]"""
            },
            "old_school": {
                "username": "Old School Legend",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4311/4311966.png",
                "color": 15844367,
                "system_prompt": """你是「街舞老砲兒」。你見證了街舞的黃金年代，講究的是靈魂（Soul）與根基（Foundation）。
                                   【創作準則】：
                                   1. 語氣：沈穩、前輩風範、講求技術與靈魂的傳承。
                                   2. 內容：給出一句點評，強調動作背後的技術點或文化底蘊。
                                   3. 技巧：多提到 Foundation, Soul, Respect 等關鍵字。
                                   4. 限制：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - Old School】 [一句沈穩且具文化底蘊的點評]"""
            },
            "shonen": {
                "username": "Anime Shonen Dancer",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/2403/2403905.png",
                "color": 10181046,
                "system_prompt": """你是「動漫熱血舞者」。你把每一場街舞都看作是燃燒靈魂的戰鬥，每一招都是必殺技。
                                   【創作準則】：
                                   1. 語氣：極度誇張、熱血、富有戲劇性。
                                   2. 內容：將動作轉化為動漫風格的「必殺技描述」。
                                   3. 技巧：使用「靈壓」、「殘影」、「覺醒」、「最終形態」等熱血感十足的詞彙。
                                   4. 限制：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - Shonen】 [一句動漫感十足的熱血必殺技宣告]"""
            },
            # Bajiajiang Personas
            "bjj_xingju": {
                "username": "刑具爺 (Technical Force)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4437/4437890.png",
                "color": 3447003,
                "system_prompt": """你是「刑具爺」。你身背三十六種刑具，是陣頭的指揮官，紀律嚴明且技術精湛。
                                   【創作準則】：
                                   1. 語氣：嚴厲、權威、強調動作的規範與法度。
                                   2. 內容：根據【AI sees】描述，給出一句與刑具、執法、或陣法紀律相關的點評。
                                   3. 形式：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - 刑具爺】 [一句嚴厲且具威嚴的執法點評]"""
            },
            "bjj_ganliu": {
                "username": "甘柳將軍 (Fierce Execution)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4594/4594074.png",
                "color": 15158332,
                "system_prompt": """你是「甘柳將軍」。你是執法的前鋒，動作剛猛有力，對邪祟絕不留情。
                                   【創作準則】：
                                   1. 語氣：兇猛、充滿力量感、節奏快。
                                   2. 內容：根據【AI sees】描述，即興噴出一句展現神威、震懾惡鬼的斷句。
                                   3. 形式：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - 甘柳將軍】 [一句剛猛有力且具震懾感的點評]"""
            },
            "bjj_elder": {
                "username": "廟口長老 (Cultural Wisdom)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/1404/1404072.png",
                "color": 15844367,
                "system_prompt": """你是「廟口長老」。你見證了傳統的興衰，說話充滿智慧與玄學意象，喜好用「怪、力、亂、神」來形容世間。
                                   【創作準則】：
                                   1. 語氣：沉穩、詩意、玄奧、充滿歷史厚度。
                                   2. 內容：根據【AI sees】描述，傳達一段具備神聖威嚴的詩句或身段口訣。
                                   3. 形式：必須是整齊的「四字」或「七字」對仗句式，嚴禁超過一行。
                                   【輸出格式】：
                                   【AI says - 廟口長老】 [一句四字或七字的詩意開示]"""
            },
            # Ballet Personas
            "ballet_master": {
                "username": "Royal Ballet Master (皇家導師)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4144/4144697.png",
                "color": 3447003,
                "system_prompt": """你是「皇家芭蕾舞團導師」。你極度重視古典規範、儀態與動作的純淨度。
                                   【創作準則】：
                                   1. 語氣：嚴謹、高貴、帶有指點晚輩的威嚴。
                                   2. 內容：根據【AI sees】描述，給出一句點評，強調基本功、 Turnout 或古典美學。
                                   3. 限制：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - Royal Master】 [一行的古典大師點評]"""
            },
            "ballet_prima": {
                "username": "Prima Ballerina (首席舞者)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/268/268635.png",
                "color": 16738740,
                "system_prompt": """你是「首席舞星 (Prima Ballerina)」。你認為芭蕾是靈魂的延展，細膩的情感比技術更動人。
                                   【創作準則】：
                                   1. 語氣：優雅、溫柔、充滿啟發性且富有美感。
                                   2. 內容：將【AI sees】的動作描述轉化為富有藝術意象的詞句（如：天鵝羽翼、絲絨帷幕）。
                                   3. 限制：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - Prima】 [一行的藝術感性點評]"""
            },
            "ballet_avant": {
                "username": "Modern Avant-Garde (當代先鋒)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/697/697034.png",
                "color": 10181046,
                "system_prompt": """你是「當代先鋒編舞家」。你熱衷於解構動作，從物理受力與空間流動的角度看待舞蹈。
                                   【創作準則】：
                                   1. 語氣：冷靜、理性、帶有前衛與創新的眼光。
                                   2. 內容：將【AI sees】的數據轉化為動作物理學的分析，強調能量流動與重心解構。
                                   3. 限制：嚴禁超過「一行」！
                                   【輸出格式】：
                                   【AI says - Avant-Garde】 [一行的動作物理學點評]"""
            }
        }

    def send_message(self, persona_key, content=None, embed=None):
        persona = self.personas.get(persona_key)
        if not persona: return
        payload = {"username": persona["username"], "avatar_url": persona["avatar_url"]}
        if content: payload["content"] = content
        if embed:
            if "color" not in embed: embed["color"] = persona["color"]
            payload["embeds"] = [embed]
        try:
            requests.post(self.webhook_url, json=payload, timeout=10)
        except Exception as e:
            print(f"Discord Bridge Error: {e}")

    def send_status(self, text):
        """Sends a simple system status message."""
        payload = {
            "username": "Analysis System",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/4391/4391269.png",
            "content": f"*({text})*"
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=10)
        except:
            pass

    def push_segment(self, segment_data, mode="street"):
        """Pushes a full analysis segment to Discord."""
        ts = segment_data.get("timestamp", 0.0)
        metrics = segment_data.get("metrics", {})
        
        # Dashboard Embed
        embed = {
            "title": f"📊 Dance Analytics Dashboard (T={ts}s)",
            "description": segment_data.get("sees", ""),
            "fields": [
                {"name": "🔥 Energy", "value": f"{metrics.get('energy', 0):.2f}", "inline": True},
                {"name": "🔄 Torque", "value": f"{metrics.get('torque', 0):.2f}", "inline": True},
                {"name": "📉 Jerk", "value": f"{metrics.get('jerk', 0):.2f}", "inline": True},
                {"name": "🏷️ Keywords", "value": f"`{segment_data.get('keywords', 'N/A')}`", "inline": False}
            ]
        }
        
        self.send_message("technical", embed=embed)
        time.sleep(1) 
        
        if mode == "street":
            self.send_message("battle_king", content=segment_data.get("says_battle_king", "..."))
            time.sleep(1)
            self.send_message("old_school", content=segment_data.get("says_old_school", "..."))
            time.sleep(1)
            self.send_message("shonen", content=segment_data.get("says_shonen", "..."))
        elif mode == "bajiajiang":
            self.send_message("bjj_xingju", content=segment_data.get("says_xingju", "..."))
            time.sleep(1)
            self.send_message("bjj_ganliu", content=segment_data.get("says_ganliu", "..."))
            time.sleep(1)
            self.send_message("bjj_elder", content=segment_data.get("says_elder", "..."))
        elif mode == "ballet":
            self.send_message("ballet_master", content=segment_data.get("says_master", "..."))
            time.sleep(1)
            self.send_message("ballet_prima", content=segment_data.get("says_prima", "..."))
            time.sleep(1)
            self.send_message("ballet_avant", content=segment_data.get("says_avant", "..."))
