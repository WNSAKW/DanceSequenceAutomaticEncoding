import requests
import time
import os

class DiscordAgentBridge:
    """
    A utility to stream dance analysis segments directly to Discord.
    """
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.personas = {
            "technical": {
                "username": "Dance Technical AI",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/11865/11865338.png",
                "color": 3447003 
            },
            # Street Dance Personas
            "battle_king": {
                "username": "Freestyle Battle King",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/2930/2930913.png",
                "color": 15158332 
            },
            "old_school": {
                "username": "Old School Legend",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4311/4311966.png",
                "color": 15844367 
            },
            "shonen": {
                "username": "Anime Shonen Dancer",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/2403/2403905.png",
                "color": 10181046 
            },
            # Bajiajiang Personas
            "bjj_xingju": {
                "username": "刑具爺 (Technical Force)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4437/4437890.png",
                "color": 3447003 
            },
            "bjj_ganliu": {
                "username": "甘柳將軍 (Fierce Execution)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4594/4594074.png",
                "color": 15158332 
            },
            "bjj_elder": {
                "username": "廟口長老 (Cultural Wisdom)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/1404/1404072.png",
                "color": 15844367 
            },
            # Ballet Personas
            "ballet_master": {
                "username": "Royal Ballet Master (皇家導師)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/4144/4144697.png",
                "color": 3447003 
            },
            "ballet_prima": {
                "username": "Prima Ballerina (首席舞者)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/268/268635.png",
                "color": 16738740 
            },
            "ballet_avant": {
                "username": "Modern Avant-Garde (當代先鋒)",
                "avatar_url": "https://cdn-icons-png.flaticon.com/512/697/697034.png",
                "color": 10181046 
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

