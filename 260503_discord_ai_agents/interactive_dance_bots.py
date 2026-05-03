import discord
from discord.ext import commands
import asyncio
import os
import json
import functools
from openai import OpenAI
from dotenv import load_dotenv
from dance_discord_utils import DiscordAgentBridge, AnalysisState

load_dotenv()
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
bridge = DiscordAgentBridge(webhook_url="") 
state_manager = AnalysisState()
bot_instances = {}
target_channel_id = None 

class DanceBot(commands.Bot):
    def __init__(self, persona_key, token, *args, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True # 這裡開了，網頁端後台也要開！
        super().__init__(command_prefix="!", intents=intents, *args, **kwargs)
        self.persona_key = persona_key
        self.persona = bridge.personas.get(persona_key)
        
    async def on_ready(self):
        print(f"✅ {self.persona['username']} 已上線")
    async def on_message(self, message):
        global target_channel_id
        if message.author.bot: return # 忽略所有機器人訊息 (包含自己)
        
        # 偵測廣播頻道
        if target_channel_id is None:
            target_channel_id = message.channel.id
            print(f"🎯 已偵測到廣播頻道 ID: {target_channel_id}")
        if self.user.mentioned_in(message):
            print(f"💬 {self.persona['username']} 被 {message.author} 標記了")
            async with message.channel.typing():
                try:
                    response = await self.generate_response(message)
                    await message.reply(response)
                    
                    state = state_manager.mark_interaction()
                    if state:
                        count = state["interaction_count"]
                        if count < 5:
                            await message.channel.send(f"💬 互動計數: {count}/5 (滿 5 次後繼續分析)")
                        else:
                            await message.channel.send(f"✅ 互動已滿，系統即將進入下一個片段...")
                except Exception as e:
                    print(f"❌ 回應失敗: {e}")
                    await message.channel.send(f"⚠️ {self.persona['username']} 訊號微弱，請稍後再試...")
    async def generate_response(self, message):
        state = state_manager.load() or {}
        context = state.get("pending_broadcast") or {}
        
        # 使用 run_in_executor 避免同步 API 阻塞 Event Loop
        loop = asyncio.get_event_loop()
        prompt = self.persona["system_prompt"] + f"\n\n背景數據：{json.dumps(context, ensure_ascii=False)}"
        user_input = message.content.replace(f"<@{self.user.id}>", "").strip()
        
        partial_func = functools.partial(
            client_ai.chat.completions.create,
            model=MODEL,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_input}],
            temperature=0.7
        )
        
        res = await loop.run_in_executor(None, partial_func)
        return res.choices[0].message.content

async def monitor_loop():
    print("👀 監控中... (請確保已在 Discord 標記過 Bot 以確定頻道)")
    while True:
        try:
            state = state_manager.load()
            if state and state.get("pending_broadcast") and target_channel_id:
                data = state["pending_broadcast"]
                print(f"📡 偵測到新數據，由 5 個角色進行廣播...")
                # 依序讓 Bot 說話
                sequence = ["system", "technical", "battle_king", "old_school", "shonen"]
                tech_text = "無"
                
                for p_key in sequence:
                    if p_key in bot_instances:
                        bot = bot_instances[p_key]
                        channel = bot.get_channel(target_channel_id)
                        if channel:
                            if p_key == "system":
                                await channel.send(f"🚀 **Analysis System**: 偵測到新數據 (T={data.get('timestamp')}s)，開始分析...")
                            elif p_key == "technical":
                                prompt = bridge.personas["technical"]["system_prompt"] + f"\n\n分析數據：{json.dumps(data, ensure_ascii=False)}"
                                res = client_ai.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": prompt}])
                                tech_text = res.choices[0].message.content
                                await channel.send(embed=discord.Embed(title="📊 Technical Analysis", description=tech_text, color=0x3498db))
                            else:
                                prompt = bridge.personas[p_key]["system_prompt"] + f"\n\n參考技術 AI 觀點：\n{tech_text}"
                                res = client_ai.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": prompt}])
                                await channel.send(res.choices[0].message.content)
                            await asyncio.sleep(1.5)
                
                state["pending_broadcast"] = None
                state_manager.save(state)
        except Exception as e:
            print(f"⚠️ 監控異常: {e}")
        await asyncio.sleep(3)

async def main():
    tokens = {"system": os.getenv("DISCORD_TOKEN_SYSTEM"), "technical": os.getenv("DISCORD_TOKEN_TECHNICAL"), "battle_king": os.getenv("DISCORD_TOKEN_BATTLE_KING"), "old_school": os.getenv("DISCORD_TOKEN_OLD_SCHOOL"), "shonen": os.getenv("DISCORD_TOKEN_SHONEN")}
    tasks = []
    for k, v in tokens.items():
        if v and "你的" not in v:
            bot = DanceBot(persona_key=k, token=v); bot_instances[k] = bot; tasks.append(bot.start(v))
    await asyncio.gather(*tasks, monitor_loop())

if __name__ == "__main__":
    asyncio.run(main())
