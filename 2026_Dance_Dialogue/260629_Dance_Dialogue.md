## 目標
將各種舞蹈（芭蕾、八家將、街舞等）的骨骼運動資料，轉化為量化指標 + 語意描述 + 文化情感解讀，最終建立可查表、可自動標註的舞蹈語意系統。
目前核心檔案與分工（建議優先閱讀順序）

### 01~03 系列（基礎流程）
- 01_Complete_Pipeline.ipynb：完整特徵提取 → LSTM 嵌入訓練 → 正規化
- 02_gpt-ballet_v20262.ipynb：即時 GPT 對話系統（AI sees / AI says）
- 03_all_metrics_with_gpt_turns.ipynb：指標分析 + GPT 回應抽取情感/動作/文化關鍵詞 + 建立查表規則

### 04~06 系列（指標到語意轉換）
- 04_video_charts.ipynb：視覺化影片與指標圖表
- 05_calculate_ranges.ipynb：計算各指標的正常範圍與門檻
- 06_metrics_to_semantic_text.ipynb：將量化指標直接轉成文字描述

### 07 系列（骨骼到語意）
- 07_metrics_to_semantic_with_skeleton_ballet.ipynb
- 07_metrics_to_semantic_with_skeleton_baijiang.ipynb
- 07_metrics_to_semantic_with_skeleton_streetdance.ipynb
→ 針對不同舞種，將骨骼特徵 + 量化指標轉成語意文本

### 08 系列（多代理系統）
- 08_multi_agents_ballet.ipynb
- 08_multi_agents_baijiang.ipynb
- 08_multi_agents_streetdance.ipynb
→ 目前最先進版本，使用多代理架構（Multi-Agents）處理不同舞種的語意生成，功能最完整。

## 現況
- 已涵蓋芭蕾、八家將、街舞三種舞風。
- 從單純 GPT 呼叫，進化到 Metrics → Semantic → Multi-Agents 的完整管道。
- 有[語意文化資料庫](https://github.com/WNSAKW/DanceSequenceAutomaticEncoding/tree/main/2026_Dance_Dialogue)，作為知識底座（例：ballet_cultural_library.json）。
