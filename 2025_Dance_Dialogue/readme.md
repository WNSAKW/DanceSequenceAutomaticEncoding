# **Dance Dialogue（2025ver.）**
- 舞蹈種類：台灣原住民祭儀舞、芭蕾舞…
- LLM：Gemini、ChatGPT
- 所有需載入之舞蹈資料集、影片、Json檔等等，皆需至 Google Drive：[Dance Dataset](https://drive.google.com/drive/folders/1sVp0VV5cnPds2-x8nupfWWji3lVYc6a8?usp=sharing) 下載

*****

### **STEP 0_資料準備**
#### 資料夾結構：
- ./weights/
- ./data/raw_csv/
  - 放置骨架座標資料（台灣原住民祭儀舞、芭蕾舞…）
- ./data/segments/
- ./data/mp4/
  1. twa.mp4           ← 臺灣原住民舞蹈
  2. ballet.mp4        ← 芭蕾
  3. ...               ← ...
  4. ...               ← ...

*****

### **STEP 1_執行 Complete Pipeline**
- 01_Complete_Pipeline.ipynb

*****

### **STEP 2_資料獲取**
#### 資料夾結構：
- ./weights/
  1. lstm_encoder_best.pth
  2. lstm_encoder_epoch1_loss3.7189.pth
  3. ...
  4. lstm_encoder_epoch80_loss2.4993.pth

- ./data/raw_csv/
  - 放置骨架座標資料（台灣原住民祭儀舞、芭蕾舞…）

- ./data/segments/
  1. mean.npy       ← 臺灣原住民舞蹈
  2. seg_00001.pt   ← 芭蕾
  3. ...            ← ...
  4. seg_00671.pt   ← ...
  5. std.npy        ← ...

- ./data/mp4/
  1. twa.mp4        ← 臺灣原住民舞蹈
  2. ballet.mp4     ← 芭蕾
  3. ...            ← ...
  4. ...            ← ...

*****

### **STEP 3_執行 Dance Dialogue**
- 02_gemini_demo.ipynb
- 02_gpt_demo.ipynb
- 02_gpt-ballet_v2026.ipynb
  - v2026：需載入查表檔案（可擴充）：ballet_cultural_library.json
- !!! 記得更換 API KEY !!!
- 獲取對話紀錄檔案：Ballet_ceremony_dialogue_00000_00000.json

*****

### **STEP 4_執行 六大指標語義對齊**
- 03_all_metrics_with_gpt_turns.ipynb
- 獲取整合檔案：lookup_rules_ballet.json、turn_analysis_mapping_ballet.csv
