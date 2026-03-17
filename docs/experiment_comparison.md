# 實驗設計對照：SDSD vs Dgrammar/LAVE

## 核心差異：Constraint % 的定義完全不同

| 維度 | **SDSD (我們的實驗)** | **Dgrammar / LAVE** |
|------|------------------------|---------------------|
| **Constraint % 定義** | 輸出 token 中，符合 DFA 合法轉移的比例 | **時間佔比**：constraint 相關運算（grammar check、mask、token select 等）佔總耗時的比例 |
| **計算方式** | `valid_transitions / total_tokens × 100` | `total_constraint_ms / (total_constraint_ms + total_forward_ms) × 100` |
| **典型數值** | 100%（constrained decoding 設計上只輸出合法 token） | 0%～26%（越低表示 constraint 開銷越小） |
| **意義** | 驗證 decoder 是否全程在 grammar 內 | 衡量 constraint 帶來的額外計算成本 |

---

## 實驗範式差異

### SDSD（Constrained Decoding）

- **生成方式**：每個 token 都從 DFA 合法轉移中選出，**保證**輸出在 grammar 內
- **NFE**：每 token 一次 forward（或 speculative 下 T/τ）
- **Constraint %**：100% — 所有 token 都是合法轉移
- **目標**：在保證 grammar 的前提下，降低 NFE、提升 throughput

### Dgrammar / LAVE（Diffusion + Post-hoc Constraint）

- **生成方式**：離散擴散 T 步（如 128 步）→ 並行 denoise → 用 grammar 做 **accept/reject** 或 **remask**
- **Constraint %**：constraint 相關運算佔總時間的比例（8%～26%）
- **目標**：在維持 syntactic 的前提下，降低 constraint 開銷、提升穩定性

---

## 對照表

| 項目 | SDSD | Dgrammar |
|------|------|----------|
| **模型** | Dream-7B / LLaDA-8B | LLaDA-8B-Instruct |
| **Dataset** | json-mode-eval | JSON-Bench (jsonschema) |
| **生成長度** | 64 tokens | 256 tokens |
| **Constraint % 語意** | Token 合法性比例 | 時間開銷比例 |
| **Syntactic** | 可 < 100%（DFA 只限制 token 集合） | 85%～100% |
| **NFE** | T 或 T/τ（forward 次數） | T=128（diffusion 步數） |

---

## 為何 Dgrammar 的 Constraint % 很低？

在 dgrammar 中，Constraint % 代表「constraint 花費的時間佔總時間的百分比」：

- **NO-CD**：0% — 完全不做 constraint
- **IG-CD**：18.1% — 約 18% 時間用在 grammar check、mask、resampling
- **LAVE**：8.5% — 約 76% 的 instance 幾乎不需要 constraint overhead
- **Dgrammar**：11.9% — 透過 async overlap 把 constraint 開銷壓到約 12%

數值越低，表示 constraint 相對 model forward 的額外成本越小。

---

## 為何 SDSD 的 Constraint % 總是 100%？

SDSD 的 Constraint % 是「輸出 token 中，符合 DFA 合法轉移的比例」。

- 所有方法（Baseline、STATIC+DINGO、Herding、SDSD）都是 constrained decoding
- 每個 token 都從 DFA 合法轉移中選出，因此輸出永遠合法
- 因此 Constraint % 恆為 100%

若要與「時間開銷」做對比，應在 SDSD 中額外計算：

- **Constraint overhead %** = `(grammar_check_time + token_select_time) / total_time × 100`

這與 Dgrammar 的 Constraint % 才對應。

---

## 總結

| 指標 | SDSD 定義 | Dgrammar 定義 |
|------|----------|---------------|
| Constraint % | Token 合法性（100%） | 時間開銷佔比（0%～26%） |
| Syntactic | 輸出是否合法 JSON | 同 |
| 實驗設計 | 比較不同 constrained decoding 的 NFE、TTFT | 比較 diffusion + constraint 的 overhead、latency |

兩者實驗設計差異大，Constraint % 的意義不同，無法直接比較數值。
