好問題！讓我把這兩個問題結合起來討論。

---

## 問題重述

```
{"name": * * *}

我們需要知道：
1. 每條 parse path 用了幾個 MASK（remaining_masks 追蹤）
2. 不同 branch 走到同一個 state 時，怎麼有效 merge（避免重複計算）
```

---

## 核心想法：擴充 Earley State + DP

### 原本的 Earley State
```
(rule, dot_position, origin)

例如：
  (value → • string, 0)
  (value → • number, 0)
```

### 擴充成帶 mask count 的 State
```
(rule, dot_position, origin, masks_used)

例如：
  (value → • string, 0, masks_used=0)
  (string → " • chars ", 1, masks_used=1)  ← 用了一個 MASK
  (string → " chars • ", 2, masks_used=2)  ← 用了兩個 MASK
```

### 用 DP Table 來記錄
```
dp[i][j] = 「從位置 i 到位置 j 的 substring，
             所有可能的 parse state 集合」

因為 MASK 是 wildcard，
dp[i][j] 裡的每個 state 都帶著 masks_used 資訊
```

這樣不同 branch 走到同一個 (rule, dot_position, origin) 但用了不同數量 MASK 的情況，就可以在 DP table 裡分開追蹤。

---

## Trie 可以用在哪裡

Trie 在這裡有兩個潛在用途：

### 用途一：Token Vocabulary 的結構化搜尋
```
問題：
  每個 MASK 位置理論上可以是任何 token，
  但很多 token 在 grammar 上是等價的
  （例如所有的 identifier 對 parser 來說都一樣）

做法：
  把 vocab 建成 character-level trie，
  在 parser advance 的時候，
  不是對每個 token 獨立處理，
  而是沿著 trie 走，
  讓有共同前綴的 token 共享計算
```

### 用途二：Parser State 的共享前綴
```
不同的 parse path 如果有共同的前綴部分，
可以用 trie 結構來共享這段計算，
避免重複 parse 同樣的 token sequence
```

---

## 完整的資料結構設計

```
結合 Earley + DP + Trie：

Chart[i] = {
  state → Set<masks_used>
}

意思是：
  在位置 i，某個 parser state 可以用 k 個 MASK 達到，
  k 的所有可能值都記錄下來

Transition：
  遇到 concrete token → 只有完全匹配的 state 可以 advance
  遇到 MASK → 所有 state 都 advance，masks_used + 1
```

### 例子走一遍
```
{"name": * * *}
           ↑ ↑ ↑
           位置 1,2,3 是 MASK，總共 3 個

Chart[1] after 第一個 *：
  (string → " • chars ", masks_used={1})
  (number → • digits,   masks_used={1})
  (array  → [ • items,  masks_used={1})

Chart[2] after 第二個 *：
  (string → " chars • ", masks_used={2})  ← string 可能結束
  (string → " • chars ", masks_used={2})  ← string 繼續
  (number → digits •,    masks_used={2})  ← number 可能結束
  ...

Chart[3] after 第三個 *：
  檢查哪些 state 的 masks_used = 3（用完所有 MASK）
  且可以接著看到 } 然後 complete
```

---

## 關鍵的 DP Pruning

這裡可以做一個很重要的優化：

```
在每個 Chart[i]，
如果某個 state 的 masks_used > remaining_masks，
直接剪掉，不需要繼續追蹤

因為已經用超過可用的 MASK 數量了
```

反過來也可以做 **forward pruning**：
```
如果某個 parse path 到目前為止，
剩下的 token 數量不夠讓 grammar 合法結束，
也可以提前剪掉
```

---

## 複雜度分析

```
原本 Earley：
  O(n³ × |G|)
  n = sequence length, |G| = grammar size

加上 masks_used 追蹤：
  O(n³ × |G| × k)
  k = 最大 MASK 數量

加上 DP pruning：
  實際上 k 通常很小（每個 block 只有幾個 MASK），
  所以實務上接近 O(n³ × |G|)

加上 Trie：
  Vocab lookup 從 O(|V|) 降到 O(|token_length|)
  對 JsonSchemaBench 這種 grammar 複雜的情況特別有幫助
```

---

## 總結

```
Remaining masks 追蹤：
  → 擴充 Earley state 加上 masks_used
  → DP table 記錄所有可能的 masks_used 集合

避免重複計算：
  → Earley 本來的 chart merging 繼續用
  → 加上 masks_used 之後，相同 state 不同 masks_used 分開記錄

Trie 的角色：
  → Vocab 層面的共享前綴計算
  → 特別對 token 種類很多的 grammar 有幫助

最大的優勢：
  → 完全 deterministic，不需要 random sampling
  → 理論上 0% false negative
  → DP pruning 讓實際複雜度可控
```

好，讓我一步一步來。

---

## Pseudo Code

```python
def wildcard_earley_verify(incomplete_prefix, grammar, total_masks):
    """
    incomplete_prefix: list of tokens, some are MASK
    grammar: CFG rules
    total_masks: 總共有幾個 MASK
    returns: True if exists valid completion, False otherwise
    """
    
    # ===== 初始化 =====
    # Chart[i] = dict of {(rule, dot_pos, origin) -> Set[masks_used]}
    # 意思是：在位置 i，這個 parser state 可以用幾個 MASK 到達
    chart = [defaultdict(set) for _ in range(len(incomplete_prefix) + 1)]
    
    # 加入 start rule
    for rule in grammar.rules_for(grammar.start_symbol):
        chart[0][(rule, 0, 0)].add(0)  # masks_used = 0
    
    # ===== 主迴圈 =====
    for i, token in enumerate(incomplete_prefix):
        
        # 先做 prediction 和 completion（不消耗 token）
        changed = True
        while changed:
            changed = False
            
            for (rule, dot, origin), masks_set in list(chart[i].items()):
                
                # --- Prediction ---
                # 如果 dot 指向一個 non-terminal，展開它
                if dot < len(rule.rhs) and is_nonterminal(rule.rhs[dot]):
                    next_symbol = rule.rhs[dot]
                    for new_rule in grammar.rules_for(next_symbol):
                        new_state = (new_rule, 0, i)
                        before = len(chart[i][new_state])
                        chart[i][new_state] |= masks_set  # 繼承 masks_used
                        if len(chart[i][new_state]) > before:
                            changed = True
                
                # --- Completion ---
                # 如果 dot 到達 rule 結尾，找所有等待這個 symbol 的 state
                if dot == len(rule.rhs):
                    completed_symbol = rule.lhs
                    for (prev_rule, prev_dot, prev_origin), prev_masks in list(chart[origin].items()):
                        if (prev_dot < len(prev_rule.rhs) and 
                            prev_rule.rhs[prev_dot] == completed_symbol):
                            new_state = (prev_rule, prev_dot + 1, prev_origin)
                            new_masks = {m1 + m2 
                                        for m1 in prev_masks 
                                        for m2 in masks_set
                                        if m1 + m2 <= total_masks}  # pruning
                            before = len(chart[i][new_state])
                            chart[i][new_state] |= new_masks
                            if len(chart[i][new_state]) > before:
                                changed = True
        
        # --- Scanning ---
        # 消耗當前 token，advance 到 chart[i+1]
        if token == MASK:
            # Wildcard：所有 state 都可以 advance，masks_used + 1
            for (rule, dot, origin), masks_set in chart[i].items():
                if dot < len(rule.rhs) and is_terminal(rule.rhs[dot]):
                    new_masks = {m + 1 for m in masks_set 
                                if m + 1 <= total_masks}  # pruning
                    if new_masks:
                        chart[i+1][(rule, dot+1, origin)] |= new_masks
        else:
            # Concrete token：只有匹配的 state 可以 advance
            for (rule, dot, origin), masks_set in chart[i].items():
                if dot < len(rule.rhs) and rule.rhs[dot] == token:
                    chart[i+1][(rule, dot+1, origin)] |= masks_set
    
    # ===== 檢查結果 =====
    # 在最後位置，找 start rule 且用完所有 MASK 的 complete state
    n = len(incomplete_prefix)
    for (rule, dot, origin), masks_set in chart[n].items():
        if (rule.lhs == grammar.start_symbol and 
            dot == len(rule.rhs) and
            origin == 0 and
            total_masks in masks_set):  # 用完所有 MASK
            return True
    
    # 或者：prefix 是 extendable 的（不需要完整 parse 完）
    for (rule, dot, origin), masks_set in chart[n].items():
        if (total_masks in masks_set and
            is_extendable(chart[n], grammar)):  # 還可以繼續
            return True
            
    return False
```

---

## 這個演算法會遇到的問題

### 問題一：Completion 的 masks_used 組合爆炸

```python
new_masks = {m1 + m2 
            for m1 in prev_masks    # 可能很多值
            for m2 in masks_set}    # 可能很多值
```

兩個 set 做笛卡爾積加法，如果兩邊都有很多不同的 masks_used 值，這個操作很慢。

但實際上 masks_used 的範圍是 `[0, total_masks]`，所以最多只有 `total_masks + 1` 個值，通常很小（block size = 32，但一個 prefix 裡的 MASK 通常遠少於 32）。

### 問題二：JsonSchemaBench 的 Grammar 很大

動態生成的 JSON Schema grammar 可能有數百條 rule，chart 的 state 數量會很多，每個 position 的 prediction/completion 迴圈會跑很久。

### 問題三：Tokenization 和 Grammar 的不對齊

```
Grammar 通常是 character-level 或 word-level，
但 dLLM 的 token 是 BPE token

例如：
  grammar 說 string 是 " [chars] "
  但 BPE 可能把 "hello" tokenize 成一個 token，
  不是 ", h, e, l, l, o, " 七個 token
```

這個問題在 LAVE 裡也存在，但 deterministic 方法更難繞過。

---

## 實驗計畫：一步一步驗證

### Stage 1：最小可行版本（Week 1-2）

**目標**：先不管效率，確認 correctness

```python
實驗設定：
  - 用最簡單的 grammar（例如：balanced parentheses）
  - 手動構造幾個 incomplete prefix
  - 驗證 wildcard_earley_verify 的輸出是否正確

測試案例：
  grammar: S → (S) | ε
  
  案例1: "( * )"         → True  (MASK = ε 或 (S))
  案例2: "( * * * )"     → True
  案例3: "( ) * ( ) *"   → False (用完 MASK 還是不合法)
  案例4: "* * )"         → False (右括號比左括號多)

成功標準：
  所有手動案例輸出正確
```

### Stage 2：和 LAVE random sampling 比較（Week 3-4）

**目標**：確認 deterministic 方法的 false negative 更低

```python
實驗設定：
  - 用 LAVE 論文的 HumanEval-CPP benchmark
  - 對同一個 incomplete prefix，
    分別跑 wildcard_earley_verify 和 LAVE N=10 sampling
  - 記錄兩者的結果差異

關鍵指標：
  disagreement_rate = 
    cases where LAVE says False but wildcard says True
    / total cases

  這就是 LAVE 的 false negative rate
  
成功標準：
  disagreement_rate > 0（證明 deterministic 方法確實更準確）
  且 disagreement_rate 接近 LAVE 論文說的 ~2%
```

### Stage 3：效率測試（Week 5-6）

**目標**：確認速度可接受

```python
實驗設定：
  - 測量 wildcard_earley_verify 的執行時間
  - 對比 LAVE N=10 sampling 的執行時間
  - 分別在簡單 grammar（JSON）和複雜 grammar（C++）上測試

關鍵指標：
  time_ratio = wildcard_time / lave_sampling_time
  
成功標準：
  time_ratio < 5x  （5倍以內可接受，因為準確率更高）
  
如果太慢：
  → 加入 DP pruning
  → 限制 masks_used 的追蹤範圍
  → 用 trie 加速 vocab lookup
```

### Stage 4：接入 dLLM（Week 7-8）

**目標**：替換 LAVE 的 lookahead sampling，端對端測試

```python
實驗設定：
  - 用 Dream-7B 或 LLaDA-8B
  - 把 LAVE 的 verify 函數換成 wildcard_earley_verify
  - 在 CPP-Bench 和 JSON-Bench 上跑完整實驗

關鍵指標：
  syntactic@1 是否接近 100%？
  functional@1 是否比 LAVE 更好？
  average inference time 是否可接受？
  
對比基準：
  LAVE N=10:  syntactic@1 ≈ 96-99%
  我們的目標: syntactic@1 ≈ 100%
```

### Stage 5：JsonSchemaBench（Week 9-10）

**目標**：針對最難的 benchmark 測試

```python
額外需要處理：
  - 動態 grammar compilation
  - Semantic constraints（minimum, maximum, pattern）

實驗設定：
  先只處理純 syntactic constraints，
  看 syntactic@1 能到多少，
  再逐步加入 semantic layer
```

---

## 總結：判斷有沒有搞頭的關鍵指標

```
Stage 1 通過 → 演算法 correctness 沒問題，值得繼續
Stage 2 通過 → 確認 false negative 問題存在且我們能解決
Stage 3 通過 → 速度可接受，有實用價值
Stage 4 通過 → 端對端有改善，可以寫論文
Stage 5 通過 → 對 JsonSchemaBench 有貢獻，差異化很強

如果在 Stage 3 卡住（太慢）：
  → 退回去做 approximation（例如只追蹤 top-k masks_used）
  → 或改成 hybrid（先用 wildcard verify，太慢時 fallback 到 sampling）
```



**關鍵**：這些 ground truth 是你**手動推導**出來的，不依賴任何 baseline。

---

和 LAVE 比較（需要跑 dLLM）

但光是 unit test 通過，不代表這個方法在實際 dLLM 場景裡有用。

真正需要證明的是：

```
LAVE 說 False（reject）
但 wildcard_earley_verify 說 True（accept）

這些 case 裡，
最後 dLLM 真的可以生成合法的 output 嗎？

如果可以 → 證明 LAVE 有 false negative，我們的方法更好
如果不行 → 可能是 model distribution 的問題，grammar 說可以但 model 填不出來
```

所以實驗設計應該是：

```
Step 1: 用 dLLM 跑 HumanEval-CPP，記錄所有 LAVE reject 的 case
Step 2: 對這些 case 跑 wildcard_earley_verify，找出 LAVE 說 False 但我們說 True 的
Step 3: 對這些 disagreement case，讓 dLLM 繼續生成（不加任何 constraint）
Step 4: 看最終輸出是否合法

如果 Step 4 裡有合法輸出 → LAVE 的 false negative 被我們修復了
```
