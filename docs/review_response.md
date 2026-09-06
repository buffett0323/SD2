# DPGrammar — Review 回應與修改追蹤

針對 CS AI review 八點負面評價的逐點分析、修改狀態、與待辦實驗。
建立於 2026-09-01。**死線：2026-09-05。**

---

## 總體判斷

八點裡：

- **只有第 2 點是真正的技術漏洞**（而且比 reviewer 說的還嚴重一點）
- 第 **1、7** 點是措辭過強，改字就能解 → **已完成**
- 第 **3、5、6** 是實驗覆蓋，其中第 5 點 repo 裡**已經有現成的東西沒放進 paper**
- 第 **4** 點 reviewer 過度解讀了一個 n=272 上的 3-instance 差異，**可以正面反駁**
- 第 **8** 點是補一個 config 表

---

## 進度表

| # | 主題 | 狀態 | 備份 |
|---|---|---|---|
| — | Missing related work（4 篇） | ✅ 已完成 | `custom.bib.bak`, `main.tex.bak` |
| 1 | exactness / top-k 截斷 | ✅ 已完成 | `main.tex.p1.bak` |
| 7 | 措辭過強 | ✅ 已完成 | `main.tex.p7.bak` |
| 2 | merge key κ | ✅ 文字 + 量測完成 | `main.tex.p2.bak`, `appendix.tex.p2.bak`, `dp_generate.py.bak` |
| 3 | 單一 backbone / schedule | ✅ schedule 兩組完成（backbone 仍一個） | `main.tex`, `appendix.tex` |
| 4 | functional@1 下降 | ✅ McNemar p=0.38 + 觸發率 14/272 | |
| 5 | baseline 不乾淨 | ✅ 完成（FA n=101） | `main.tex.p5.bak` |
| 6 | L / k sensitivity | ✅ 兩條曲線，零成本，不需 sweep | `main.tex.p68.bak` |
| 8 | 可重現性 | ✅ 已完成 | `appendix.tex.p68.bak` |
| — | Fig.1 移正文 / Fig.3 / Table 1 latency | ✅ 已完成 | `main.tex.figtab.bak`, `appendix.tex.figtab.bak` |

---

## 1. "exact" 過強（top-k 截斷）— 成立，但可以反守為攻 ✅

**Reviewer 指控**：Abstract 說 "global highest-likelihood valid assignment"、§3.3 說 "returns the exact maximiser"，但 Alg.1 line 5 只取 `L(q)` 的 top-k。附錄自承只有 `|L(q)| ≤ k` 時才窮舉（k=100 時 39.7%）。

**成立。** 但有一個關鍵區分沒講清楚，講出來反而變賣點：

- **可行性**：top-k 截斷**永遠不會**讓 DP 找不到合法解（C ⊆ L(q)，且 6,657 個 live state 中 0 個空 legal set）
- **最優性**：只賠 likelihood

這正是對 filter-then-propose 的核心論證的另一半——**vocabulary top-k 會賠掉合法性，automaton top-k 只會賠掉機率**。

**已做的修改（5 處）**：

1. Abstract：`global highest-likelihood` → `within the candidate lattice it expands` + `narrowing that lattice can cost likelihood but never validity`
2. Intro：`exact maximiser` → `maximiser over that lattice`，並補「從 L(q) 排名才讓有界搜尋安全」
3. §3.2：目標函數後補一句 `The search solves it over a subgraph: a budget k retains the k likeliest edges at each node`
4. §3.4：把 `returns the best assignment the grammar accepts whenever one exists` 換成三項 itemize：
   - *Validity is unconditional* — 任何 k 都不可能輸出非法 span
   - *An answer is returned* — 0/6,657 空 legal set
   - *Optimality is relative to the lattice* — 截斷賠機率不賠合法性
5. Conclusion：拿掉 `exactly`

**仍待辦**：k sensitivity 曲線（併入第 6 點）。

---

## 2. state-merging key κ(q)=bytes(L(q)) — 真的有問題 ✅文字 / 🔬量測中

**Reviewer 指控**：附錄 D 宣稱「admit 相同 token 的兩個 parser state 對所有後續決策不可區分」，這對一般 CFG incremental parser 是強宣稱且未證明。

**這句話是錯的，不只是未證明。**

### code 裡確認的事實（`dp_generate.py:296-470`）

- `init_key = bytes(matcher.compute_logit_bias())` — key 只有 mask。實作註解自己寫 **"a proxy for the DFA node"**，比 paper 誠實
- `winners[new_key] = (prev_key, tid, ...)` + `back` — backtrace 走 **predecessor link**，不是從 key 重建 → **合法性不會被 collision 破壞**
- `if not winners: n_done = step; break` — 撞死是**結束 span**，不是失敗

### 結論

| | 狀態 |
|---|---|
| 產生非法 span | **不可能** |
| 找不到解 | **不會**（退化成較短的 repair） |
| 丟掉最優解 | **會**（唯一真實損失） |

### llguidance API 查證（`_lib.pyi`）

**沒有暴露任何 state identifier。** 只有 `deep_copy` / `is_accepting` / `is_stopped` / `rollback` / `compute_ff_bytes` / `compute_bitmask` / `compute_logit_bias` / `get_captures`。真正的 state 是 Earley item set + lexer state，拿不到。

→ **exact merge 在這個 parser 上透過 public API 做不到。** 這正好是 reviewer 自己說的
「*then it should be stated as an implementation-specific assumption*」。

### 已做的修改（main.tex 6 處 + appendix.tex 1 處）

1. Abstract：拿掉 `sufficient statistic`
2. Intro：改寫 + **主動自曝**「we identify states by the token set the parser admits, which merges somewhat more aggressively than that argument licenses」
3. §3.3：刪掉 `It is exact rather than a pruning heuristic`；`reachable parser states |S|` → `distinct keys |S|`
4. §3.3 新增 `\paragraph{What the key identifies.}`：κ 是 abstraction 非 identifier；**key 只剪枝不重建** → *A collision cannot produce a span the grammar rejects*；代價付在 objective 和改寫範圍
5. Conclusion：拿掉 `sufficient statistic` / `merge without discarding the optimum`
6. §6 Limitations：新增「exact only up to two approximations we bound but do not measure」
7. Appendix D 整段重寫：給出 `minItems: 5` 分離反例、為什麼仍 sound、為什麼實務上 collision 少（byte-level BPE 整 token mask 帶 lookahead）、免費精緻化路徑（`is_accepting` / bracket depth）

### ⚠️ 待驗證

附錄那個 `minItems` 反例是**推理出來的，沒在 llguidance 上實測**。投稿前花十分鐘：拿 `minItems: 5` 的 schema，teacher-force 到第 2 和第 3 個元素後，比對兩次 `compute_logit_bias()` 的 bytes。若不同，換 `maxItems` 或 `required` 的不同已滿足子集。

### 🔬 量測：`bench/merge_probe.py`

見本文件末的〈merge probe〉一節。

---

## 3. 單一 backbone / 單一 schedule ✅（schedule 部分）

Reviewer 原話：*at least one additional backbone **or** one materially different
sampler/remasking schedule*。我們做了後者，**兩組**。

### 加的 plumbing

`remasking` 原本寫死在 runner 裡，但 `dp_generate.py:1094` 早就實作了
（`else: x0_p = torch.rand(...)`）。加 `argv[18]`（`run_dgrammar_timed.py`）
和 `--remasking`（`modal_dgrammar_bench.py`），**預設 `low_confidence`
所以既有結果不受影響**。`--block-ar 0` 本來就接出來了。

跑之前用 $0 的本機檢查驗證 19 個 argv 全部對位（模擬 Modal 組出來的 cmd）。

### 結果（n=511，`bench/measure_valid.py` 計分）

| Schedule | schema@1 DPG/no-DP | Δ | content DPG/no-DP | Δ | rs/inst |
|---|---|---|---|---|---|
| low-confidence, block 32（論文） | 96.7 / 91.8 | **+4.9** | 84.9 / 74.6 | **+10.4** | 1.2 / 33.2 |
| random unmasking order | 92.0 / 91.6 | **+0.4** | 76.3 / 59.1 | **+17.2** | 3.1 / 63.3 |
| full parallel, block 256 | 92.4 / 77.3 | **+15.1** | 76.9 / 63.8 | **+13.1** | 1.0 / 30.0 |

**content margin 三個 schedule 全部成立，而且在較難的 schedule 上更大。**
**validity margin 從 +0.4 到 +15.1。**

### random order 的 +0.4 是怎麼回事（機制）

| | 違規 | DP 呼叫 | **dead-ends** | 無效 | 其中步數耗盡 |
|---|---|---|---|---|---|
| reported | 1,106 | 315 | **3** | 17 | 16（94%） |
| random | **2,576** | **733** | **1** | 41 | 40（**98%**） |
| fullpar | 872 | 320 | **0** | 39 | 39（100%） |

**搜尋本身沒有失敗**（733 次只 dead-end 1 次）。掉的 validity 全部來自
`schedule_end`——$T{=}128$ 步用完時文件還沒閉合。random order 製造 2.3 倍違規、
每個都要花步數修，46% 的 instance 在文件完成前用光步數（reported 是 31%）。

**論文寫法**：正文 §4.2 一段（163 字）+ 附錄 `app:schedules`（表 + 三段）。
關鍵句是把 `schedule-agnostic` 限定成**適用性**而非**增益大小**，並把 +0.4
轉成「步數預算」的機制觀察，誠實收在 `Raising $T$ is the obvious remedy and one
we do not test here.`

### 沒做的

換 backbone（Dream-7B）。reviewer 的要求是 backbone **或** schedule，已滿足。

## 4. functional@1 52.6 vs 53.7 — reviewer 過度解讀 ⬜

n=272，1.1pp = **3 個 instance**。任何檢定下都不顯著。但目前寫法（"our repair layer costs 1.1pp"）等於自己承認一個雜訊。

**該做**：

- 報告 **DP layer 在 JSON-Mode-Eval 上實際觸發了幾個 instance**。no-DP schema@1 已 97.4%，違規極少，DP 大概只在個位數 instance 開火 → 這張表**對 repair layer 沒有統計檢定力**，要明講
- paired 分析：DP 觸發的 instance 中，correct→incorrect 幾個、incorrect→correct 幾個
- 加 McNemar / bootstrap CI
- §5.2 改成：「在 layer 幾乎不觸發的資料集上，兩臂差 3 個 instance，無法區分」

---

## 5. Baseline 不乾淨 ✅文字 / 🔬 FA 實驗待跑

### (a) 還原被註解掉的段落 ✅

`main.tex` 裡整段解釋 DINGO / `dang2026fadllm` 為何不比較的文字被 `%` 註解掉（`% no-need`）。
已還原成正式的 `\paragraph{The exact methods we do not run.}`，並加一句指向新的
§`sec:prevention`：我們不重實作他們的論文，而是用自己的 exact constrained inference
在有 FA 的子集上量測 prevention 的代價。

### (b) LAVE timeout 會計 ✅

`\paragraph{Baselines.}` 補上：cap 對每個 arm 相同、觸頂算失敗、medium 上是
LAVE 的 60/511 而 no-DP 的 0，並指向 §5.1 會報告「LAVE 在跑完的 instance 上的分數」。

從 shard 重算：**LAVE medium 跑完的 451 個裡 402 個合法 = 89.1%**（vs 全集 78.7%）。
正文用的是**從 Table 1 自己的數字推導**的 88.7%（78.3% × 511 = 400；400/451 = 88.7%），
以保持內部一致。

> ⚠️ **要你確認**：我從 shard 重算 medium LAVE 全集是 **78.7%（402/511）**，
> 但 Table 1 寫 **78.3%**，差 2 個 instance。可能是 dedup 規則不同。
> 確認後兩處數字要一致。

### (c) shared-351 搬進正文 ✅

§5.1 新增 `\paragraph{Comparing decoders rather than front-ends or budgets.}`，
把兩個「被評分在自己沒選的母體上」的 arm 都在正文處理：

- **CD-CFG**：rustformlang 只收 383/586，llguidance 收 511；在兩者都收的 351 個上，
  CD-CFG 45.0%（19 timeouts）vs DPGrammar 97.4%（0），且每個 arm 絕對值都上升
  → 只有 llguidance 收的 160 個是**較難**的而非較易的
- **LAVE**：跑同樣的 511，但 60 個觸頂；在跑完的 451 個上 88.7% vs 我們全集 96.7%
- 兩個調整都不改變 arm 的排序，「margin 不是 coverage 或 timeout 的產物」

Reviewer 看過 Table 5 但沒被說服，因為它離主表太遠——現在在正文了。

### (d) FA baseline — 新增 §`sec:prevention`，數字待填 🔬

`dgrammar/fa_generate.py` 已實作 proactive exact FA inference（HMM forward-backward
over 約束後驗，viterbi/marginal 兩個 decoder），`bench/modal_fa_bench.py` 已支援
`--task jsb_medium --sweep --chunks N`。**基礎設施是現成的，只差跑。**

已在 `main.tex` 寫好 `\subsection{Prevention on the subset where prevention is available}`
含公式、三項代價（coverage / latency / content）的論證骨架、和
`\label{tab:prevention}` 空表。**所有數字都是 `\fixme{}`，在 preamble 定義成紅色粗體，
不可能靜悄悄地混進投稿 PDF。**

現有 pilot（5 instances，**太薄不能寫進正文**）：

| instance | fa_viterbi | fa_marginal |
|---|---|---|
| o10217 | valid, 8 leaves, 66.2s, overhead 90.0% | valid, **0 leaves** `{"months":[]}`, 85.4s, overhead 92.5% |
| o10518 | valid, 17 leaves, 54.5s, overhead 89.1% | valid, 10 leaves, 74.0s, overhead 91.6% |
| o10297 / o1050 / o10566 | **None**（automaton 建不起來） | **None** |

三個線索：**overhead ~90%**（prevention 的代價是真的）、**mode decoder 塌縮成空文件**
（直接強化 content floor 論證）、**5 個裡 3 個建不出 automaton**（coverage 是第一級發現）。

**要跑的指令**：

```bash
modal run bench/modal_fa_bench.py \
  --task jsb_medium --sweep --chunks 8 --steps 128 --gen-length 256 --tag full
modal volume get dgrammar-results "fa_*_jsb_medium_s0_t128*full*.jsonl" results/
```

跑完要填的六格：automaton 建成數 / schema@1 / content / mean s / constraint %，
viterbi 和 marginal 各一列，再加 no-DP 和 DPGrammar 的 constraint %。

## 6. L / k 沒有 sensitivity ✅文字 / 🔬 sweep 待跑

### 主文的 span 規則跟實作不符 — 已更正 ✅

**實際規則**（`find_constraint_end()`）：從 $v$ 往前掃，停在第一個
「**被文法接受且 bracket depth 回到 0**」的原 token（junction）；找不到就用
`max_lookahead = 48` 的 cap。之後 `dp_fix_prefix`（`include_masked=False`）
再截在第一個 MASK。`max_positions=64` 因為 48 < 64 所以對 repair 從不生效。

docstring 明講目的：*"prevents the DP span from stopping inside an unclosed array
or object (which would allow the DP to collapse the remainder to [] or {})"*
—— **這正是 content floor 論證的機制**，卻沒寫進論文。

**現成量測**（v6dp run 的 `span_sites_detail`，n=315，正好對上 §5.3 的「315 sites」）：

| 綁住 span 的是什麼 | 次數 | 比例 |
|---|---|---|
| 第一個 MASK | 238 | **75.6%** |
| depth-0 junction | 31 | 9.8% |
| 48 的 cap（lookahead 25 + dead 21） | 46 | 14.6% |

span 長度 **mean 13.8 / median 9 / max 48**。

主文 §3.2 已改寫成完整規則 + 為什麼要 depth 測試 + 上述量測數字。
原本的 `e = min(v+L, first MASK)` 描述的是最常見的 76% 那條路徑，
不算全錯但漏掉唯一有設計意圖的部分。

### 新增 §`sec:budgets`（數字待填）🔬

`\subsection{How the budgets matter}` + `tab:budgets`，k ∈ {8,32,100,256}、
L ∈ {8,16,48,96} × (schema@1, content, mean s, DP calls)。全部 `\fixme{}`。

**已補上 CLI plumbing 讓 sweep 可跑**（原本 `top_k_dp` / `max_lookahead` 從外面碰不到）：

- `dgrammar/dp_generate.py`：`generate_dp` 新增 `max_lookahead: int = 48`，
  並傳進 `find_constraint_end`
- `bench/run_dgrammar_timed.py`：`argv[16]=k`、`argv[17]=L`，**兩者預設值即為
  reported configuration，省略時完全重現主結果**
- `bench/modal_dgrammar_bench.py`：新增 `--top-k-dp` / `--max-lookahead`

**指令**（每個 arm 一次）：

```bash
for K in 8 32 256; do
  modal run bench/modal_dgrammar_bench.py --method dp --dataset jsonschema \
    --total 511 --chunks 8 --top-k-dp $K --tag k$K
done
for L in 8 16 96; do
  modal run bench/modal_dgrammar_bench.py --method dp --dataset jsonschema \
    --total 511 --chunks 8 --max-lookahead $L --tag L$L
done
```

> 注意：附錄 `app:legalsets` 那張「|L(q)|≤k 的 state 比例」是**輸入端**統計，
> `tab:budgets` 是**輸出端**。兩者未經量測前不要宣稱一致（已寫進 `\fixme{}` 註記）。

## 7. 措辭過強 ✅

**已做的修改（9 處）**：

| # | 位置 | 舊 → 新 |
|---|---|---|
| 1 | §2 Viterbi 段（reviewer 點名的 p.3） | `have no impact on the final output` → `an explicit edit penalty never becomes non-zero in any of our runs, and swapping the objective for minimum edit distance leaves 93.9% of outputs byte-identical` |
| 2 | §2 LAVE | `the dominant published baseline` → `the strongest published baseline we are able to run` |
| 3 | §5.1 | `costing a mere 3.3s` → `at 3.3s of additional mean wall time and a wider tail (p95 21.4→36.2s)` |
| 4 | §5.2 | `Grammar constraints buy structure and not semantics.` → 前面加 `On this benchmark,` |
| 5 | §5.3 標題 | `the objective does not matter, the search space does` → `what moves the metric is the search space, not the objective` |
| 6 | Table 4 caption | `knobs that measure to exactly zero`（**與表內 min-edit 的 93.9% 自相矛盾**）→ 明確說明下半區「leaves the output identical, except for the min-edit swap, which rewrites 6.1% of them」 |
| 7 | §5.3 開頭 | 新增 `within one split, one backbone and one seed, and we read it as evidence about this configuration rather than as a general law` |
| 8 | §5.3 內文 | `Among the inert components` → `Among the components in the lower block` |
| 9 | §5.3 結語 | `Changing what the search optimises cannot help` → `Across every configuration we measured, changing what the search optimises left the metric where it was` |

外加：`two components move the metric` → `move schema@1`。

### ⚠️ 需要你確認的數字

Table 4 的 min-edit 那列只給 `−0.5 edits, 93.9% identical`，**沒有 schema@1**。
現在的 caption 和內文隱含「min-edit 不改變 schema@1」。
`results/` 裡只找到 59 筆 partial shard，無法自行驗證。**若 min-edit 其實有動到 schema@1，caption 和 #9 都要再調。**

---

## 8. 可重現性 ✅

### "additive-increase batching" 是錯的 — 已更正

程式碼是 `current_batch = min(current_batch * 2, max_batch_size)`：
**乘法**遞增（×2）、上限 8、**只在 `_global_step >= steps*3//4` 之後才允許成長**、
任何 violation 重設回 1。§3.6 已改寫成正確描述。

### 新增 Appendix `app:config`「Implementation and hyperparameters」

一張表列完所有常數，全部從程式碼讀出來：

| 類別 | 內容 |
|---|---|
| Backbone / schedule | T=128、gen_length=256、block=32、temp=0.2、low-confidence remasking、seed 0、A100-80GB |
| Repair layer | **k=100**、**L=48**、DP position cap 64、retry depth 10、window=full、merge key κ=bytes(L(q))、paths per key=1 |
| Budgets / batching | hand-backs 100、DP wall-clock 240s、batch cap 8、**×2 成長**、成長 gated 到 3T/4 |
| Parser | llguidance 1.7.0、`max_items_in_row=20000`、`step_max_items=600000`、EOS 126081 / EOT 126348 / MASK 126336、\|V\|=126,349 |

加上一段「表格裝不下的兩個行為」：

1. 失敗的 single-token retry **不計入** hand-back budget（只有 commit 或 fall-through 才算），
   所以 rank > 10 的違規不會在 DP 有機會跑之前燒光預算
2. 下一個位置的 mask 是非同步發出、在 forward pass 之後 join；frontier 一有狀態改變
   就丟棄 pending mask 而不是用過期的

## Figure / Table 三點 ✅

### A. Fig.1 移到正文 ✅

`pipeline.pdf` 從 Appendix A 移進 §3.2，放在四個 control-flow case 的 enumerate 之後（`[t]` float）。
**Appendix A 整節刪除**，`app:pipeline` 的引用全數清乾淨（已 grep 確認無 dangling ref）。
§3.2 開頭原本只說「圖在附錄」，現在把圖的內容直接寫進正文：
每個 denoising step 推進 parser、rejection 經過 termination test 和 bounded retry 才到 joint search、
除 emission 外每個出口都回到下一步、shaded box 是本文貢獻。

### B. Fig.3 / latency ✅

原本只有一段輕描淡寫，且解釋埋在 Appendix F。改成正文一個獨立
`\paragraph{What the extra wall time is.}`，四件事講清楚：

1. **開場就認**：`The layer is not a latency win and we do not report it as one` ——
   把 reviewer 的話先講掉
2. **但 3.3s 不是搜尋**：constraint machinery 909 vs 920 ms；差異集中在 **173 個真的觸發的 instance**，
   另外 338 個 DPGrammar 還比較快
3. **正規化**：那 173 個裡 baseline 停在 133 token、DPGrammar 續寫到 227；
   1.5× forward pass 買 1.7× token，**per token 兩臂成本相當**
4. **baseline 的速度是買來的**：`The baseline is faster in part because it gives up earlier` ——
   18.8% 的合法輸出少於五個 leaf，對比我們的 12.1%。這跟 content floor 是同一個行為
5. **easy split 上 DPGrammar 更快**（mean 5.31 vs 7.58、median 3.40 vs 7.58），
   原本完全沒 highlight，reviewer 因此以為「一律更慢」
6. **主動報告變差的部分**：medium p95 21.4→36.2s，hard 上 2 個 timeout 而 baseline 0 個

> 注意：**沒有**寫「per token 快 27%」那種話。133/227 是 173 個 instance 的平均，
> 13.45/16.75 是全部 511 個的平均，兩者母體不同，相除會是造假。
> App F 的 `(c) Per unit the two arms are close` 只支撐「相當」，就只寫「相當」。

### C. Table 1 / abstract 用 rs/inst ✅

Abstract 補上 end-to-end 成本：
`at $3.3$\,s of additional mean latency spent writing $1.7\times$ the tokens`。
誠實反而拆掉「用 mechanism-internal 指標迴避」這個攻擊面。

§5.1 原本第 7 點加的 `and a wider tail (p95 21.4→36.2s)` 改回
`which the next paragraph accounts for`，避免同一組數字講兩次。

## Missing related work ✅

四篇都已加入 `custom.bib` 並在 §2 用上，書目資料全部上網查證：

| key | 出處（已驗證） | 放在哪、做什麼 |
|---|---|---|
| `geng2023grammar` | EMNLP 2023, pp. 10932–10952 | §2 第一段段首，當基礎工作，**用來擋掉「JSON-only」質疑** |
| `tuccio2025grammarllm` | Findings of ACL 2025, pp. 3412–3422 | §2 第一段，延伸到 taxonomy / domain rule |
| `nakshatri2025speculative` | NAACL 2025 Long, pp. 4681–4700 | §2 第二段，掛在 LAVE 上，接到 Table 1 的 timeout |
| `cardei2025constraineddiscretediffusion` | **NeurIPS 2025**（arXiv:2503.09790，v3 才掛 NeurIPS） | §2 第二段，補完 prevention 陣營（differentiable projection vs automaton） |

---

## merge probe（第 2 點的量測）

### 檔案

- `dgrammar/dp_generate.py` — `dp_fix_prefix` 新增 `beam_per_key`（預設 `1`，`None` = 完全不合併）、`max_live`；`out` 回報 `best_score` / `assignment` / `exhaustive` / `n_live_max` / `capped`。**`beam_per_key=1` 與原版逐位元相同**
- `bench/merge_probe.py` — CPU-only，不用 GPU 不用模型
- `bench/modal_merge_probe.py` — Modal wrapper（只掛 3MB v6dp shards）

> ⚠️ `bench/test_min_edit.py` 目前是壞的（unpack 錯誤），但**改之前就壞了**（用原版檔案跑過確認），簽章比 `(replacements, reached_end)` 舊。不是 regression。

### 兩層設計

| | Tier A | Tier B |
|---|---|---|
| lattice | **論文真實設定** k=100, span≤8 | 縮小 k=6, span≤4 |
| 比較 | beam=1 vs beam=2/4/8 | beam=1 vs **完全不合併** |
| 強度 | 證據 | **證明**（該 trial 上） |
| 為什麼 | k=100 不合併是 100⁸ 條路徑，跑不動 | 6⁴=1296 條，可窮舉 |

`exhaustive` flag 只在全域 `max_live` cap **從未觸發**時為 True。

### 指令

```bash
# 本機小樣本（CPU，約 25 分鐘）
cd ~/Desktop/dllms/anlp_final
.venv/bin/python bench/merge_probe.py \
  --instances 16 --seeds 2 --stride 5 --max-sites-per-instance 10 \
  --out results/merge_probe_pilot.jsonl

# Modal 中樣本（先確認 image build，約 10 分鐘）
modal run bench/modal_merge_probe.py --instances 64 --chunks 4 --seeds 2 --tag pilot

# Modal 全樣本（511 instances，16 容器，約 1.5–2 小時）
modal run bench/modal_merge_probe.py \
  --instances 511 --chunks 16 --seeds 3 \
  --stride 5 --max-sites-per-instance 12 --tag full

# 收結果
modal volume ls  merge-probe-results
modal volume get merge-probe-results "merge_probe_full_off*.jsonl" results/
```

彙總腳本見對話記錄（讀 `results/merge_probe_*.jsonl`，統計 `tierA.arms[*].differs` 和
`tierB.differs` / `tierB.exhaustive`）。

### 全量結果 ✅（447 instances / 4,529 sites / 13,513 trials，2026-09-02）

```
Tier B (k=6, span<=4)   exhaustive 7,836 → 與真正最優差異 0/7,836 (0.000%)
Tier A (k=100, span<=8) beam 1→2: 26/13,513 (0.19%) 答案改變
                        beam 1→4: 44/13,513 (0.33%) 答案改變
```

**合併確實會丟掉最優解，但只有 0.33%，而且只賠 likelihood。**

- 那 44 次**每一次**都是 beam=1 分數較低 → 確實丟了最優
- B2 的 26 個是 B4 的 44 個的**子集**（單調，符合預期）
- 影響 34/4,529 sites（0.75%）、22/447 instances（4.9%）
- likelihood 損失中位數 **4.63 nats**（範圍 0.18–21.2）
- **validity 從未受損** —— 44 次全是合法輸出，只是機率較低
- Tier B：在小到能窮舉的 lattice 上，合併**每次都回傳真正最優**
- 坍縮規模：unmerged 中位數 1,869 條路徑 → beam=1 只留 5 條

**這比全零好**，因為它精確符合寫進論文的理論（sound but lossy，代價落在 objective），
而且是量測不是宣稱。§6 從「we bound but do not measure」改成量測值。

### 兩個 Modal entrypoint 預設值的失誤（我的）

1. `beams` 預設 Modal 是 `"1,2,4"`、本地 script 是 `"2,4,8"` → 跑出無用的
   「B1 vs B1」arm，且**沒有 B8**。0.19% → 0.33% 還在上升，不知在哪飽和。
2. `max_live` 預設 Modal 2048、本地 20000 → Tier B 只有 58% 真正窮舉完
   （另 42% 撞 cap，不列入證明）。

要補的話：`--beams 4,8 --max-live 20000` 重跑一次全量，約 $11，可得飽和點。

### 這個量測不能證明什麼（要誠實寫進 limitations）

1. **是逐次試驗的證明，不是定理**。「這 N 個設定下沒丟」≠「永不丟」。全稱命題需要 parser 層級的證明，而沒有 state identifier 就做不到
2. **Tier B 的 lattice 較小**（k=6/span 4），只在較窄候選集下可達的 state 才被測到。這正是 Tier A 存在的理由——它在真實 k=100/span 8 上顯示加寬 beam 不改變答案
3. **隨機分數 ≠ 模型分數**。隨機排序探索的分數組態比模型的尖峰分佈更多，所以是**更嚴苛**的測試，但仍是不同分佈
4. **只取結構位置**（不在字串內、`|L(q)| ≤ 4000`）。字串內部沒測——但那裡每個延續都是內容，問題本身沒意義

---

## 建議優先序

| 優先 | 動作 | 成本 | 解掉 |
|---|---|---|---|
| ~~1~~ | ~~拆解 exactness claim~~ | ✅ | #1, #7 |
| ~~2~~ | ~~κ 改寫成 sound-but-lossy~~ | ✅ | #2 |
| 3 | 還原被註解的 DINGO 段落 + shared-351 搬進 Table 1 | 改字 | #5 一半 |
| 4 | 主文補真正的 span 規則（junction/depth-0）+ 所有超參數值 | 改字 | #6, #8 |
| 5 | Fig.1 移正文；abstract 加 end-to-end latency；latency/token | 改字 | fig/table 三點 |
| 6 | merge probe 全樣本 | 小實驗（跑中） | #2 |
| 7 | k / L sensitivity | 小實驗 | #1, #6 |
| 8 | JSON-Mode-Eval 觸發率 + paired flip + CI | 小實驗 | #4 |
| 9 | **`fa_generate.py` pilot 擴到 medium 全 FA 子集** | 中實驗 | #5 另一半 + 強化 content 論證 |
| 10 | 多 schedule（random remask / semi-AR / 不同 T,block） | 中實驗 | #3 一半 |
| 11 | Dream-7B on medium | 大實驗 | #3 |
