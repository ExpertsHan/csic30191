# Lab1 - DDPM 完整教學筆記

> 這份筆記的目的是帶你**理解**每一個 TODO 背後的數學意義，並給你足夠的線索、程式碼骨架與檢查方法，讓你能夠**自己**把程式碼補完、拿到滿分。我不會直接把最終可貼上的答案寫出來（尤其是需要你自己組合公式的地方），但公式本身已經在課程投影片裡完整給出，所以你的工作主要是「把數學正確地翻譯成 PyTorch tensor 運算」。

- 截止日期：2026/10/1 (四) 23:59
- 你需要交出：`{學號}_lab1.zip`，內含 `report.pdf`、`2d_plot_diffusion_todo/`、`image_diffusion_todo/`（**不要**附上 `data/`、`results/`、`samples/`、`fid/afhq_inception_v3.ckpt`）

---

## 0. 先做這件事：環境設置與時間規劃

```bash
conda create -n ddpm python=3.9 -y
conda activate ddpm
pip install -r requirements.txt
```

⚠️ **Task 2 的訓練一次要跑 6 小時以上**（50k–100k iterations），而且投影片要求你至少比較：
- 3 種 beta scheduler（linear / quadratic / cosine，固定用 noise predictor）
- 3 種 predictor（noise / x0 / mean，固定用 linear scheduler）

也就是說你最少要跑 **5 組完整訓練**（因為 noise+linear 這組兩邊都會用到，不用重跑）。請**現在就先把 Task 2 的 code 補完並開始跑第一組訓練**，跑訓練的同時再回頭寫 Task 1 跟報告，否則時間會非常緊張。

---

## 1. DDPM 概念複習（兩個 Task 都會用到）

DDPM 有兩個過程：

1. **Forward process（加噪）** $q(x_t \mid x_0)$：把乾淨資料 $x_0$ 逐步加高斯雜訊，直到變成純雜訊 $x_T \approx \mathcal{N}(0, I)$。
   DDPM 的一個關鍵性質是：你**不需要**一步一步地加噪 $T$ 次，而是可以直接「一步跳到」任意時刻 $t$：

$$q(x_t \mid x_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\big)$$

   其中 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{i=1}^t \alpha_i$（也就是程式裡的 `alphas_cumprod`）。
   用 reparameterization trick 展開成可微分的取樣：

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)$$

   這就是整個作業裡**最常用**的一條公式，`q_sample`（Task1）跟 `add_noise`（Task2）都是在實作它。

2. **Reverse process（去噪）** $p_\theta(x_{t-1} \mid x_t)$：訓練一個神經網路，學會把雜訊一步步去掉。DDPM 證明這個反向分布也是高斯分布：

$$p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\big(\mu_\theta(x_t,t),\ \tilde\beta_t I\big)$$

   其中 posterior variance（後驗變異數）：

$$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

   而 posterior mean（後驗平均值）如果用「網路預測雜訊 $\hat\epsilon_\theta$」的參數化方式，可以寫成（DDPM paper Eq. 11）：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon_\theta(x_t,t)\right)$$

   取樣時：

$$x_{t-1} = \mu_\theta(x_t,t) + \sqrt{\tilde\beta_t}\, z,\qquad z\sim\mathcal N(0,I)\ (t>0\text{ 時}),\quad z=0\ (t=0\text{ 時})$$

3. **訓練目標（loss）**：DDPM paper 證明，只要讓網路去預測「加進去的雜訊」，並用簡單的 MSE 就等價於在最大化 variational lower bound（簡化版，Eq. 14）：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\hat\epsilon_\theta(x_t, t) - \epsilon\|^2\right]$$

把這三塊記熟，Task1 跟 Task2 的所有 TODO 幾乎都是在不同地方重複使用這三條公式而已。

---

## 2. Task 1 — Swiss Roll（`2d_plot_diffusion_todo/`）

整個流程都在 `ddpm_tutorial.ipynb` 裡跑，**一定要在 `2d_plot_diffusion_todo/` 目錄下開啟 notebook**（因為它用相對路徑 import `dataset.py`、`network.py`、`ddpm.py`）：

```bash
cd 2d_plot_diffusion_todo
jupyter lab ddpm_tutorial.ipynb
```

### TODO #1 — `SimpleNet`（`network.py`）

**目標**：輸入 `x`（2 維座標，shape `(B, 2)`）和 `t`（timestep，shape `(B,)`），輸出對應形狀的「預測雜訊」 $\hat\epsilon_\theta(x_t,t)$（shape 一樣是 `(B, 2)`）。

投影片給了明確提示：**用 `TimeLinear` 這個 building block**。看一下 `network.py` 裡已經寫好的 `TimeLinear`：

```python
class TimeLinear(nn.Module):
    def forward(self, x, t):
        x = self.fc(x)                       # 一般的線性層
        alpha = self.time_embedding(t)...     # 把 t 編碼成向量
        return alpha * x                      # 用 time embedding 「調變」特徵
```

也就是說 `TimeLinear` 已經幫你把「時間資訊」和「線性層」融合好了，行為很像一般的 `nn.Linear`，差別只是多吃一個 `t` 參數。所以你要做的事情，其實跟寫一個一般的 MLP（多層感知機）幾乎一樣：

- `__init__` 裡：
  - 用 `dim_in -> dim_hids[0] -> dim_hids[1] -> ... -> dim_out` 的順序，把每一層都換成 `TimeLinear(前一層維度, 這一層維度, num_timesteps)`，串成一個 `nn.ModuleList`（**不要用 `nn.Sequential`**，因為 `Sequential.forward` 只吃一個參數，沒辦法把 `t` 一起傳進每一層）。
  - 記得在每個 `TimeLinear` 之間加一個非線性 activation（例如 `nn.ReLU()`），最後一層（輸出層）**不要**加 activation，因為輸出的是雜訊，數值可正可負、範圍不受限。
- `forward` 裡：
  - 依序把 `x` 丟進每一個 `TimeLinear` 層並傳入 `t`，中間穿插 activation。
  - 最後一層是 `TimeLinear`，輸出維度是 `dim_out`，直接回傳。

**常見錯誤**：
- 忘記在最後一層之後加 activation 是對的（不要加），但忘記在中間層加 activation 就會讓整個網路退化成線性模型，學不出 Swiss Roll 那種彎曲形狀。
- `ModuleList` 裡的層數要跟 `dim_hids` 的長度 +1 一致（hidden 層之間 + 輸入到第一個 hidden + 最後一個 hidden 到輸出）。

### TODO #2 — `q_sample`（`ddpm.py`）

投影片已經把公式跟一半程式碼都給你了：

```python
def q_sample(self, x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)  # 已經幫你取好 \bar{α}_t
    xt = x0   # <-- 你要改這一行
    return xt
```

對照第 1 節的公式 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，你只需要把 `alphas_prod_t`（也就是 $\bar\alpha_t$）代入這個公式，用 `x0` 和 `noise` 組合出 `xt`。

**重點提醒**：
- `alphas_prod_t` 的 shape 已經被 `extract()` reshape 成 `(B, 1)`（對應 `x0` 的 `(B, 2)`），所以可以直接用 broadcasting 相乘，不需要自己再 reshape。
- 記得用 `.sqrt()`，不要漏開根號（$\bar\alpha_t$ 本身不是標準差，是 variance 的係數）。

**驗收方式**：跑 notebook 裡「Visualize q(x_t)」那格，`t` 從 0 增加到 450 時，散點圖應該從清晰的螺旋逐漸「糊掉」變成接近高斯分布的一團雲，和投影片裡 `assets/images/qs.png` 的效果一致。如果你的圖在 `t` 還很小的時候就已經完全變成一團雲，代表加噪加太快，去檢查有沒有漏開根號或用錯變數。

### TODO #3 — `p_sample`（`ddpm.py`）

這是**唯一一個需要你自己把多個公式串起來**的地方，但投影片把每一步都列出來了：

```python
beta_t            = extract(self.var_scheduler.betas,          t, xt)   # β_t
alpha_t           = extract(self.var_scheduler.alphas,         t, xt)   # α_t
alpha_bar_t       = extract(self.var_scheduler.alphas_cumprod, t, xt)   # \bar{α}_t
t_prev            = (t - 1).clamp(min=0)
alpha_bar_t_prev  = extract(self.var_scheduler.alphas_cumprod, t_prev, xt)  # \bar{α}_{t-1}

# 1. predict noise
# 2. Posterior mean
# 3. Posterior variance
# 4. Reverse step
```

請照著這四個步驟寫，對照第 1 節的公式：

1. **predict noise**：呼叫 `self.network(xt, t)`，得到 $\hat\epsilon_\theta(x_t,t)$。
2. **Posterior mean**：套用 Eq. 11 —
   $\mu_\theta(x_t,t) = \dfrac{1}{\sqrt{\alpha_t}}\left(x_t - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon_\theta\right)$。
   注意投影片上方還定義了一個 `eps_factor`（已經幫你算好 $\dfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}$），這其實**不是**上面公式要用的係數（上面公式的係數是 $\beta_t/\sqrt{1-\bar\alpha_t}$，而 $\beta_t = 1-\alpha_t$，所以其實 `eps_factor` 就等於這個係數！只是寫法不同而已）。你可以直接用 `eps_factor`，也可以自己用 `beta_t` 重新算一次，兩者數學上等價。
3. **Posterior variance**：$\tilde\beta_t = \dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$。這行用剛剛準備好的 `alpha_bar_t_prev`、`alpha_bar_t`、`beta_t` 直接照公式寫即可。
4. **Reverse step**：$x_{t-1} = \mu_\theta + \sqrt{\tilde\beta_t}\, z$，其中 $z\sim\mathcal N(0,I)$（用 `torch.randn_like(xt)`）。

   **特別注意**：當 `t == 0` 時（也就是最後一步、要輸出最終乾淨結果時），**不應該再加雜訊**，否則你的最終樣本會帶有殘留雜訊、變模糊。常見寫法是做一個 mask：
   ```python
   nonzero_mask = (t != 0).float().view(-1, *([1] * (xt.dim() - 1)))
   x_t_prev = mean + nonzero_mask * sqrt(posterior_var) * z
   ```

**常見錯誤**：
- 忘記在 `t==0` 時關掉雜訊。
- 把 `alpha_bar_t_prev` 跟 `alpha_bar_t` 搞反。
- variance 開根號忘記加。

### TODO #4 — `p_sample_loop`（`ddpm.py`）

對應 DDPM paper 的 Algorithm 2（Sampling）。邏輯很單純：

1. 從 $x_T \sim \mathcal N(0,I)$ 開始（已經幫你寫好：`xt = torch.randn(shape).to(self.device)`）。
2. 用一個 for 迴圈，讓 `t` 從 `T-1` 倒數到 `0`，每一步呼叫你剛剛寫好的 `self.p_sample(xt, t)`，把結果指定回 `xt`。
3. 迴圈結束後，`xt` 就是最終的 $x_0$ 預測值，回傳它（而不是回傳 `None`）。

小提示：`self.var_scheduler.num_train_timesteps` 就是總步數 $T$；`range(T-1, -1, -1)` 可以幫你倒著跑。因為函式有 `@torch.no_grad()` 裝飾器，不需要額外處理梯度。

### TODO #5 — `compute_loss`（`ddpm.py`）

投影片已經把步驟寫在註解裡了，對照第 1 節的 loss 公式：

```python
t = torch.randint(...)   # 已經幫你寫好：隨機取樣 timestep

# 2) get GT noise, and use q_sample to get x_t
#    -> 呼叫 torch.randn_like(x0) 產生 ground-truth 雜訊 eps
#    -> 呼叫 self.q_sample(x0, t, noise=eps) 得到 x_t

# 3) predict noise
#    -> 呼叫 self.network(x_t, t) 得到 eps_pred

# 4) MSE loss (eps, eps_pred)
#    -> F.mse_loss(eps_pred, eps)
```

四行程式碼，分別對應公式裡的 $\epsilon$、$x_t$、$\hat\epsilon_\theta$、$\|\hat\epsilon_\theta - \epsilon\|^2$。

**驗收方式**：訓練 5000 iterations 後，loss curve 應該從一開始的 ~1.0+ 快速下降到 0.2~0.4 附近並震盪（跟投影片給的範例圖一致），最終生成的樣本疊在 target distribution 上應該幾乎重合，Chamfer Distance 應該 < 20（notebook 裡有這個檢查）。

- 如果 loss 完全不下降：檢查 `SimpleNet` 是不是有 activation、learning rate 有沒有被改到。
- 如果 loss 下降但生成結果是一團模糊、沒有螺旋形狀：通常是 `p_sample` 寫錯（例如係數用錯、或忘記在 t=0 關掉雜訊會導致整體結果偏移）。
- 如果生成結果直接爆炸成 NaN 或超大數值：通常是 `q_sample`/`p_sample` 裡某個 `sqrt()` 對到負數，或者 broadcasting 維度不對導致算錯。

### TODO #6 — Train & Evaluate

直接照著 notebook 的順序執行到底即可，不需要額外寫程式碼，只需要確保前面 TODO 都正確。

### TODO #7 — Report（Task 1）

報告需要包含：
1. **完整解釋你所有 TODO 的實作**（20 分）：針對每一個 TODO，寫出你用了什麼公式、程式碼怎麼寫、為什麼這樣寫（不用整段貼 code，講清楚邏輯與對應公式即可，貼關鍵的 1-3 行程式碼佐證即可）。
2. **`q_sample` 的視覺化圖**（10 分）：notebook 裡「Visualize q(x_t)」那格產生的圖。
3. **Loss curve 與最終生成分布圖**（10 分）：notebook 訓練完後產生的 loss 曲線圖 + target 與 samples 疊圖。

---

## 3. Task 2 — Image Generation（`image_diffusion_todo/`）

這一部分把 Task 1 的概念套用到真實圖片（AFHQ 資料集），並且要支援：
- 3 種 **beta scheduler**：linear / quadratic（已經寫好）/ **cosine（TODO）**
- 3 種 **predictor**：**noise（已經寫好，作為範例）**/ x0（TODO）/ mean（TODO）

所有指令都在 `image_diffusion_todo/` 目錄下執行。

### 建議的實作/驗證流程

1. 先把 `scheduler.py`、`model.py` 的 TODO 全部寫完。
2. 執行 `pytest tests/test_todo.py -q`：這是**選擇性、不計分**的自我檢查，會用固定輸入比對你的實作跟 TA 參考答案的數值是否一致。CPU 上一秒內跑完，可以先確認邏輯正確再開始花 6 小時訓練，**強烈建議做**（可以省下大量除錯訓練結果的時間）。
3. 確認測試通過（或至少你確信邏輯正確）後才開始 `train.py` 長時間訓練。

### TODO #1 — `add_noise`（`scheduler.py`）

跟 Task 1 的 `q_sample` **是同一條公式**，只是這裡的張量多了 channel/height/width 維度 `[B,C,H,W]`：

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

程式碼骨架已經給你 `eps`（雜訊），你需要：
1. 用 `extract(self.alphas_cumprod, t, x_0)` 取出對應 batch 中每張圖的 $\bar\alpha_t$（這行你可能需要自己加，前面 `scheduler.py` 頂端已經 import 了 `extract` 函式，用法跟 Task1 的 `ddpm.py` 一模一樣）。
2. 套用同一條公式算出 `x_t`。

因為 `extract()` 的 reshape 邏輯是通用的（`[t.shape[0], 1, 1, ...]`），對 4 維影像張量一樣可以直接 broadcasting，不需要額外處理。

### TODO #2 — cosine beta scheduler（`scheduler.py`）

投影片的提示已經給出完整演算法（Nichol & Dhariwal, 2021）：

1. 定義 $\bar\alpha_t = f(t/T)$，其中
   $f(t) = \cos^2\!\left(\dfrac{t/T + s}{1+s}\cdot\dfrac{\pi}{2}\right)$，$s=0.008$。
   實作上通常會先算出所有 $t=0,\dots,T$（**注意是 $T+1$ 個點，包含 $t=0$**）的 $\bar\alpha_t$，因為下一步要用到 $\bar\alpha_{t-1}$。
2. 把 $\bar\alpha_t$ 轉成 $\beta_t$：$\beta_t = 1 - \dfrac{\bar\alpha_t}{\bar\alpha_{t-1}}$。
3. 把 $\beta_t$ clip 到最大 0.999（$t=T$ 時分母可能非常小，數值不穩定）。
4. 回傳長度為 `num_train_timesteps` 的 `betas` tensor。

**實作建議**（用 `torch.linspace`/`torch.arange` 產生 $t$ 的序列，向量化計算，避免 for 迴圈太慢）：
- 先算出 `alphas_cumprod` 對應的原始函數值（常稱為 `f(t)`，注意這裡的 $\bar\alpha_t$ 定義是「相對值」$f(t)/f(0)$，所以你算完 $\cos^2(\cdot)$ 之後記得除以 $t=0$ 時的值做 normalize，讓 $\bar\alpha_0 = 1$）。
- 這個函數需要算 $t = 0, 1, \dots, T$ 共 $T+1$ 個點的 $\bar\alpha$，然後用相鄰兩個點的比值算出 $T$ 個 $\beta_t$。

**常見錯誤**：
- 忘記在最前面加上 $t=0$ 那個點，導致算出來的 `betas` 長度少一個，或者 $\bar\alpha_{-1}$ 對不上。
- 忘記 normalize（除以 $f(0)$），導致 $\bar\alpha_0 \neq 1$，訓練初期的加噪行為會不正確。
- 忘記 clip，導致訓練到後面出現 NaN。

**驗收方式**：`pytest tests/test_todo.py -q` 裡有針對 cosine schedule 的測項，先跑過再訓練。

### TODO #3 — 三種 predictor 的 `step_*`（`scheduler.py`）

`step()` 的作用等同 Task 1 的 `p_sample`：拿到網路輸出，算出 $x_{t-1}$。差別是這裡網路輸出的意義（noise / x0 / mean）不同，所以要分三種寫法。

#### `step_predict_noise`（範例已給提示，跟 Task1 p_sample 幾乎一樣）

投影片註解已列出完整步驟：
1. 取出 `beta_t, alpha_t, alpha_bar_t, alpha_bar_t_prev`。做法可以完全比照 Task 1 `ddpm.py` 裡 `p_sample` 已經寫好的樣板：
   ```python
   t_prev = (t - 1).clamp(min=0)
   alpha_bar_t_prev = extract(self.alphas_cumprod, t_prev, x_t)
   ```
   嚴格來說 $t=0$ 時的 posterior 定義用的是 $\bar\alpha_{-1}=1$，但這裡用 `clamp(min=0)` 取到 $\bar\alpha_0$ 來近似，是常見 DDPM 實作的簡化寫法（因為 $t=0$ 這一步最後會被下面第 5 步的「不加雜訊」規則覆蓋掉，影響很小）。跟著 starter code 的風格寫即可，不需要額外處理。
2. 把預測雜訊轉成預測乾淨影像：
   $\hat x_0 = \dfrac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}{\sqrt{\bar\alpha_t}}$，然後 `clamp(-1, 1)`（因為訓練圖片通常正規化到 $[-1,1]$ 之間，這個 clamp 能大幅提升取樣穩定度，是 DDPM 官方實作的標準做法）。
3. 用**跟 Task1 p_sample 一樣**的 posterior mean 公式，但這裡改用剛算出的 $\hat x_0$ 版本（DDPM paper 的另一種等價寫法）：
   $\tilde\mu_t = \dfrac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\hat x_0 + \dfrac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$
4. Posterior variance：跟之前一樣 $\tilde\beta_t = \dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$。
5. 加雜訊：$t\neq 0$ 時加 $\sqrt{\tilde\beta_t}\,z$，$t=0$ 時不加。
6. 回傳 $x_{t-1}$。

#### `step_predict_x0`

網路直接輸出 $\hat x_0$（不用再用公式反推），流程跟上面幾乎一樣，只是**跳過第 2 步**（不用從 $\hat\epsilon$ 換算 $\hat x_0$），直接對輸入的 `x0_pred` 做 `clamp(-1, 1)`，然後套用第 3、4、5 步（posterior mean/variance/加噪）算出 $x_{t-1}$。也就是說這個函式跟 `step_predict_noise` 只差在「怎麼取得 $\hat x_0$」這一步，其餘完全一樣——可以把共同的第 3-5 步邏輯抽成共用寫法，或直接複製調整。

#### `step_predict_mean`

網路直接輸出 posterior mean $\mu_\theta(x_t,t)$，這個函式最簡單：
1. 算出 posterior variance $\tilde\beta_t$（公式同上）。
2. $x_{t-1} = \mu_\theta + \sqrt{\tilde\beta_t}\,z$（$t=0$ 時不加雜訊）。

### TODO #4 — Loss functions（`model.py`）

`get_loss_noise` 已經寫好給你當範例：

```python
def get_loss_noise(self, x0, class_label=None, noise=None):
    B = x0.shape[0]
    t = self.var_scheduler.uniform_sample_t(B, x0.device)
    x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
    eps_pred = self.network(x_t, t)
    return F.mse_loss(eps_pred, eps)
```

#### `get_loss_x0`

對應投影片公式 $\mathbb E\big[\|\hat x_\theta(x_t,t) - x_0\|^2\big]$。照抄上面範例的結構，只差最後一行比較對象：

- 一樣要 `uniform_sample_t` 取 `t`，`add_noise` 得到 `x_t`（這裡拿到的 `eps` 用不到，但函式還是要呼叫，因為要得到 `x_t`）。
- 把 `x_t, t` 丟進 `self.network`，這次網路輸出的意義是 $\hat x_0$（而不是雜訊）。
- loss 是 `F.mse_loss(x0_pred, x0)`（跟乾淨影像 `x0` 本身比較，而不是跟 `eps` 比較）。

#### `get_loss_mean`

對應投影片公式 $\mathbb E\big[\|\mu_\theta(x_t,t) - \tilde\mu(x_t,x_0)\|^2\big]$，這是三個裡面最麻煩的一個，因為「正確答案」 $\tilde\mu$ 需要你自己用閉式公式算出來：

1. 一樣取 `t`、`add_noise` 得到 `x_t`（以及對應的 `eps`）。
2. 網路輸出 `mean_pred = self.network(x_t, t)`，代表 $\mu_\theta(x_t,t)$。
3. **算出 ground-truth 的 posterior mean** $\tilde\mu(x_t,x_0)$：可以用第 1 節公式（Eq.11 版本，用 `eps` 表示）：
   $\tilde\mu = \dfrac{1}{\sqrt{\alpha_t}}\left(x_t - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$
   這裡的 `eps` 就是 `add_noise` 回傳、真正加進去的雜訊（不是網路預測的），所以這是「真值」。你需要從 `self.var_scheduler` 裡把 `alphas`、`betas`、`alphas_cumprod` 用 `extract(consts, t, x_t)` 取出來（`model.py` 開頭已經 `from scheduler import extract`，直接用即可）。
4. `loss = F.mse_loss(mean_pred, mu_true)`。

**常見錯誤**：
- 三個 loss function 都要記得先呼叫 `uniform_sample_t` 和 `add_noise`，不要漏。
- `get_loss_mean` 裡如果直接拿 `add_noise` 回傳的 `eps` 去跟網路的 `mean_pred` 算 MSE（忘記先轉換成 $\tilde\mu$），loss 雖然能跑但訓練出來的模型完全不對，取樣時 `step_predict_mean` 會出來一團爛（因為量綱、意義都不對）。

### 訓練、取樣、評估指令

```bash
# 1) 準備資料（一定要先做，不然 FID 會算錯）
python dataset.py

# 2) 訓練（--mode: linear/quad/cosine, --predictor: noise/x0/mean）
python train.py --mode linear --predictor noise
```

- 訓練 50k–100k iterations 才會收斂，大約 6 小時以上，**建議先跑 linear + noise 這組打底**，確認 loss 有下降、trajectory 圖看得出動物臉的輪廓，再去跑其他組合。
- checkpoint 跟中間取樣結果會存在 `results/predictor_{PREDICTOR}/beta_{MODE}/{TIMESTAMP}/`，裡面的 `step={STEP}-traj.png` 就是報告要用的去噪過程示意圖。
- 如果時間不夠，可以調高 `--log_interval` 減少 log 頻率，不影響最終模型品質。

```bash
# 3) 用訓練好的 checkpoint 生圖
python sampling.py --ckpt_path {CKPT} --save_dir {SAVE} --num_samples 500
# 報告用的每個 predictor 8 張圖，可以用：
python sampling.py --ckpt_path {CKPT} --save_dir {SAVE_8} --num_samples 8

# 4) 算 FID（一定要先跑過 dataset.py）
python fid/measure_fid.py data/afhq/eval {SAMPLING_SAVE_DIR}
```

FID 分數對照表（報告 20 分部分）：
| FID | 分數 |
|---|---|
| < 15 | 20 |
| 15–20 | 15 |
| 20–30 | 10 |
| 30–40 | 5 |
| ≥ 40 | 0 |

如果算出來的 FID 異常大（例如投影片提到的 229 這種「錯誤 FID」），第一件事先檢查你有沒有忘記先跑 `python dataset.py` 準備好 `data/afhq/eval`。

### TODO #5 — 實驗設計

務必依照投影片規劃：
- **beta scheduler 比較**：固定用 noise predictor，跑 linear / quad / cosine 三組，比較 trajectory 圖跟最終結果。
- **predictor 比較**：固定用 linear scheduler，跑 noise / x0 / mean 三組，每組取 8 張樣本（共 24 張）放進報告討論。

### TODO #6 — Report（Task 2）

需要包含：
1. 所有 TODO 程式碼的完整解釋（30 分）。
2. 三種 beta scheduler 的 trajectory 圖比較與討論（10 分）：討論不同 schedule 對去噪過程/最終畫質的影響（例如 cosine 通常在極端 t 附近加噪較慢，訓練初期資訊保留較久）。
3. 三種 predictor 的結果圖與討論（10 分）：討論何者訓練較穩定、收斂較快、視覺品質差異，並嘗試解釋原因（例如直接預測 mean 的數值範圍隨 t 變化很大，訓練上通常比預測 noise 更不穩定）。
4. 最佳 FID 的截圖（20 分）。

---

## 4. 除錯（debug）的通用心法

1. **先驗證形狀（shape）**：任何一個 TODO 寫完後，先用小 batch 手動跑一次，`print(x.shape)` 確認跟預期一致，尤其是 `extract()` 出來的常數張量有沒有正確 broadcast。
2. **先驗證數值範圍**：例如 $\bar\alpha_t$ 應該是介於 0~1 之間、隨 t 遞減；`betas` 應該隨 t 遞增且都是正數。可以在 Python 互動模式下直接 `print` 出來檢查。
3. **善用 `pytest tests/test_todo.py -q`**（Task 2）：雖然不計分，但可以在你花 6 小時訓練之前，先確認 `scheduler.py`、`model.py` 的數值邏輯跟參考答案一致。
4. **Task 1 的視覺化是最好的檢查工具**：`q_sample` 對不對，一眼就能從散點圖看出來；訓練完的樣本疊圖對不對，也是一眼就能看出來。先把 Task 1 完全弄懂、正確，再去寫 Task 2，因為兩者的核心公式完全相同。
5. **NaN 問題**：通常來自於（a）`sqrt()` 吃到負數（数值誤差導致 $\bar\alpha_t$ 略小於 0，可以在算完 schedule 後 clip 到 `[1e-8, 1]` 附近；或 cosine schedule 沒有做 clip）、（b）忘記在 `x0_pred`／`x̂0` clamp 到 `[-1,1]`。

---

## 5. 交件前檢查清單

- [ ] Task 1 所有 TODO 完成，notebook 從頭跑到尾不報錯，Chamfer Distance < 20。
- [ ] Task 2 `scheduler.py`、`model.py` 所有 TODO 完成，`pytest tests/test_todo.py -q` 通過（或你已理解不一致的原因）。
- [ ] 至少完成 5 組訓練（linear/quad/cosine × noise，以及 linear × x0/mean）並記錄 FID、trajectory 圖。
- [ ] `python dataset.py` 有先跑過，FID 數值合理（個位數到十幾，不是兩三百）。
- [ ] 報告涵蓋所有評分項目（見上方兩個 TODO #6/#7 清單）。
- [ ] zip 檔案結構正確、不含 `data/`、`results/`、`samples/`、`afhq_inception_v3.ckpt`，檔名為 `{學號}_lab1.zip`。

祝你作業順利！把每一個 TODO 對應的公式抄一遍、親手推導一次「為什麼是這樣」，會比直接抄程式碼更容易在考試或口試時講得清楚。
