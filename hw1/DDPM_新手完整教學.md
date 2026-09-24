# DDPM 新手完整教學：從零開始

> 這份筆記假設你**只會基本 Python**，其他（機率、神經網路、PyTorch、DDPM、SSH、GPU 伺服器）都從頭講。目標是讓你先建立「直覺」，再一步一步推導到作業要求的公式，最後對應到程式碼，以及理解「為什麼要用 SSH、tmux、port forwarding 這些工具」。
>
> 這份是「**原理版**」。要實際動手時，打開 `DDPM_教學筆記.md`（「**動手版**」）：裡面有已經寫好的程式碼逐段解說、從第一次 SSH 登入到打包繳交的每一行指令。

---

# 目錄

- Part A：先修知識（機率、神經網路、PyTorch）
- Part B：DDPM 的直覺與數學，從零推導
- Part C：名詞小字典
- Part D：對應到作業程式碼（白話版）
- Part E：在實驗室電腦上跑實驗的原理（SSH、rsync、tmux、port forwarding、GPU）
- Part F：從零到繳交的完整流程總覽

---

# Part A：先修知識

## A1. 什麼是「生成模型」(generative model)？

假設你有 1000 張貓的照片。這些照片雖然長得不一樣，但都「符合某種規律」——都有兩隻眼睛、一個鼻子、毛茸茸的輪廓等等。我們可以想像存在一個「貓臉照片的機率分佈」，這 1000 張照片就是從這個分佈裡「抽樣」出來的樣本。

**生成模型**要做的事情就是：看過這 1000 張照片之後，學會這個機率分佈長什麼樣子，然後能夠自己「生出」新的、從沒看過、但看起來也像貓臉的照片。

DDPM（Denoising Diffusion Probabilistic Model）就是一種生成模型。這次作業裡：
- Task 1：資料是 2D 座標點，組成一個螺旋形狀（Swiss Roll）。目標是學會這個螺旋形狀的分佈，生成新的、也落在螺旋上的點。
- Task 2：資料是動物臉的照片（AFHQ 資料集）。目標是學會生成新的動物臉照片。

## A2. 機率分佈基礎

### 隨機變數與機率分佈

「隨機變數」就是一個會有不確定結果的量。例如丟骰子的點數就是一個隨機變數，1~6 每個數字出現的機率是多少，就叫做這個隨機變數的「機率分佈」。

在這次作業裡，我們關心的隨機變數是「一個資料點/一張圖片長什麼樣子」，它的機率分佈就是「所有可能的圖片中，哪些看起來像貓、哪些不像」這件事的統計規律。

### 常態分佈 (Normal / Gaussian distribution)

常態分佈是機器學習裡最常用的機率分佈，長得像一個鐘型曲線。它由兩個數字完全決定：
- **平均值 (mean, $\mu$)**：分佈的中心點，最有可能出現的值。
- **變異數 (variance, $\sigma^2$)**，或它的平方根**標準差 (std, $\sigma$)**：分佈的「胖瘦」，數字愈大代表數值愈分散、愈不確定。

寫作 $x \sim \mathcal{N}(\mu, \sigma^2)$，意思是「$x$ 是從平均值 $\mu$、變異數 $\sigma^2$ 的常態分佈中抽出來的一個樣本」。

**標準常態分佈**是指 $\mu=0,\ \sigma^2=1$ 的常態分佈，記作 $\mathcal N(0,1)$，是「最單純的隨機雜訊」，在程式裡對應 `torch.randn(...)`。

### 高維度的情況：向量、圖片

一張圖片其實就是一大串數字（例如 64×64 的彩色圖有 $64\times64\times3 \approx 12288$ 個數字）。DDPM 把整張圖片、或 2D 座標點，都當成一個「向量」，然後假設向量裡的每一個數字都各自獨立地服從常態分佈，共用同一個 $\mu$、同一個 $\sigma^2$（也就是「每個維度用同樣的規則加同樣強度的雜訊」）。這種假設寫作：

$$x \sim \mathcal N(\mu \mathbf{1},\ \sigma^2 I)$$

其中 $I$ 是單位矩陣，代表「維度之間互不影響、雜訊強度都一樣」。你不需要真的懂矩陣代數，只要記得：**在這份作業裡，看到 $\mathcal N(\text{某個值},\ \text{某個係數} \cdot I)$，就等於「每個數字獨立地加上同樣強度的常態雜訊」**。

### 「重參數化技巧」(reparameterization trick) —— 這是整份作業最關鍵的技巧

如果 $x \sim \mathcal N(\mu, \sigma^2)$，那麼我們可以把「抽樣」這件事拆成兩步：
1. 先從標準常態分佈抽一個雜訊 $\epsilon \sim \mathcal N(0, 1)$（也就是完全跟 $\mu,\sigma$ 無關的隨機數）。
2. 再用公式 $x = \mu + \sigma \cdot \epsilon$ 算出 $x$。

這兩種做法在統計上完全等價，但第二種寫法的好處是：$\mu$、$\sigma$ 可以是神經網路算出來的（可以微分、可以訓練），而「隨機」的部分被隔離到 $\epsilon$ 裡（不需要微分）。**DDPM 裡幾乎所有的「加噪」、「去噪取樣」步驟都是用這個技巧寫成的**，記住這個公式的形狀，後面看到很多次都是它的變形。

## A3. 神經網路基礎

### 神經網路是什麼？

你可以先不管內部細節，把神經網路想成一個**函數** $f_\theta$：
- 輸入：一些數字（例如一張圖片、一個座標點）
- 輸出：另一些數字（例如「這是不是貓」的機率、或是「這張圖被加了多少雜訊」的估計值）
- $\theta$：函數內部一大堆可以調整的參數（權重），一開始是隨機的，訓練過程會不斷調整它們，讓函數的輸出愈來愈準。

這次作業裡，網路的工作是：**輸入一張「被加了雜訊的圖片 $x_t$」和「加噪的程度 $t$」，輸出「它猜測被加進去的雜訊長什麼樣子」**（或是猜測乾淨圖片、或是猜測某個統計量，依照作業設定不同而不同）。

### Loss function（損失函數）與訓練

要讓網路變準，我們需要一個「打分數」的方式，告訴網路「你這次猜得多差」，這就是 **loss function**。訓練的目標永遠是把 loss 愈練愈小。

這次作業一律使用 **MSE（均方誤差，Mean Squared Error）**：

$$\text{MSE}(a, b) = \text{平均}\big((a-b)^2\big)$$

也就是把「猜測值」跟「正確答案」逐項相減、平方、取平均。差距愈小，loss 愈小。程式裡對應 `F.mse_loss(a, b)`。

### 訓練迴圈長什麼樣子

```
重複很多次：
    1. 從資料集裡隨機拿一批資料 (batch)
    2. 把資料丟進網路，得到網路的猜測 (forward pass)
    3. 用 loss function 算出猜測跟正確答案差多少
    4. 呼叫 loss.backward()：PyTorch 自動幫你算出「每個參數該往哪個方向調整，才能讓 loss 變小」（這叫 backpropagation，反向傳播，你不用自己推導微分）
    5. optimizer.step()：把參數依照上一步算出的方向，往「讓 loss 變小」的方向移動一小步
```

這份作業的 notebook / `train.py` 已經把整個迴圈都寫好了，你只需要負責寫「網路架構長什麼樣（forward pass）」跟「loss 怎麼算」這兩塊。

## A4. PyTorch 基礎概念

- **Tensor**：PyTorch 裡的「多維陣列」，可以想成是 numpy array 的升級版，差別是它可以在 GPU 上運算、還能自動算微分。
- **Shape**：一個 tensor 的形狀。例如 `x.shape == (128, 2)` 代表這是一批 128 筆資料，每筆是 2 個數字（Task 1 的座標點）；`x.shape == (128, 3, 64, 64)` 代表 128 張圖片，每張是 3 個 channel（RGB）、高 64、寬 64。
- **Batch dimension**：最前面那個維度，代表「這次一次處理幾筆資料」。訓練時通常一次處理一批（例如 128 筆），而不是一筆一筆處理，這樣運算比較快。
- **Broadcasting（廣播）**：當兩個 tensor 的形狀不完全一樣，但其中一個維度是 1 時，PyTorch 會自動把它「複製擴張」成一樣的形狀再做運算。例如 `(128, 1) * (128, 2)` 會自動把前者擴張成 `(128, 2)` 再逐項相乘。這在作業裡非常重要：因為 $\bar\alpha_t$ 這種「每筆資料各自有一個值」的係數，形狀通常是 `(128, 1)`，需要跟形狀 `(128, 2)` 的資料相乘，靠的就是 broadcasting，你完全不用手動 reshape。
- **`.sqrt()`**：對 tensor 逐項開根號。
- **`torch.randn_like(x)`**：產生一個跟 `x` 形狀一樣、內容是標準常態分佈亂數的 tensor（也就是抽樣 $\epsilon \sim \mathcal N(0,1)$）。
- **`@torch.no_grad()`**：告訴 PyTorch「這段程式碼不需要計算梯度」，通常用在「生成新資料 (sampling)」的時候，因為那時候不需要訓練、只需要用網路做預測，可以省記憶體、加速。
- **`.clamp(min=a, max=b)`**：把數值限制在 `[a,b]` 範圍內，超過的部分會被「夾」到邊界值。

## A5. 圖片怎麼變成 Tensor？

一張彩色圖片可以表示成形狀 `(3, H, W)` 的 tensor：3 個 channel 分別是紅、綠、藍，`H`、`W` 是高跟寬，每個數字代表該位置該顏色通道的強度。

原始像素值通常是 0~255，但神經網路訓練時，通常會先正規化到 `[-1, 1]` 這個範圍（讓數值有正有負、比較穩定好訓練）。這也是為什麼作業裡許多地方要求把預測出的「乾淨圖片」`clamp(-1, 1)`：因為訓練資料的範圍就是 `[-1,1]`，網路預測出界外的值代表它算錯了，用 clamp 硬性拉回合理範圍可以讓生成過程更穩定。

---

# Part B：DDPM 的直覺與數學，從零推導

## B1. 大方向：Diffusion Model 在做什麼？

想像一台老電視機的畫面：你把一張清晰的照片顯示出來，然後訊號干擾愈來愈強，畫面漸漸變成雪花雜訊，最後完全看不出原本的照片，變成純粹的隨機雜訊。

Diffusion Model 做的事情分兩階段：

1. **Forward process（前向 / 加噪過程）**：把一張乾淨的圖片，用一個固定的、不需要學習的規則，逐步加入雜訊，經過 $T$ 步之後（$T$ 通常是 1000），變成完全看不出原圖、跟純雜訊沒兩樣的東西。這個過程是**人為設計好的公式**，不用訓練。

2. **Reverse process（反向 / 去噪過程）**：訓練一個神經網路，讓它學會「看到某個加了雜訊的畫面，猜出上一步（雜訊少一點點）的畫面長什麼樣」。如果每一小步都猜得夠準，那麼從純雜訊開始，一步一步地反覆去噪 $T$ 次，最後就能生出一張「像真的」的乾淨圖片。

**類比**：forward process 就像是「把一塊雕像敲碎成沙子」（規則很簡單，怎麼敲都行），reverse process 則是在訓練一個雕刻家，讓他學會「看到一堆沙子，一點一點把它還原回雕像的樣子」。這個雕刻家沒辦法一步到位，只能學會「每次往正確方向修一點點」，重複很多次之後就能拼出完整的雕像。

生成新資料的方法，就是：憑空抽一堆純雜訊（沙子），叫這個訓練好的雕刻家，一步一步（重複 $T$ 次）把它修整成一張新的、之前沒看過的圖片。

## B2. Forward Process：怎麼加噪？

### 一步步加噪（概念上）

Forward process 被定義成一個馬可夫鏈（Markov chain）：每一步只跟前一步有關，跟更之前的狀態無關。

$$q(x_t \mid x_{t-1}) = \mathcal N\big(\sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\big)$$

白話翻譯：**在第 $t$ 步，我們把上一步的圖片 $x_{t-1}$ 稍微縮小一點（乘上 $\sqrt{1-\beta_t}$，這是一個接近 1 的數字），再加入一點點強度為 $\beta_t$ 的雜訊**。

$\beta_t$（beta）叫做「加噪排程」(noise schedule)，是一串**事先設計好、不需要訓練**的小數字，通常隨著 $t$ 增加而慢慢變大（也就是：一開始加很少雜訊，後面加愈來愈多）。作業裡的 3 種 schedule（linear / quadratic / cosine）就是 3 種不同的 $\beta_t$ 遞增規則，我們在 Part D 會細講。

我們定義 $\alpha_t = 1 - \beta_t$（一個接近 1、代表「這一步保留了多少原始訊號」的數字）。

### 「一步到位」公式：直接跳到任意時刻 $t$

如果照著上面的公式，你想知道加噪 500 步之後的樣子，理論上要真的迴圈跑 500 次。但 DDPM 有一個很漂亮的數學性質：**因為每一步都是常態分佈疊加常態分佈，而常態分佈疊加常態分佈仍然是常態分佈**，所以可以直接推導出，從 $x_0$（原圖）一步跳到任意時刻 $t$ 的公式：

$$q(x_t \mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t)\, I\big)$$

其中 $\bar\alpha_t = \alpha_1 \times \alpha_2 \times \cdots \times \alpha_t$（把每一步的 $\alpha$ 全部乘起來，程式裡叫 `alphas_cumprod`，cumprod = cumulative product = 累積乘積）。

**直覺理解**：$\bar\alpha_t$ 代表「經過 $t$ 步之後，原圖訊號還剩下多少比例」。因為每一步的 $\alpha_i$ 都小於 1，愈乘愈小，所以 $t$ 愈大，$\bar\alpha_t$ 愈接近 0，圖片裡「原圖的成分」愈來愈少、「雜訊的成分」愈來愈多。當 $t=T$（例如 1000）夠大時，$\bar\alpha_T$ 幾乎是 0，這時候 $x_T$ 幾乎完全是雜訊，跟原圖沒有任何關係了——這正是我們想要的效果。

再用 A2 提到的「重參數化技巧」把這個常態分佈抽樣寫成公式：

$$\boxed{x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\qquad \epsilon \sim \mathcal N(0, I)}$$

**這行公式是整份作業裡最重要、出現最多次的一行**，請務必記熟。白話：**把原圖乘上一個係數（隨 $t$ 增大而變小），再加上乘了另一個係數的純雜訊（隨 $t$ 增大而變大）**。$t=0$ 時 $\bar\alpha_0\approx1$，幾乎就是原圖；$t$ 很大時 $\bar\alpha_t\approx0$，幾乎全是雜訊。

Task 1 的 `q_sample` 和 Task 2 的 `add_noise`，做的就是把這一行公式寫成 PyTorch 程式碼。

## B3. 為什麼訓練目標是「預測雜訊」？

現在問題來了：我們想訓練網路做「去噪」，那網路到底應該輸出什麼東西？

直覺上有幾種可能的設計：
- **方案 A**：網路直接輸出「猜測的乾淨圖片 $\hat x_0$」。
- **方案 B**：網路輸出「猜測被加進去的雜訊 $\hat\epsilon$」。
- **方案 C**：網路直接輸出「下一步要用的平均值 $\hat\mu$」（更抽象一點的統計量）。

由於 forward process 的公式是 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$，這三種其實都是「同一件事」的不同表達方式——只要知道其中一個，配合已知的 $x_t$、$t$，就可以反推出另外兩個。作業裡的 Task 2 剛好要求你三種都實作看看，比較差異。

DDPM 原始論文發現，**訓練網路去預測雜訊 $\epsilon$（方案 B）**，在數學上等價於訓練它最大化資料的機率（technically 是 variational lower bound），而且訓練起來最穩定、效果最好，所以這是最常見的預設做法。直覺上的理由：雜訊 $\epsilon$ 本身永遠是標準常態分佈（數值範圍、統計特性都固定），比起要猜測「內容差異很大的原始圖片」或「數值範圍隨 $t$ 劇烈變化的平均值」，這是一個範圍穩定、比較容易學習的目標。

## B4. Loss Function：怎麼訓練這個網路？

訓練方式非常直接：

1. 拿一張真實圖片 $x_0$。
2. 隨機選一個時間點 $t$（介於 $1$ 到 $T$ 之間）。
3. 隨機抽一個雜訊 $\epsilon$，用 B2 的公式算出 $x_t$（也就是「假裝」這張圖已經被加噪到第 $t$ 步了）。
4. 把 $(x_t, t)$ 丟進網路，得到網路的猜測 $\hat\epsilon_\theta(x_t, t)$。
5. 因為我們在第 3 步是自己動手加的雜訊，所以「正確答案」$\epsilon$ 我們是知道的！用 MSE 比較網路猜的 $\hat\epsilon_\theta$ 跟真正的 $\epsilon$：

$$\mathcal L = \mathbb E\Big[\ \|\hat\epsilon_\theta(x_t,t) - \epsilon\|^2\ \Big]$$

**這就是整個訓練唯一用到的 loss**。注意這個過程完全不需要人工標註答案——因為雜訊是我們自己加的，所以「正確答案」永遠是已知的。這也是為什麼 diffusion model 可以用大量沒有標籤的圖片訓練（unsupervised）。

## B5. Reverse Process：怎麼一步步去噪、生成新圖片？

訓練好網路（它現在很會猜「這張圖被加了什麼雜訊」）之後，我們要用它來**生成新圖片**。想法是：

- 先隨機抽一張純雜訊 $x_T \sim \mathcal N(0, I)$（想像成一堆沙子）。
- 對 $t = T, T-1, \dots, 1$ 依序做：用網路猜出 $x_t$ 裡的雜訊，藉此估計出「雜訊少一點點的版本」$x_{t-1}$。
- 重複 $T$ 次之後，$x_0$ 就是生成出來的新圖片。

DDPM 論文證明，每一步的「去噪分佈」$p_\theta(x_{t-1}\mid x_t)$，也可以寫成一個常態分佈：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal N\big(\mu_\theta(x_t, t),\ \tilde\beta_t I\big)$$

這裡有兩個東西要算：**平均值 $\mu_\theta$**（去噪之後最可能的樣子）跟**變異數 $\tilde\beta_t$**（這一步還殘留多少不確定性）。

### Posterior variance（後驗變異數）$\tilde\beta_t$

這是一個固定公式（不需要網路），代表「如果我們同時知道 $x_t$ 又知道真正的 $x_0$，$x_{t-1}$ 還會有多少不確定性」：

$$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$$

你不需要理解這條公式怎麼推導出來的（牽涉貝氏定理），只要知道：這是一個由 $\alpha, \beta, \bar\alpha$ 組合出來、每個時刻都固定的數字，直接代公式算就好。

### Posterior mean（後驗平均值）$\mu_\theta$

如果網路是「預測雜訊」的版本（B3 的方案 B），可以推導出（DDPM 論文 Eq. 11）：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon_\theta(x_t,t)\right)$$

**直覺理解**：括號裡是「把 $x_t$ 減掉一部分猜測的雜訊」，外面除以 $\sqrt{\alpha_t}$ 是把 forward process 那個「縮小」的效果「放大回來」（因為 forward process 每一步都會把訊號乘上 $\sqrt{\alpha_t}$ 縮小一點，去噪自然要把它放大回來）。

### 抽樣（真正得到 $x_{t-1}$）

算出 $\mu_\theta$ 和 $\tilde\beta_t$ 之後，一樣用「重參數化技巧」抽樣：

$$x_{t-1} = \mu_\theta(x_t, t) + \sqrt{\tilde\beta_t}\, z,\qquad z \sim \mathcal N(0, I)$$

**唯一的例外**：最後一步（程式裡時間是從 0 開始編號，所以是 `t == 0` 那一步）要輸出最終結果，我們不希望結果裡還殘留隨機性，所以這一步**不加雜訊**，直接讓最終輸出等於 $\mu_\theta$。

把這個過程重複 $T$ 次（從 $t=T$ 到 $t=1$），就是 DDPM 論文 Algorithm 2（Sampling）的完整流程，也是作業裡 `p_sample`（單步）跟 `p_sample_loop`（整個迴圈）要實作的內容。

### 用 $\hat x_0$ 或 $\hat\mu$ 直接參數化的版本

上面是「網路預測雜訊」版本的推導。如果網路改成直接預測 $\hat x_0$，那麼可以先用網路輸出的 $\hat x_0$，代入下面這個「用 $x_0$ 表示的後驗平均值」公式（這其實跟上面是同一條公式，只是變數代換過）：

$$\tilde\mu(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t$$

如果網路直接輸出 $\hat\mu_\theta$（第三種參數化），那就更簡單，網路輸出的就是 $\mu_\theta$，直接拿去抽樣，不需要再做任何轉換。

---

# Part C：名詞小字典

| 名詞 | 意思 |
|---|---|
| $x_0$ | 乾淨的原始資料（圖片或座標點） |
| $x_t$ | 在第 $t$ 步、已經被加了一些雜訊的資料 |
| $x_T$ | 加噪到最後一步，幾乎完全是純雜訊 |
| $T$ | 總共的擴散步數（作業裡設為 1000） |
| $t$ | 目前的時間步（介於 0 ~ T） |
| $\epsilon$（epsilon） | 標準常態分佈的雜訊，$\epsilon\sim\mathcal N(0,I)$ |
| $\beta_t$（beta） | 第 $t$ 步加噪的強度，由 noise schedule 決定 |
| $\alpha_t$（alpha） | $=1-\beta_t$，代表這一步「保留原訊號」的比例 |
| $\bar\alpha_t$（alpha bar） | $=\alpha_1\alpha_2\cdots\alpha_t$，代表到第 $t$ 步為止「原圖訊號總共剩下多少」 |
| `alphas_cumprod` | 程式裡 $\bar\alpha_t$ 的變數名（cumulative product，累積乘積） |
| forward process / $q$ | 加噪過程，規則固定、不用訓練 |
| reverse process / $p_\theta$ | 去噪過程，用神經網路學習 |
| $\hat\epsilon_\theta(x_t,t)$ | 網路對「雜訊」的預測 |
| $\hat x_\theta(x_t,t)$ | 網路對「乾淨圖片」的預測 |
| $\mu_\theta(x_t,t)$ | 去噪一步之後，網路預測的平均值 |
| posterior mean / variance | 「後驗平均值/變異數」，去噪那一步高斯分佈的參數 |
| $\tilde\beta_t$ | posterior variance 的符號 |
| noise schedule | $\beta_t$ 隨 $t$ 變化的規則（linear／quadratic／cosine） |
| predictor | 網路實際輸出的東西是什麼（noise／x0／mean） |
| sampling | 用訓練好的模型生成新資料的過程 |
| checkpoint (`.ckpt`) | 儲存訓練好的網路參數的檔案 |
| FID (Fréchet Inception Distance) | 一個「生成圖片好不好」的分數，數字愈小代表生成的圖片跟真實圖片的分佈愈接近 |
| Chamfer Distance | Task 1（2D 點）用來衡量生成點跟目標點分佈接近程度的指標，愈小愈好 |
| `extract(consts, t, x)` | 作業提供的工具函式：從一個長度為 $T$ 的常數表（例如整條 $\bar\alpha$ 表）裡，依照 batch 裡每一筆資料各自的 $t$，取出對應的值，並且自動 reshape 成方便 broadcasting 的形狀 |

---

# Part D：對應到作業程式碼（白話版）

這一節把 Part B 的公式對應到程式碼，只講「為什麼」。**所有 TODO 都已經實作完成並驗證過**（Task 1 Chamfer Distance 11.9、Task 2 測試 28/28 通過），逐行解說請看 `DDPM_教學筆記.md` 第 1 節。

## D1. Task 1（Swiss Roll，2D 點）在做什麼

資料是 2D 平面上的點（`shape=(B, 2)`，`B` 是 batch size）。

- **`SimpleNet`**：就是 Part A3 講的那個神經網路 $f_\theta$，輸入是 `(x, t)`，輸出是「猜測的雜訊」，形狀跟輸入的 `x` 一樣 `(B, 2)`。作業提供了 `TimeLinear` 這個積木，它同時吃「座標」跟「時間 t」，我們疊了 4 層（2→128→128→128→2），中間穿插非線性函數 SiLU。沒有非線性函數的話，疊再多層也等於一層線性函數，畫不出彎曲的螺旋。
- **`q_sample`**：Part B2 那行最重要的公式 $x_t = \sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。
- **`p_sample`**：Part B5 的「單步去噪」。
- **`p_sample_loop`**：把 `p_sample` 從 t=999 呼叫到 t=0，共 1000 次。
- **`compute_loss`**：Part B4 的訓練 loss，四個步驟：隨機選 t → 加噪 → 網路猜雜訊 → MSE。

## D2. Task 2（圖片生成）多了什麼

觀念完全一樣，只是資料從 `(B, 2)` 的座標點換成 `(B, 3, 64, 64)` 的圖片，網路從小 MLP 換成 UNet（已經寫好），並且多了兩個「選項」讓你比較。

### 為什麼要比較 3 種 noise schedule？

$\beta_t$ 決定「每一步加多少雜訊」，進而決定 $\bar\alpha_t$（原圖訊號剩多少）隨時間怎麼下降：

- **Linear**：$\beta_t$ 從 0.0001 均勻增加到 0.02。$\bar\alpha_t$ 在前半段就掉得很快，後面好幾百步幾乎都是「純雜訊→純雜訊」，對模型來說有點浪費。
- **Quadratic**：$\beta_t$ 一開始增加得比較慢，後面加速。前段保留訊號比較久，但 $\bar\alpha_T$ 不夠接近 0（約 0.0007），最後一步還殘留一點原圖。
- **Cosine**（Nichol & Dhariwal, 2021）：直接設計 $\bar\alpha_t$ 的形狀讓它像餘弦曲線一樣平緩地從 1 降到 0，於是每個時間步的「雜訊程度」分佈得比較平均，模型每一步都學得到有用的東西，通常品質更好。

**這只是換一種方式決定 $\beta_t$ 這串數字，後面所有用到 $\beta_t,\alpha_t,\bar\alpha_t$ 的公式（加噪、去噪）完全不變。**

### 為什麼要比較 3 種 predictor？

如同 Part B3，網路可以輸出「雜訊」、「乾淨圖片」或「平均值」，數學上都能達到同樣目標，但實務上難度差很多：

- **noise**：目標永遠是標準常態分佈，不管 t 是多少，尺度都一樣，最好學。
- **x0**：t 很大時 $x_t$ 幾乎是純雜訊，網路根本猜不出原圖，MSE 會讓它輸出「所有可能圖片的平均」→ 模糊、糊成色塊（投影片範例中間那張白色色塊就是這樣）。
- **mean**：$\tilde\mu$ 和 $x_t$ 本身非常接近（係數 $\frac{1}{\sqrt{\alpha_t}}\approx 1$），網路要學的「有用訊號」只是其中一個很小的修正量，MSE 很容易被「直接輸出 $x_t$」這種偷懶答案主導，細節學不好。

作業要你三種都訓練、放圖比較，就是在親身驗證「為什麼大家都選預測雜訊」。

## D3. 為什麼要先跑 `pytest tests/test_todo.py`？

訓練一次要好幾個小時，如果程式邏輯寫錯，跑完才發現結果是一團雜訊，非常浪費時間。這個測試用固定的小輸入，一秒內檢查 `scheduler.py`、`model.py` 的數值是否跟助教參考答案一致，等於「上路前先檢查煞車」。不計分，但強烈建議。

## D4. 為什麼 FID 分數能衡量生成品質？

拿一個事先訓練好的圖片辨識網路（Inception，就是 `fid/afhq_inception_v3.ckpt`），把「真實圖片」跟「生成圖片」分別丟進去，取出中間層的特徵向量，比較兩組特徵的統計分佈（平均值、共變異數）差多少。生成的圖片越像真實圖片、種類越多樣，兩組分佈就越接近，FID 越低。

這也解釋了兩件事：
- **為什麼一定要先跑 `python dataset.py`**：它會建立 `data/afhq/eval/` 這個「真實圖片」資料夾。沒有它，比較的基準就錯了，FID 會變成兩三百。
- **為什麼要生成 500 張**：統計量（平均、共變異數）需要足夠多的樣本才準。

## D5. 為什麼訓練要這麼久？

每一次訓練迭代只看 16 張圖、只練一個隨機的時間步 t。要讓 UNet 在 1000 個時間步都學好，需要 5 萬～10 萬次迭代。而且**取樣**也很慢：生成一張圖要跑 UNet 1000 次（每個時間步一次）。這就是為什麼 `--log_interval` 設太小會拖慢訓練──每次 log 都要完整生成好幾張圖。

---

# Part E：在實驗室電腦上跑實驗的原理

你的 Mac 沒有 NVIDIA GPU，Task 2 如果用 CPU 跑，一組可能要好幾天。實驗室電腦（伺服器）有好幾張 GPU，但它放在機房，沒有螢幕、鍵盤給你用。所以我們要**從 Mac 遠端控制它**。這一部分解釋每個工具「在做什麼、為什麼需要」。

## E1. SSH：遠端操作另一台電腦

**SSH（Secure Shell）** 是一個程式，讓你在自己電腦的終端機裡，打開**另一台電腦**的終端機。

```
 你的 Mac                                   實驗室電腦（伺服器）
┌──────────────┐     加密的網路連線       ┌──────────────────────┐
│ ssh（客戶端）│ ───────────────────────► │ sshd（伺服器端程式） │
│ 你打的指令   │ ◄─────────────────────── │ 真正執行指令、有 GPU │
└──────────────┘     指令輸出傳回來       └──────────────────────┘
```

- 你在 `ssh lab` 之後打的每個字，都會被加密後送到伺服器執行，輸出再送回來顯示。**程式是跑在伺服器上、用伺服器的 CPU/GPU/硬碟**，你的 Mac 只是一個「遙控器 + 螢幕」。
- 所以伺服器上的檔案跟你 Mac 上的是**兩份不同的東西**，要用 E2 的工具互相搬。

### 為什麼用「金鑰」而不是密碼？

`ssh-keygen` 會產生一對檔案：
- **私鑰** `~/.ssh/id_ed25519`：只留在你的 Mac，絕對不能給別人。
- **公鑰** `~/.ssh/id_ed25519.pub`：可以公開，`ssh-copy-id` 會把它放進伺服器的 `~/.ssh/authorized_keys`。

登入時，伺服器用公鑰出一道「只有持有對應私鑰的人才解得開」的數學題，你的 Mac 用私鑰解題證明身分。整個過程私鑰不會離開你的電腦，比每次打密碼更安全，也更方便（不用再打密碼，rsync、port forwarding 都能自動登入）。

### `~/.ssh/config` 在做什麼？

只是「通訊錄」。寫了 `Host lab` 之後，`ssh lab` 就等於 `ssh -p 22 <帳號>@<實驗室IP>`。所有使用 SSH 的工具（`rsync`、`scp`、VS Code Remote-SSH）都會讀這個檔，所以之後到處都能用 `lab` 這個縮寫。`ServerAliveInterval 60` 則是每 60 秒送一個小封包，避免閒置太久被網路設備斷線。

## E2. rsync：在兩台電腦之間搬檔案

```bash
rsync -avz 來源/ 目的地/
```

- `rsync` 透過 SSH 連線傳檔案，所以同樣可以用 `lab:路徑` 表示「伺服器上的路徑」。
- 它會**比對兩邊**，只傳有差異的部分。第二次同步時只傳你改過的檔案，非常快。
- `-a`：保留資料夾結構、權限、時間；`-v`：列出傳了什麼；`-z`：傳輸時壓縮；`--exclude`：跳過某些檔案（例如幾 GB 的資料集和 checkpoint）。
- **來源結尾有沒有 `/` 意思不同**：`Lab1-DDPM/` 是「資料夾裡面的內容」，`Lab1-DDPM` 是「資料夾本身」。

在這份作業中：程式碼 **Mac → 伺服器**；結果圖片、執行過的 notebook **伺服器 → Mac**（寫報告、打包都在 Mac 上）。

## E3. tmux：斷線也不會停的終端機

**問題**：你在 SSH 裡執行 `python train.py`，這個程式是你那個 SSH 連線的「子程序」。一旦連線斷掉（闔上筆電、Wi-Fi 切換、網路不穩），系統會送一個「掛斷」訊號（SIGHUP）給這個連線底下的所有程式，訓練就會被中止。Task 2 一次要跑好幾個小時，不可能一直開著電腦不斷線。

**解法**：`tmux` 是一個跑在**伺服器上**的「終端機管理員」。

```
沒有 tmux：  SSH 連線 ── python train.py        （連線斷 → 程式死）

有 tmux：    tmux 伺服器（一直活在實驗室電腦上）
               └─ session "ddpm"
                    ├─ window 0: python train.py ...linear noise
                    ├─ window 1: python train.py ...quad noise
                    └─ window 2: jupyter lab
             SSH 連線 ──(attach/detach)──► 只是「看」這個 session 的視窗
```

- `tmux new -s ddpm`：建立 session，程式是 tmux 的子程序，不是 SSH 連線的。
- `Ctrl+b d`（detach）：離開畫面，但 session 和裡面的程式繼續跑。
- 下次 `ssh lab` 再 `tmux attach -t ddpm`，就回到離開時的畫面，進度條還在跑。

所以原則是：**所有會跑超過幾分鐘的東西，都放在 tmux 裡跑。**

## E4. Port forwarding：在 Mac 瀏覽器看伺服器上的 Jupyter

Jupyter Lab 是一個**網頁伺服器**：它在實驗室電腦上開一個「埠號（port）」8888，等瀏覽器連進來。但：

- 這個 8888 只對伺服器「自己」開放（`localhost`），外面連不到，也不應該連得到（不安全）。
- 你的瀏覽器在 Mac 上。

`ssh -N -L 8888:localhost:8888 lab` 的意思是：

```
Mac 瀏覽器 ──► Mac 的 localhost:8888 ══(SSH 加密隧道)══► 伺服器 ──► 伺服器的 localhost:8888（Jupyter）
```

- `-L 本機埠:目標主機:目標埠`：把「Mac 的 8888」接到「從伺服器角度看的 localhost:8888」。
- `-N`：不開終端機、只做轉送，所以指令看起來像「卡住」，其實它在工作。
- 瀏覽器以為自己在連 Mac 本機的網站，但流量都被 SSH 加密送到伺服器上的 Jupyter。

Jupyter 印出的 **token** 就是密碼，避免同一台伺服器上的其他使用者連進你的 Jupyter。

（VS Code Remote-SSH 在背後做的其實是同一件事，只是自動幫你處理好。）

## E5. GPU、CUDA、nvidia-smi、conda

- **GPU**：擅長「同時做大量相同的簡單運算」，剛好就是神經網路的矩陣乘法。同樣的訓練，GPU 可以比 CPU 快幾十倍。
- **CUDA**：NVIDIA 讓程式使用 GPU 的軟體介面。PyTorch 用 `.to("cuda:0")` 把資料和模型搬到第 0 張 GPU 上。
- **`nvidia-smi`**：顯示每張 GPU 的編號、記憶體用量、使用率、誰在用。實驗室是多人共用，**跑之前一定要看**，挑空的卡，避免把別人擠爆（`CUDA out of memory`）。
- **驅動版本 vs PyTorch 版本**：`nvidia-smi` 右上角的 `CUDA Version` 是驅動「最高支援」的版本。pip 裝的 PyTorch 如果是用更新的 CUDA 編譯的，就會 `torch.cuda.is_available() == False`。解法是裝指定 CUDA 版本的 PyTorch（手冊 3.3）。
- **`--gpu N` 和 `CUDA_VISIBLE_DEVICES=N`**：前者是作業程式自己的參數，會用 `cuda:N`；後者是環境變數，讓程式「只看得到」第 N 張卡（而且在程式裡它會變成 `cuda:0`）。`measure_fid.py` 沒有 `--gpu` 參數，所以用後者指定。
- **conda 環境**：伺服器上可能有很多人、很多專案需要不同版本的套件。`conda create -n ddpm python=3.9` 建一個獨立的「套件盒子」，`conda activate ddpm` 進入它，裝的東西不會影響別人或其他專案。**每開一個新的終端機 / tmux window 都要重新 activate。**

## E6. 為什麼要用相對路徑、`cd` 到正確資料夾？

`train.py` 寫的是 `AFHQDataModule("./data", ...)`、存檔到 `results/...`；notebook 用 `from dataset import ...`。這些都是**相對於「目前所在資料夾」**的路徑。所以：

- Task 2 所有指令都要先 `cd ~/Lab1-DDPM/image_diffusion_todo`。
- Task 1 的 Jupyter 要在 `~/Lab1-DDPM/2d_plot_diffusion_todo` 裡啟動。

在錯的資料夾執行，就會出現 `ModuleNotFoundError` 或「又下載一次資料集」。

---

# Part F：從零到繳交的完整流程總覽

```
[Mac]  1. ssh-keygen → ssh-copy-id → 寫 ~/.ssh/config          （E1，只做一次）
[Mac]  2. rsync 程式碼到 lab                                    （E2）
[lab]  3. conda 建環境、pip install、確認 cuda True、pytest 28 passed   （E5）
[lab]  4. tmux new -s ddpm                                      （E3）
[lab]  5. python dataset.py（下載 AFHQ + 建 eval 資料夾）         （D4）
[lab]  6. 挑空 GPU，開始 5 組 train.py（各一個 tmux window）       ← 最花時間，越早越好
[lab]  7. 等待期間：另開 window 跑 jupyter lab                    （E4）
[Mac]  8. ssh -N -L 8888:... → 瀏覽器跑 Task 1 notebook、存 3 張圖
[Mac]  9. rsync 拿回 Task 1 圖和 notebook
[lab] 10. 每組訓練完：sampling.py（500 張 + 8 張）→ measure_fid.py → 截圖 FID
[Mac] 11. rsync 拿回 samples、traj 圖、loss 圖
[Mac] 12. 寫 report.pdf → 打包 zip → 檢查沒夾帶 data/results/samples/ckpt
```

每一步的實際指令都在 `DDPM_教學筆記.md` 對應章節。

---

# 建議的讀法

1. 先把 Part A、Part B 從頭讀一遍，弄懂「forward process 在幹嘛、reverse process 在幹嘛、loss 為什麼這樣設計」。
2. 讀 Part E，知道 SSH、tmux、port forwarding 各解決什麼問題，之後照手冊打指令時才知道自己在做什麼、出錯時知道從哪裡查。
3. 打開 `DDPM_教學筆記.md`，**先照第 2～5 節把訓練丟上 GPU**，再邊等邊看第 1 節的程式碼解說、跑 Task 1。
4. 寫報告時，Part D 的「為什麼」＋手冊第 1 節的「公式 + 程式碼」＋第 8 節的清單，就是報告的骨架。

DDPM 的數學一開始看起來嚇人，但整份作業從頭到尾只反覆用了三條公式（B2 的加噪公式、B5 的去噪公式、B4 的 loss）。SSH 那一套也一樣，只有四個動作：**連上去（ssh）、搬檔案（rsync）、讓程式在背景跑（tmux）、把網頁接回來（port forwarding）**。把這幾樣真正搞懂，其他都是重複套用而已。
