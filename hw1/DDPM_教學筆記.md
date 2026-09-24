# Lab1 - DDPM 實作對照 + SSH 實戰手冊

> 這份是「**動手版**」：告訴你程式碼每一行在做什麼、怎麼用 SSH 在實驗室電腦上把**所有**結果跑出來、以及報告要放什麼。
> 如果你看到某個公式或名詞不懂，回去查 `DDPM_新手完整教學.md`（那份是「**原理版**」，Part A～D 講 DDPM 數學，Part E 講 SSH / tmux / GPU 的原理）。

- 截止日期：**2026/10/1 (四) 23:59**
- 繳交：`{學號}_lab1.zip`，內含 `report.pdf`、`2d_plot_diffusion_todo/`、`image_diffusion_todo/`
  （**不要**附上 `data/`、`results/`、`samples/`、`fid/afhq_inception_v3.ckpt`）

---

## 0. 目前狀態總覽（先看這裡）

| 項目 | 狀態 | 驗證方式 |
|---|---|---|
| Task 1 全部 TODO（`network.py`、`ddpm.py`） | ✅ 已實作 | 本機 CPU 跑 5000 iters：Chamfer Distance = **11.9**（門檻 < 20）、q_sample 圖與投影片一致 |
| Task 2 全部 TODO（`scheduler.py`、`model.py`） | ✅ 已實作 | `pytest tests/test_todo.py -q` → **28 passed**（與助教參考答案數值一致） |
| Task 2 端到端 | ✅ 已驗證 | 5 種 (schedule, predictor) 組合都能訓練、取樣、存檔、讀檔，無 NaN |
| Task 1 notebook 正式跑一次並存圖 | ⬜ 你要做 | 第 4 節 |
| Task 2 五組訓練 + 取樣 + FID | ⬜ 你要做（最花時間） | 第 5～7 節 |
| 報告 + 打包 | ⬜ 你要做 | 第 8～9 節 |

**時間規劃**：今天 9/24，截止 10/1。一組訓練約 4～8 小時，總共 5 組。**第一件事就是把訓練丟上 GPU**，等訓練的同時再跑 Task 1、寫報告。

---

## 1. 程式碼實作對照（報告「解釋所有 TODO」要用）

報告需要「完整解釋 TODO 實作」（Task 1 20 分、Task 2 30 分）。下面每一段都可以直接改寫成報告內容：**公式 → 程式碼 → 為什麼**。

### Task 1（`2d_plot_diffusion_todo/`）

#### TODO #1 `SimpleNet`（`network.py:86-111`）

```python
dims = [dim_in] + list(dim_hids) + [dim_out]            # [2, 128, 128, 128, 2]
self.layers = nn.ModuleList(
    [TimeLinear(dims[i], dims[i + 1], num_timesteps) for i in range(len(dims) - 1)]
)
self.act = nn.SiLU()
...
for i, layer in enumerate(self.layers):
    x = layer(x, t)
    if i < len(self.layers) - 1:   # 最後一層不加 activation
        x = self.act(x)
```

- 4 層 `TimeLinear`：2→128→128→128→2。每一層都會把 timestep `t` 編碼成向量，乘到線性層輸出上，讓網路知道「現在雜訊有多強」。
- 用 `nn.ModuleList` 而不是 `nn.Sequential`：因為每一層都要多吃一個 `t`，`Sequential` 只能傳一個參數。
- 中間層加 `SiLU`（平滑版 ReLU，`TimeEmbedding` 裡也是用它）；**輸出層不加**，因為預測的雜訊 $\epsilon$ 可正可負、沒有範圍限制。

#### TODO #2 `q_sample`（`ddpm.py:90`）

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

```python
xt = alphas_prod_t.sqrt() * x0 + (1 - alphas_prod_t).sqrt() * noise
```

`extract()` 已經把 $\bar\alpha_t$ reshape 成 `(B, 1)`，跟 `(B, 2)` 的 `x0` 自動 broadcasting。

#### TODO #3 `p_sample`（`ddpm.py:121-133`）

```python
eps_theta = self.network(xt, t)                                   # 1. 預測雜訊
mean = (xt - eps_factor * eps_theta) / alpha_t.sqrt()             # 2. Eq.11 posterior mean
var = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t         # 3. posterior variance
z = torch.randn_like(xt)
nonzero_mask = (t != 0).float().view(-1, *([1] * (xt.dim() - 1)))
x_t_prev = mean + nonzero_mask * var.sqrt() * z                   # 4. t=0 時不加雜訊
```

- `eps_factor` 是 starter code 給的 $\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}$，因為 $1-\alpha_t=\beta_t$，它就是 Eq. 11 裡的 $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$。
- `nonzero_mask`：最後一步（t=0）要輸出乾淨結果，不能再加隨機雜訊。

#### TODO #4 `p_sample_loop`（`ddpm.py:152-154`）

```python
for t in range(self.var_scheduler.num_train_timesteps - 1, -1, -1):   # 999, 998, ..., 0
    xt = self.p_sample(xt, t)
x0_pred = xt
```

就是 DDPM Algorithm 2：從純雜訊 $x_T$ 開始，連續去噪 1000 次。

#### TODO #5 `compute_loss`（`ddpm.py:243-250`）

```python
eps = torch.randn_like(x0)                  # 真正加進去的雜訊（正確答案）
x_t = self.q_sample(x0, t, noise=eps)       # 加噪到第 t 步
eps_pred = self.network(x_t, t)             # 網路猜雜訊
loss = F.mse_loss(eps_pred, eps)            # DDPM Eq.14 簡化 loss
```

### Task 2（`image_diffusion_todo/`）

#### TODO #1 `add_noise`（`scheduler.py:256-257`）

跟 Task 1 `q_sample` 同一條公式，只是張量變成 `[B, C, H, W]`，`extract()` 會 reshape 成 `[B,1,1,1]`。

```python
alpha_bar_t = extract(self.alphas_cumprod, t, x_0)
x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps
```

#### TODO #2 cosine schedule（`scheduler.py:47-52`）

$$f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big),\quad \bar\alpha_t=\frac{f(t)}{f(0)},\quad \beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}},\quad s=0.008$$

```python
s = 0.008
steps = torch.arange(num_train_timesteps + 1, dtype=torch.float64)    # t = 0..T，共 T+1 個點
f = torch.cos(((steps / num_train_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
alphas_cumprod = f / f[0]                                              # normalize 讓 ᾱ_0 = 1
betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]                   # 相鄰比值 → T 個 β
betas = betas.clamp(max=0.999).float()                                 # 避免 t=T 附近 β→1 爆掉
```

- 要算 $T+1$ 個點，因為每個 $\beta_t$ 需要 $\bar\alpha_t$ 和 $\bar\alpha_{t-1}$ 兩個點。
- 用 `float64` 算再轉回 `float32`：$t$ 接近 $T$ 時 $\bar\alpha$ 非常小，雙精度避免相除誤差。

#### TODO #3 三種 predictor 的 `step`（`scheduler.py:125-222`）

三種 predictor 共用同一組「後驗分佈」計算，所以抽成 4 個小 helper：

| helper | 公式 |
|---|---|
| `_alpha_bar_prev` | $\bar\alpha_{t-1}$，且 **t=0 時定義為 1**（TODO 註解明確要求） |
| `_posterior_mean` | $\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\hat x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$ |
| `_posterior_variance` | $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ |
| `_add_posterior_noise` | $x_{t-1}=\text{mean}+\mathbb 1[t\neq0]\sqrt{\tilde\beta_t}\,z$ |

- **`step_predict_noise`**：先把預測的雜訊換成 $\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}}$，`clamp(-1,1)`，再代入 $\tilde\mu_t$、$\tilde\beta_t$ 抽樣。
- **`step_predict_x0`**：網路直接給 $\hat x_0$，`clamp(-1,1)` 後走一樣的流程。
- **`step_predict_mean`**：網路直接給 $\mu_\theta$，只要算 $\tilde\beta_t$ 加雜訊。

> 為什麼要 `clamp(-1, 1)`？訓練圖片被 normalize 到 $[-1,1]$（`dataset.py` 的 `Normalize(0.5, 0.5)`），網路預測超出範圍就是錯的，強制拉回來可以讓取樣穩定很多。
>
> 小差異：Task 1 的 `p_sample` 是 starter code 寫好的 `t_prev.clamp(min=0)`（t=0 時拿 $\bar\alpha_0$），Task 2 照 TODO 註解用 $\bar\alpha_{-1}=1$。兩者只影響 t=0 那一步的 variance，而那一步本來就不加雜訊，所以結果幾乎一樣。

#### TODO #4 loss（`model.py:25-58`）

| predictor | 網路輸出 | 正確答案 |
|---|---|---|
| noise（已給） | $\hat\epsilon_\theta$ | `eps` |
| x0 | $\hat x_\theta$ | `x0` |
| mean | $\mu_\theta$ | $\tilde\mu=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\big)$ |

`get_loss_mean` 的正確答案用真正加進去的 `eps` 算（不是網路預測的），這個形式和 $\tilde\mu(x_t,x_0)$ 的原始公式數學上完全相等（把 $x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}$ 代進去化簡即可）。

---

## 2. 第一次連上實驗室電腦（在你的 Mac 上做）

下面 `<帳號>`、`<實驗室IP>` 請換成實驗室給你的資訊（問學長姐或管理員）。以下指令前面的 `mac$` 代表「在你的 Mac 終端機打」，`lab$` 代表「在實驗室電腦上打」（不用打出這個前綴）。

### 2.1 產生 SSH 金鑰（只做一次）

```bash
mac$ ls ~/.ssh/id_ed25519.pub        # 已經有就跳過下一行
mac$ ssh-keygen -t ed25519 -C "chan.john.1027@gmail.com"   # 一路 Enter 即可
```

### 2.2 把公鑰放到實驗室電腦（只做一次，之後不用打密碼）

```bash
mac$ ssh-copy-id <帳號>@<實驗室IP>     # 會問一次實驗室密碼
```

### 2.3 設定縮寫（只做一次）

編輯 `~/.ssh/config`（沒有就新建），加入：

```
Host lab
    HostName <實驗室IP>
    User <帳號>
    Port 22
    ServerAliveInterval 60
```

之後只要 `ssh lab` 就能登入。如果實驗室要先經過一台跳板機（gateway），再加一行 `ProxyJump <跳板機帳號>@<跳板機IP>`。

### 2.4 登入並偵察環境

```bash
mac$ ssh lab
lab$ nvidia-smi          # 看有幾張 GPU、各用了多少記憶體、有沒有別人在跑
lab$ df -h ~             # 家目錄還剩多少空間（資料集 + 5 組 checkpoint 需要約 10 GB 以上）
lab$ which conda tmux wget unzip
```

`nvidia-smi` 表格裡 `Memory-Usage` 幾乎是 0、`GPU-Util` 是 0% 的那張卡就是空的。記下它的編號（0、1、2…）。

---

## 3. 在實驗室電腦上準備環境

### 3.1 把程式碼傳上去

你的 Mac 上的 `Lab1-DDPM/` 已經是「寫好答案」的版本，直接用 `rsync` 同步上去（排除不需要的東西）：

```bash
mac$ cd ~/Documents/csic30191/hw1
mac$ rsync -avz --exclude '.git' --exclude 'data' --exclude 'results' --exclude 'samples' \
      --exclude '__pycache__' --exclude '.ipynb_checkpoints' --exclude '.DS_Store' \
      Lab1-DDPM/ lab:~/Lab1-DDPM/
```

> 注意 `Lab1-DDPM/` 後面的斜線：代表「把這個資料夾裡面的東西」同步到遠端的 `~/Lab1-DDPM/`。
> `fid/afhq_inception_v3.ckpt`（83 MB）**要**傳上去，算 FID 會用到；只是交作業時不要放進 zip。
>
> 之後如果在 Mac 上改了程式碼，重新跑同一行 rsync 就會只傳有變動的檔案。

### 3.2 安裝 conda（實驗室電腦沒有 `conda` 才需要）

```bash
lab$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
lab$ bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
lab$ ~/miniconda3/bin/conda init bash
lab$ exec bash                                # 重新載入 shell，提示字元前面會出現 (base)
```

### 3.3 建立環境

```bash
lab$ conda create -n ddpm python=3.9 -y
lab$ conda activate ddpm
lab$ cd ~/Lab1-DDPM
lab$ pip install -r requirements.txt
lab$ python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

最後一行必須印出 `True`。如果是 `False`，通常是 pip 裝到的 PyTorch CUDA 版本比實驗室顯卡驅動新。看 `nvidia-smi` 右上角 `CUDA Version`，例如 12.2，就重裝對應版本：

```bash
lab$ pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3.4 先跑一次正確性測試（1 秒）

```bash
lab$ cd ~/Lab1-DDPM/image_diffusion_todo
lab$ pytest tests/test_todo.py -q          # 應該看到 28 passed
```

---

## 4. 用 tmux 讓程式在你斷線後繼續跑（必學）

SSH 斷線（闔上筆電、換 Wi-Fi）時，在那個連線裡跑的程式會跟著被殺掉。`tmux` 會在實驗室電腦上開一個「不會因為你離開而消失」的終端機。原理見新手教學 Part E3。

| 動作 | 指令 |
|---|---|
| 開一個新 session，取名 ddpm | `tmux new -s ddpm` |
| 暫時離開（程式繼續跑） | 按 `Ctrl+b` 放開，再按 `d` |
| 回到 session | `tmux attach -t ddpm` |
| 列出所有 session | `tmux ls` |
| 在 session 裡開新分頁（window） | `Ctrl+b` 再按 `c` |
| 切換分頁 | `Ctrl+b` 再按 `0`／`1`／`2`… |
| 往上捲看輸出 | `Ctrl+b` 再按 `[`，用方向鍵捲，按 `q` 離開 |

每次開新的 tmux window 都要重新 `conda activate ddpm`。

---

## 5. Task 2：準備資料 → 開始訓練（最優先）

### 5.1 下載資料集 + 建立 FID 評估集（只做一次）

```bash
lab$ tmux new -s ddpm
lab$ conda activate ddpm
lab$ cd ~/Lab1-DDPM/image_diffusion_todo
lab$ python dataset.py
```

它會用 `wget` 從 Dropbox 下載 AFHQ（數 GB）、解壓到 `data/afhq/`，再把 val 圖片縮成 64×64 放到 `data/afhq/eval/`（算 FID 的「真實圖片」）。**等它印出 `Constructed eval dir at data/afhq/eval` 再繼續**。

> ⚠️ 一定要等資料下載完再同時開多組訓練。資料夾不存在時 `train.py` 也會自己下載，同時開 5 組就會 5 個程式搶著下載、解壓到同一個資料夾，很可能弄壞檔案。
>
> 如果實驗室電腦不能連外網：在 Mac 上用瀏覽器下載 `https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1`，
> 然後 `rsync -avP afhq.zip lab:~/Lab1-DDPM/image_diffusion_todo/data/`，在實驗室 `cd data && unzip afhq.zip && rm afhq.zip && cd ..`，再跑 `python dataset.py`。

### 5.2 要跑的 5 組實驗

| # | `--mode` | `--predictor` | 用在報告哪裡 |
|---|---|---|---|
| 1 | linear | noise | scheduler 比較 **和** predictor 比較（共用） |
| 2 | quad | noise | scheduler 比較 |
| 3 | cosine | noise | scheduler 比較 |
| 4 | linear | x0 | predictor 比較 |
| 5 | linear | mean | predictor 比較 |

### 5.3 訓練指令

```bash
lab$ python train.py --mode linear --predictor noise --gpu 0 --log_interval 5000
```

參數說明：

- `--gpu 0`：用第幾張卡（照 `nvidia-smi` 找空的那張）。
- `--log_interval 5000`：每 5000 步存一次 checkpoint、loss 圖、4 張樣本和一張去噪軌跡圖 `step={STEP}-traj.png`。
  **預設是 200**，代表每 200 步就要額外跑 2 次完整的 1000 步取樣，100k 步下來要做 500 次，會多花好幾個小時、產生 2500 張圖。調成 5000 仍然會有 20 張軌跡圖可以放報告，模型本身完全不受影響（README 也這樣建議）。
- 預設 `--train_num_steps 100000`（投影片要求 50k～100k）。**時間不夠時**可加 `--train_num_steps 50000`，約省一半時間。

### 5.4 同時跑多組（如果有多張空 GPU）

在同一個 tmux session 裡每組開一個 window：

```bash
# window 0（已經在裡面）
lab$ python train.py --mode linear --predictor noise  --gpu 0 --log_interval 5000
# Ctrl+b c 開 window 1
lab$ conda activate ddpm && cd ~/Lab1-DDPM/image_diffusion_todo
lab$ python train.py --mode quad   --predictor noise  --gpu 1 --log_interval 5000
# Ctrl+b c 開 window 2 ... 依此類推
lab$ python train.py --mode cosine --predictor noise  --gpu 2 --log_interval 5000
lab$ python train.py --mode linear --predictor x0     --gpu 3 --log_interval 5000
lab$ python train.py --mode linear --predictor mean   --gpu 4 --log_interval 5000
```

- 只有 1 張 GPU：先跑 #1（linear+noise，最有機會拿到 FID < 15），跑完再依序跑其他組。GPU 記憶體夠大（`nvidia-smi` 顯示剩 >20 GB）也可以同一張卡同時跑 2 組，只是每組都會變慢。
- 想要「一組接一組自動跑」，可以把指令用 `&&` 串起來丟在同一個 window：
  ```bash
  lab$ python train.py --mode quad --predictor noise --gpu 0 --log_interval 5000 && \
       python train.py --mode cosine --predictor noise --gpu 0 --log_interval 5000
  ```
- 實驗室是共用資源：不要佔滿所有卡，跑之前看一下有沒有人在用，有規定就照規定。

### 5.5 監看進度

```bash
lab$ tmux attach -t ddpm                     # 看 tqdm 進度條與 Loss
lab$ nvidia-smi                              # GPU 有在動就是有在跑
lab$ ls ~/Lab1-DDPM/image_diffusion_todo/results/predictor_noise/beta_linear/   # 每次訓練一個時間戳資料夾
```

結果資料夾：`results/predictor_{PREDICTOR}/beta_{MODE}/{月-日-時分秒}/`，裡面有：

- `last.ckpt`：模型權重（取樣要用）
- `loss.png`：loss 曲線
- `step={STEP}-traj.png`：去噪過程（報告的 scheduler 比較要用）
- `step={STEP}-0~3.png`：當時的 4 張樣本
- `config.json`：這次訓練用的參數

**在 Mac 上看圖**（不用下載 checkpoint）：

```bash
mac$ rsync -avz --exclude '*.ckpt' lab:~/Lab1-DDPM/image_diffusion_todo/results/ ~/Documents/csic30191/hw1/lab_results/
mac$ open ~/Documents/csic30191/hw1/lab_results
```

**健康檢查**：

- noise predictor 的 loss 應該很快從 ~1 降到 0.1 以下，之後在低點震盪（x0 / mean predictor 的 loss 尺度不同，數值不能直接比）。
- 訓練 1～2 萬步後，`traj.png` 最右邊應該看得出動物臉輪廓。
- 如果 loss 變成 `nan` 或 traj 一直是雜訊，停下來（`Ctrl+c`）回報錯誤。

---

## 6. Task 1：跑 notebook 並存圖（等 Task 2 訓練時做）

Task 1 很輕量（在 GPU 上一兩分鐘），用 Jupyter 跑。因為 Jupyter 是網頁介面，而它跑在實驗室電腦上，所以要用 **SSH port forwarding** 把實驗室的網頁「接」到你 Mac 的瀏覽器（原理見新手教學 Part E4）。

### 6.1 在實驗室啟動 Jupyter（開一個新的 tmux window）

```bash
lab$ tmux attach -t ddpm        # 然後 Ctrl+b c 開新 window
lab$ conda activate ddpm
lab$ cd ~/Lab1-DDPM/2d_plot_diffusion_todo
lab$ jupyter lab --no-browser --port 8888
```

它會印出一行類似 `http://localhost:8888/lab?token=abc123...`，把 token 那串記下來。

> 如果顯示 8888 被佔用，它會自動改用 8889 之類，下面的指令就跟著改數字。

### 6.2 在 Mac 打通隧道（開一個新的 Mac 終端機分頁）

```bash
mac$ ssh -N -L 8888:localhost:8888 lab
```

這行執行後不會有任何輸出、會一直「卡住」，這是正常的（它在轉送流量）。然後在 Mac 瀏覽器開 `http://localhost:8888/lab?token=abc123...`。

### 6.3 執行 notebook

1. 打開 `ddpm_tutorial.ipynb`。
2. 第 4 格 hyperparameters 裡 `device = "cuda:0"`：如果 GPU 0 在跑 Task 2 訓練，改成空閒的卡（例如 `"cuda:1"`）。notebook 註解說只能改 `device`，其他超參數不要動。
3. 為了拿到報告用的 PNG，**在下面三個位置各插入一格**（點選該格後按 `b` 在下方新增）：
   - 「Visualize q(x_t)」那格之後：
     ```python
     fig.savefig("report_q_sample.png", bbox_inches="tight")
     ```
   - Training 那格之後：
     ```python
     plt.figure(); plt.plot(losses); plt.title("Loss curve"); plt.savefig("report_loss_curve.png", bbox_inches="tight")
     ```
   - 最後 Evaluation 那格之後：
     ```python
     fig.savefig("report_samples.png", bbox_inches="tight")
     ```
4. 選單 **Run → Run All Cells**。
5. 確認最後印出 `DDPM Chamfer Distance: xx.xxxx` 且小於 20，存檔（`Cmd+s`）。

### 6.4 把結果拿回 Mac

```bash
mac$ rsync -avz lab:~/Lab1-DDPM/2d_plot_diffusion_todo/report_*.png ~/Documents/csic30191/hw1/lab_results/task1/
mac$ rsync -avz lab:~/Lab1-DDPM/2d_plot_diffusion_todo/ddpm_tutorial.ipynb ~/Documents/csic30191/hw1/Lab1-DDPM/2d_plot_diffusion_todo/
```

第二行把「有執行結果的 notebook」帶回來，交作業時一起放進 zip。
用完之後在 Jupyter 那個 tmux window 按 `Ctrl+c` 兩次關掉，Mac 上的隧道也 `Ctrl+c` 關掉。

> **替代方案（不用 port forwarding）**：VS Code 安裝「Remote - SSH」擴充套件，左下角 `><` → Connect to Host → `lab`，直接在 VS Code 裡開 notebook、選 `ddpm` kernel 執行，圖片右鍵就能存。

---

## 7. Task 2：取樣 + 算 FID（每組訓練結束後）

先確認訓練結束（tqdm 到 100%，並印出 `Saved the final checkpoint at step ...`）。

```bash
lab$ cd ~/Lab1-DDPM/image_diffusion_todo
lab$ CKPT=$(ls -td results/predictor_noise/beta_linear/*/ | head -1)last.ckpt   # 取最新一次的 checkpoint
lab$ echo $CKPT

# (a) 500 張給 FID 用
lab$ python sampling.py --ckpt_path $CKPT --save_dir samples/linear_noise --gpu 0

# (b) 8 張給報告的 predictor 比較用，順便存一張完整去噪軌跡
lab$ python sampling.py --ckpt_path $CKPT --save_dir samples/linear_noise_8 --num_samples 8 --save_traj --gpu 0

# (c) 算 FID（第一個參數是真實圖，第二個是你生成的圖）
lab$ CUDA_VISIBLE_DEVICES=0 python fid/measure_fid.py data/afhq/eval samples/linear_noise
```

- `sampling.py` 會從 checkpoint 自動讀出 schedule 和 predictor，不用再指定。
- `measure_fid.py` 沒有 `--gpu` 參數，固定用「看得到的第一張卡」，所以用 `CUDA_VISIBLE_DEVICES=N` 指定要用哪張。
- 最後會印出 `FID: 12.34...`。**把這行連同終端機畫面截圖**（報告要「最佳 FID 截圖」，20 分）。

其他 4 組照樣替換路徑與資料夾名稱：

| 組別 | checkpoint 資料夾 | `--save_dir`（FID / 8 張） |
|---|---|---|
| linear + noise | `results/predictor_noise/beta_linear/` | `samples/linear_noise`、`samples/linear_noise_8` |
| quad + noise | `results/predictor_noise/beta_quad/` | `samples/quad_noise`、`samples/quad_noise_8` |
| cosine + noise | `results/predictor_noise/beta_cosine/` | `samples/cosine_noise`、`samples/cosine_noise_8` |
| linear + x0 | `results/predictor_x0/beta_linear/` | `samples/linear_x0`、`samples/linear_x0_8` |
| linear + mean | `results/predictor_mean/beta_linear/` | `samples/linear_mean`、`samples/linear_mean_8` |

FID 評分：

| FID | 分數 |
|---|---|
| < 15 | 20 |
| 15 ≤ FID < 20 | 15 |
| 20 ≤ FID < 30 | 10 |
| 30 ≤ FID < 40 | 5 |
| ≥ 40 | 0 |

- FID 出現 200 多（投影片的「Incorrect FID」例子）→ 幾乎一定是沒先跑 `python dataset.py`，或 `samples/xxx` 裡混了別的圖。
- 只需要「最好的那一組」達標，通常是 linear+noise 或 cosine+noise。x0 / mean predictor 的 FID 差是正常現象（投影片範例裡 x0 predictor 的圖也是一團色塊），拿來討論就好。
- 如果最好的 FID 在 15～20 之間且還有時間：可以用 `--train_num_steps 100000`（若之前用 50k）重跑最好那組。

把所有取樣結果拿回 Mac：

```bash
mac$ rsync -avz lab:~/Lab1-DDPM/image_diffusion_todo/samples/ ~/Documents/csic30191/hw1/lab_results/samples/
mac$ rsync -avz --exclude '*.ckpt' lab:~/Lab1-DDPM/image_diffusion_todo/results/ ~/Documents/csic30191/hw1/lab_results/results/
```

---

## 8. 報告內容清單

### Task 1（40 分）

1. **解釋所有 TODO（20 分）**：照第 1 節 Task 1 的五段寫，每段「公式 + 1～3 行關鍵程式碼 + 為什麼」。
2. **q_sample 視覺化（10 分）**：`report_q_sample.png`（t = 0, 50, …, 450 從螺旋逐漸變成高斯雲）。
3. **loss 曲線 + 生成分佈（10 分）**：`report_loss_curve.png` + `report_samples.png`，附上 Chamfer Distance 數值。

### Task 2（60 分）

1. **解釋所有 TODO（30 分）**：照第 1 節 Task 2 的四段寫。
2. **三種 beta schedule 的軌跡比較（10 分）**：放 linear / quad / cosine 各一張 `traj.png`（取最後一個 step 的 `step=...-traj.png` 或 `samples/xxx_noise_8_traj.png`），附 FID 表格。可以討論的點：
   - linear 的 $\bar\alpha_t$ 在前段就掉很快，軌跡前面好幾格都是純雜訊，影像在最後幾格才突然浮現。
   - cosine 的 $\bar\alpha_t$ 下降較平緩，雜訊程度在各時間步分佈比較平均，影像輪廓較早出現、變化較平順。
   - quad 前段 $\beta$ 很小、後段變大，介於兩者之間（$\bar\alpha_T$ 較大，最後一步殘留一點訊號）。
3. **三種 predictor 的結果（10 分）**：linear+noise / x0 / mean 各 8 張（`samples/linear_*_8/`，共 24 張），討論：
   - noise：目標永遠是標準常態分佈、尺度固定，最好學，品質最好。
   - x0：t 很大時要從幾乎純雜訊猜整張圖，最佳解是「所有可能圖片的平均」→ 模糊、色塊化。
   - mean：$\tilde\mu$ 幾乎等於 $x_t$ 本身，網路要學的有用訊號只佔很小一部分，loss 對各時間步的權重也不平均，細節容易出錯。
4. **最佳 FID 截圖（20 分）**：終端機畫面截圖。

---

## 9. 打包繳交（在 Mac 上做）

```bash
mac$ cd ~/Documents/csic30191/hw1
mac$ mkdir -p submit && cp report.pdf submit/                 # report.pdf 是你寫好的報告
mac$ rsync -a --exclude 'data' --exclude 'results' --exclude 'samples' --exclude 'afhq_inception_v3.ckpt' \
      --exclude '__pycache__' --exclude '.ipynb_checkpoints' --exclude '.pytest_cache' --exclude '.DS_Store' \
      Lab1-DDPM/2d_plot_diffusion_todo Lab1-DDPM/image_diffusion_todo submit/
mac$ cd submit && zip -r ../<學號>_lab1.zip report.pdf 2d_plot_diffusion_todo image_diffusion_todo && cd ..
mac$ unzip -l <學號>_lab1.zip | grep -E 'ckpt|data/|results/|samples/' || echo "OK：沒有夾帶不該交的檔案"
```

---

## 10. 常見問題

| 症狀 | 原因 / 解法 |
|---|---|
| `ssh: connect to host ... Connection refused / timed out` | IP 或 port 錯、不在校內網路（可能要先連學校 VPN）、或要走跳板機 |
| `Permission denied (publickey)` | `ssh-copy-id` 沒成功，重做 2.2 |
| `torch.cuda.is_available()` 是 `False` | 見 3.3 重裝對應 CUDA 版本的 torch |
| `CUDA out of memory` | 那張卡別人在用，換 `--gpu`；或同一張卡跑太多組 |
| `ModuleNotFoundError: dataset` | 沒有 `cd` 到 `image_diffusion_todo/`（或 Task 1 的 `2d_plot_diffusion_todo/`）再執行 |
| 斷線回來程式不見了 | 沒在 tmux 裡跑；用 `tmux ls` 確認 session 還在 |
| `sampling.py` 報 `does not record its beta schedule` | 用了舊版程式碼訓練的 checkpoint，用現在的程式碼重新訓練 |
| FID 兩三百 | 沒先 `python dataset.py`，或 `--save_dir` 目錄裡有其他圖片 |
| 瀏覽器開 `localhost:8888` 連不上 | Mac 上的 `ssh -N -L` 那個視窗被關了，或 port 數字和 Jupyter 印出來的不一樣 |
