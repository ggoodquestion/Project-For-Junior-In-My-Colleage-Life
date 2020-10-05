2020專題學習筆記
==
## HMM - 隱藏式馬可夫模型(Hidden Markov Model)

### Markov Chain(馬可夫鏈)：
Markov Chain為一隨機過程，該過程要求具備「無記憶」的性質：下一狀態的概率分布只能由當前狀態決定，在時間序列中它前面的事件均與之無關，稱作**馬可夫性質**，可用以下數學式描述 :
>$P(q_i=a|q_1...q_{i-1})=P(q_i=a|q_{i-1})$

### 模型描述
而當我們感興趣之事件並不可直接觀測時，稱其為隱藏事件(Hidden)，比如詞性標記需以上下文辨認，這時可以使用HMM。

>參數:
>1. $O={o_1,o_2,…,o_T}$ 
: observations(觀測序列)
>2. $Q={q_1,q_2,…,q_N}$ 
: states(隱序列，我們感興趣的事件)
>3. $A=a_{11}...a_{ij}...a_{NN}$ 
: transition probability matrix
   (轉移機率矩陣，紀錄當前隱藏狀態轉移到下一隱藏狀態之機率)
>4. $B=b_i(o_t)$ 
: observation likelihoods/emission probabilities
   (發射機率分布，紀錄當前隱藏狀態影響觀測狀態之機率)
>5. $\pi=\pi_1,\pi_2,...,\pi_N$ 
: 初始狀態
#### 範例
>![](https://i.imgur.com/Rh9m0hU.png)

以上圖為例，假設$O=${$1，2，3$}表示某人吃的冰淇淋個數，$Q=${$H$(Hot)，$C$(Cold)}則為我們想知道的天氣狀況，$\pi=[0.8，0.2]$表示初始狀態為$H$的機率為$0.8$；$C$為$0.2$。
轉移機率矩陣    $T=$
$$
  \begin{bmatrix}
   0.6 &0.4 \\
   0.5 &0.5 \\ 
  \end{bmatrix} \
$$
發射機率矩陣    $B=$(以列為Q，行為O)
$$
  \begin{bmatrix}
   0.2 &0.4 &0.4 \\
   0.5 &0.4 &0.1 \\
  \end{bmatrix} \
$$

#### 馬可夫性質數學推演
馬可夫性質是有階層的，一階馬可夫表示當前狀態只與前一狀態有關；二階馬可夫表示當前狀態只與前兩狀態有關；推廣至$n$階馬可夫則表示當前狀態與前$n$個狀態有關。

首先定義$P(Q)$為發生某狀態序列的機率
$$P(Q)=P(q_1,q_2,...,q_T)$$
$$=P(q_T|q_{T-1},q_{T-2},...,q_1)P(q_{T-1},q_{T-2},...,q_1)$$
$$=P(q_T|q_{T-1},q_{T-2},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)P(q_{T-2},...,q_1)$$
$$...$$
$$=P(q_T|q_{T-1},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)...P(q_2|q_1)P(q_1)$$
$$=\prod_{t=1}^TP(q_t|q_{t-1},...,q_1)$$

現在套用馬可夫性質

**1-th**

We know,
$$P(Q)=P(q_T|q_{T-1},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)...P(q_2|q_1)P(q_1)$$
$$\implies P(Q)=P(q_T|q_{T-1})P(q_{T-1}|q_{T-2})...P(q_2|q_1)P(q_1)$$
$$=\prod_{t=1}^TP(q_t|q_{t-1})$$

**2-th**
$$P(Q)=P(q_T|q_{T-1},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)...P(q_2|q_1)P(q_1)$$
$$\implies P(Q)=P(q_T|q_{T-1}.q_{T-2})P(q_{T-1}|q_{T-2},q_{T-3})...P(q_3|q_2,q_1)P(q_2|q_1)P(q_1)$$
$$=\prod_{t=1}^TP(q_t|q_{t-1},q_{t-2})$$

**n-th**

最後可以推得在$n$階馬可夫時
$$P(Q)=\prod_{t=1}^TP(q_t|q_{t-1},q_{t-2},...,q_{t-n})$$

#### Three fundamental problems
>1. Likelihood : The Forward Algorithm
>2. Decoding : The Viterbi Algorithm
>3. Learning : The Forward-Backward Algorithm

### HMM基礎三問題的數學推演
- #### Likelihood(相似性問題)：Forwarding Algorithm
> 本問題主要為計算發生某個觀測序列之最大機率，也就是把所有會發生此事件的路徑的機率加總即為所求。
> 假設觀測序列$O$有$T$個觀測結果即$O=${$o_1,o_2,...,o_T$}，且每個$q$有$N$種狀況
> 
> 目標： $P(O|\lambda)$, $\lambda$={$A,B,\pi$}
> 
> $$\implies P(O|Q)=\prod_{i=1}^T P(o_i|q_i)$$
> 
> 透過一階馬可夫性質：
> $$P(O,Q)=P(O|Q)\cdot P(Q)=\prod_{i=1}^TP(o_i|q_i)\cdot \prod_{i=1}^TP(q_i|q_{i-1})$$
> $$\implies P(O) = \sum_QP(O,Q)=\sum_QP(O|Q)\cdot P(Q)$$
> $$=\sum_Q[\prod_{t=1}^TP(o_t|q_t)\cdot P(q_1)\cdot \prod_{t=2}^TP(q_t|q_{t-1})]$$
> 得到的式子表示發生$O$的機率為找出所有$Q$的可能性並相加所有結果，但是這種遍尋的方式效率太低，因此我們將每個節點發生的機率記錄下來並用於下一次的計算，此稱為Forwarding Algorithm。
> 
> 在此定義$a_t(j)$ 表示在$j$狀況下時，且觀察到$o_t$的機率
> $$a_t(j)=P(o_1,o_2...,o_t,q_t=j|\lambda)$$
> $$\implies a_t(j)=\sum_{i=1}^Na_{t-1}(i)a_{ij}b_j(o_t)=\sum_{i=1}^NP(o_1,o_2,...,q_{t-1}=i|\lambda)a_{ij}b_j(o_t)$$ 
> 並定義$b_j(o_t)$表示在$j$狀況之下發生$o_t$的機率
> $$b_j(o_t)=P(o_t|q_j)$$
> 得到上述式子後我們可以總結出Forwarding Algorithm的步驟
> 1. Initialization
> $$a_t(j)=\pi_jb_j(o_1)$$
> 2. Recursion
> $$a_t(j)=\sum_{i=1}^Na_{t-1}(i)a_{ij}b_j(o_t) \quad,1\leq j\leq N,\quad 1< t\leq T$$
> 3. Termination
> $$P(O|\lambda)=\sum_{i=1}^Na_T(i)$$
> 
> 經過該演算之後即可得到發生序列$O$的最大機率，為Likelihood問題的解

- #### Decoding(解碼)：Viterbi Algorithm
> 本問題主要在找出造成觀測序列$O$最有可能的隱序列$Q$，在每次的遞迴中記錄著當次找到的狀態$q$，在結束遞迴後trace back即可找出該隱序列。
> 
> 目標：$Q^*=\arg \max_QP(O,Q|\lambda)$
> 
> 首先，與Likelihood問題的式子相似
> $$a_t(j)=\sum_{i=1}^Na_{t-1}(i)a_{ij}b_j(o*-t)$$
> 定義$V_t(j)$為在觀測到$o_1...o_t$且經過$q_1...q_{t-1}$路徑且當前狀態為$j$的最大機率
> $$v_t(j)=\max_{q_1...q_{t-1}}P(q_1,...,q_{t-1},o_1,...o_t,q_t=j|\lambda)$$
> $$=\max_{i=1}^Nv_{t-1}(j)a_{ij}b_j(o_t)$$
不同於Likelihood是找所有可能的加總，Decoding是找最大的可能並將路徑記錄下來，即為答案(Viterbi Path)，此處為了記錄路徑，**定義$k_t(j)$為在地$t$個觀測結果時隱狀態為$j$的前一狀態**，因此歸納出以下步驟。
> 1. Initialization
> $$v_1(j)=\pi_jb_j(o_1) \quad 1\leq j\leq N$$
> $$k_1(j)=0 \quad 1\leq j\leq N$$
> 2. Recursion
> $$v_t(j)=\max_{i=1}^Nv_{t-1}(i)a_{ij}b_j(o_t) \quad 1\leq j\leq N , 1<t\leq T$$
> $$k_t(j)=\arg \max_{i=1}^Nv_t(j)=\arg \max_{i=1}^Nv_{t-1}(i)a_{ij}b_j(o_t) \quad 1\leq j\leq N , 1<t\leq T$$
> 3. Terminal: The best probaility and start of backtrace
> $$P^*=\max_{i=1}^Nv_T(i)$$
> $$q_T^*=\arg \max_{i=1}^Nv_T(i)$$

#### 結論
在解決語音辨識問題時使用HMM是最為快速的方式，觀測到的聲音為觀測序列，而隱性序列則是欲說出口的字，該字造成觀測到之波型的產生，而決定下一隱性狀態的則為欲說出的詞，而這些資料都有它一定的轉移機率，這些資料即構成一個HMM。

### Viterbi algorithm - 維特比演算法
維特比演算法是一種動態規劃演算法(dynamic programing)，用於尋找在HMM中最有可能導致觀測序列的隱序列。
> #### Dynamic programing - 動態規劃
> 動態規劃演算法適用於有重疊子問題與最佳子結構性質的問題，亦即該大問題的最佳解包含其子問題的最佳解，並且子問題的最佳解確定後即不受以後的問題影響，因此可以藉由前次小問題之最佳解遞迴算出大問題的最佳解。
> 基本思路為將大問題分割成小問題，並將每此的小問題的解當作下個小問題的解直到解決問題。

#### 步驟

給定一個HMM，其共有$k$個狀態，初始狀態$i$的機率為$\pi_i (0< i <= k)$，從狀態$i$轉移到$j$的轉移機率為$a_{ij} (0 < j <= k)$。令觀察序列$o_1,o_2,...,o_n$，產生結果最有可能的隱序列$q_1,q_2,...,q_n$。

令$v_t(j)$為$q_j$在經過$t$次觀測且經過最大可能值序列$q_1...q_{t-1}$，則可得 
$v_t(j) = \max\limits_{i}v_{t-1}(i)a_{ij}b_j(o_t)$

 
透過遞迴計算$v_t(j)$直到$t=N$可得所有觀測結果$o_i$之最大可能$q_j$，接著trace back記錄好的$q_j$即可求出該觀測序列$O$最有可能的隱性序列$Q$。

#### 結論
在建立聲學模型時，搭配維特比演算法便可以透過聲音波型去對應語料庫，藉由維特比的的trace back，即可找到符合該聲音波型的句子。
***
## GMM - 高斯混和模型(Gaussian Mixture Model)

假定資料$X=${$x_1,...,x_N$}**不規則**分布於$d$維空間中，這時使用單一高斯分布模型並不能良好的近似，因此產生出變通方案 : **採用數個高斯函數的加權平均來表示**，即產生GMM。
>#### 想法
>單一個波中有許多的峰（multimodal），可以將他們拆成許多的單峰圖形（unimodal），並給定權重（weight），再將之組合。

>#### 應用
>背景濾除(Background Subtraction) :
>影像的像素質因移動、亮度的改變而不固定，且大部分情況下，顏色的分布在某幾個值做變動，因此適合使用GMM。

### 模型描述
$\lambda=${$w_i,\mu_i,\sum_i$}，$i=1,...,M$
>參數 : 
>$w_i$ : mixture weight(混和加權值)
>$\mu_i$ : mean vector(平均向量)
>$\sum_i$ : covariance matrix(共變數矩陣)
>$M$ : 高斯分布個數

若資料$X=${$x_1,..,x_n,..,x_N$}在$D$維空間中分布，其高斯混和模型的相似函數可表示如下 : 
>$p(x_n|\lambda)=\sum_{i=1}^{M}w_ig_i(x_n)$

其中
>$g_i(x)=\dfrac{1}{(2\pi)^{D/2}(|\sum_i|)^{1/2}}e^{-\dfrac{(x-\mu_i)^T(\sum_i)^{-1}(x-\mu_i)}{2}}$，

為第$i$個高斯分布密度函數(joint gaussian)，且
>$\sum_{i=1}^{M}w_i=1$

如下圖所示：

>![](https://i.imgur.com/9MMthNJ.png)


### 參數初始化
使用K-means Cluster(K平均值分類法)
>0. 收集資料 : 獲取N個欲訓練的特徵向量
>1. 初始化 : 假設群數為K，隨機取K個向量當成每群的中心點
>2. 以新的群中心分群 : 其他(N-K)個向量對這K個群中心做距離測量，每個向量被分到距離最短的中心
>3. 更新群中心 : 對每一群算出新的向量平均值，以此為新的群中心
>4. 判斷分群是否收斂 : 與舊的群中心比較，不再有變動則表示收斂，則繼續步驟5.，反之則重複2.、3.
>5. 判斷是否合併群 : 如果這K群中，任2群距離太近或是某一群的向量點只有一個，表示群數須減少，則群數減一(K<=K-1)，並回到1.重新分群；反之，則做6.。
>6. 得到初始化參數 : 將最後分群的個數、群的中心、群的變異數以及每一群的資料個數當作高斯混合模型的**初始參數$(M，\mu，\sum，w)$**。

### 參數估測 - EM(期望值最大演算法，Expectation Maximization)
為了估測最佳的高斯混合模型參數$\lambda$，所謂「最佳」指的是資料真正的分佈，與模型參數$\lambda$估測出來的分佈有最大的相似度，常用的方法是**MLE**(最佳相似性估測法，Maximum Likelihood Estimation)。
由上述模型描述可知，若$x_i，i=1,...,N$為互相獨立之事件，則發生$X=${$x_1,...,x_N$}之相似函數可表示成
>$P(X|\lambda)=\prod_{i=1}^{N}P(x_i|\lambda)$

由於$X$是確定的，因此MLE主要就是找出使得GMM的相似函數值為最大時的參數$\lambda'$，也就是
>$\lambda'=arg\max\limits_{\lambda}P(X|\lambda)$

但上式對$\lambda$而言是一個非線性的方程式，無法直接最大化相似函數，所以採用EM，利用疊代的方式找出MLE的估測參數$\lambda'$。
#### 做法
由之前K-means Cluster找出的初始化參數$\lambda$，利用EM估計出新的參數$\bar{\lambda}$，使得滿足
>$P(X|\bar{\lambda})$$\ge$$P(X|\lambda)$

令$\lambda=\bar{\lambda}$重新疊代估計新的$\lambda$，直到$P(X|\lambda)$收斂或是達到某個門檻值才停止。

#### E-step
目的是測試我們所求的likelihood函數值，是否達到我們的要求，若符
合要求，EM演算法就停止，反之就繼續執行EM演算法。
以3個高斯分布函數組合而成之模型為例，其密度函數為 : 
>$P(x_i)=w_1g(x_i;\mu_1,\sum_1)+w_2g(x_i;\mu_2,\sum_2)+w_3g(x_i;\mu_3,\sum_3)$

且令參數
>$\lambda=[w_1,w_2,w_3,\mu_1,\mu_2,\mu_3,\sum_1,\sum_2,\sum_3]$

求出likelihood的最大值：
>$E(\lambda)=\ln{(\prod_{i=1}^{N}P(x_i))}=\sum_{i=1}^{N}\ln{[w_1g(x_i;\mu_1,\sum_1)+w_2g(x_i;\mu_2,\sum_2)+w_3g(x_i;\mu_3,\sum_3)]}$

令$\beta_j(x)$為事後機率，代表觀測到亂數向量值$x$，為第$j$個高斯密度函數所產生的，則
>$\beta_j(x)=p(j|x)=\dfrac{p(j)p(x|j)}{p(x)}=\dfrac{w_jg(x;\mu_j,\sum_j)}{w_1g(x;\mu_1,\sum_1)+w_2g(x;\mu_2,\sum_2)+w_3g(x;\mu_3,\sum_3)}$

#### M-step
為了要找到使likelihood函數最大化的參數，因此我們分別對$w_i$、$\mu_i$、$\sum_i$做偏微分。
假設初始參數是$\lambda_{old}$，我們希望找出新的$\lambda$值滿足
>$E(\lambda)>E(\lambda_{old})$

推得
>$E(\lambda)-E(\lambda_{old})=\sum_{i=1}^{N}\ln{\dfrac{w_1g(x_i;\mu_1,\sum_1)+w_2g(x_i;\mu_2,\sum_2)+w_3g(x_i;\mu_3,\sum_3)}{w_{1,old}g(x_i;\mu_{1,old},\sum_{1,old})+w_{2,old}g(x_i;\mu_{2,old},\sum_{2,old})+w_{3,old}g(x_i;\mu_{3,old},\sum_{3,old})}}$
>$=\sum_{i=1}^{N}\ln{[\dfrac{w_1g(x_i;\mu_1,\sum_1)}{D(\lambda_{old})}}\dfrac{\beta_1(x_i)}{\beta_1(x_i)}+\dfrac{w_2g(x_i;\mu_2,\sum_2)}{D(\lambda_{old})}\dfrac{\beta_2(x_i)}{\beta_2(x_i)}+\dfrac{w_3g(x_i;\mu_3,\sum_3)}{D(\lambda_{old})}\dfrac{\beta_3(x_i)}{\beta_3(x_i)}]$
>$\ge$$\sum_{i=1}^{N}[\beta_1(x_i)\ln{\dfrac{w_1g(x_i;\mu_1,\sum_1)}{D(\lambda_{old})\beta_1(x_i)}}+\beta_2(x_i)\ln{\dfrac{w_2g(x_i;\mu_2,\sum_2)}{D(\lambda_{old})\beta_2(x_i)}}+\beta_3(x_i)\ln{\dfrac{w_3g(x_i;\mu_3,\sum_3)}{D(\lambda_{old})\beta_3(x_i)}}]=Q(\lambda)$

上式推導是因為$\ln{(x)}$是一個**凸函數(Convex Function)** ： 任意兩點的割線必在函數上方，滿足下列不等式：
>$\ln{[\alpha x_1+(1-\alpha)x_2]}$$\ge$$\alpha \ln{(x_1)}+(1-\alpha)\ln{(x_2)}$


推廣至**傑森不等式(Jensen Inequality)**：
>$\ln{(\sum_{i=1}^{n}\alpha_ix_i)}$$\ge$$\sum_{i=1}^{n}\alpha_i\ln{(x_i)}$，$\sum_{i=1}^{n}\alpha_i=1$

可得
>$E(\lambda)$$\ge$$E(\lambda_{old})+Q(\lambda)$

目標為求出使得$Q(\lambda)$最大的$\lambda$值
>$Q(\lambda)=\sum_{i=1}^{N}\sum_{j=1}^{3}\beta_j(x_i)[\ln{w_j}+\ln{g(x_i;\mu_j,\sum_j)}]+cl$
>$=\sum_{i=1}^{N}\sum_{j=1}^{3}\beta_j(x_i)[\ln{w_j}+\ln{(\dfrac{1}{(2\pi)^{d/2}(|\sum_j|)^{1/2}}e^{-\dfrac{(x_i-\mu_j)^T(\sum_j)^{-1}(x_i-\mu_j)}{2}})}]+cl$

再將$Q$對$\mu_j$及$\sum_j$偏微分且值為0，
得到
>$\mu_j=\dfrac{\sum_{i=1}^{N}\beta_j(x_i)x_i}{\sum_{i=1}^{N}\beta_j(x_i)}$
>$\sum_j=\dfrac{\sum_{i=1}^{N}\beta_j(x_i)(x_i-\mu_j)(x_i-\mu_j)^T}{\sum_{i=1}^{N}\beta_j(x_i)}$

欲得到最佳$w_j$之值，須將$w_j$的總合為1的條件加入，引進**Lagrange Multiplier**，並定義新的目標函數為：
>$E_{new}(\lambda)=E(\lambda)+\alpha(w_1+w_2+w_3-1)$

將$E_{new}$對3個weight做偏微分，得
>$-\dfrac{1}{w_j}\sum_{i=1}^{N}\beta_j(x_i)+\alpha=0，j=1,2,3$
>$\implies$ $(w_1+w_2+w_3)\alpha=-\sum_{i=1}^{N}[\beta_1(x_i)+\beta_2(x_i)+\beta_3(x_i)]$
>$\implies$ $\alpha=-\sum_{i=1}^{N}1=-N$
>$\implies$ $w_j=\dfrac{1}{N}\sum_{i=1}^{N}\beta_j(x_i)，j=1,2,3$

Likelihood function $E(\lambda)$最大化的示意圖 : 
>![](https://i.imgur.com/i0K3lie.png)
#### 結論
將N個準備拿來訓練模型的資料點，經過K-means Clustering 後得到初始的參數，再由EM演算法得到的三個方程式，
>$\mu_j=\dfrac{\sum_{i=1}^{n}\beta_j(x_i)x_i}{\sum_{i=1}^{n}\beta_j(x_i)}$
>$\sum_j=\dfrac{\sum_{i=1}^{n}\beta_j(x_i)(x_i-\mu_j)(x_i-\mu_j)^T}{\sum_{i=1}^{n}\beta_j(x_i)}$
>$w_j=\dfrac{1}{n}\sum_{i=1}^{n}\beta_j(x_i)$

進行參數的更新，並計算新的相似函數的值，如此不斷地更新模型的參數，直到相似函數的值已經沒什麼變動，或是疊代的次數超過某個門檻值，才停止疊代。

GMM建立的流程圖 : 
>![](https://i.imgur.com/Cp9O27M.png)

### 機率推演
欲求
>$\lambda^*=arg\max\limits_{\lambda}p(O|\lambda)$

其中$O$為已知事件，$\lambda$為模型參數。
令$Q$為，可得
>$p(O,Q|\lambda)=\dfrac{p(O,Q,\lambda)}{p(\lambda)}=\dfrac{p(Q,(O,\lambda))}{p(\lambda)}=\dfrac{p(Q|O,\lambda)p(O,\lambda)}{p(\lambda)}=p(O|\lambda)p(Q|O,\lambda)$

等式同取$\ln$，得
>$\ln{p(O,Q|\lambda)}=\ln{[p(O|\lambda)p(Q|O,\lambda)]}$
$\implies$ 
$\ln{p(O|\lambda)}=\ln{p(O,Q|\lambda)-\ln{p(Q|O,\lambda)}}$

等式對已知參數$\bar{\lambda}$同取期望值，得
>$E_{Q|O,\bar{\lambda}}[\ln{p(O|\lambda)}]=E_{Q|O,\bar{\lambda}}[\ln{p(O,Q|\lambda)}-\ln{p(Q|O,\lambda)}]$ 

>$\implies$
$\sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{p(O|\lambda)} = \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{p(O,Q|\lambda)} - \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{p(Q|O,\lambda)}$

>$\implies$
$\ln{p(O|\lambda)} = \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{\dfrac{p(O,(Q,\lambda))}{p(\lambda)}} - \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{\dfrac{p(Q,(O,\lambda))}{p(O,\lambda)}}$ 

>$\implies$
$\ln{p(O|\lambda)} = \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{\dfrac{p(O|Q,\lambda)p(Q,\lambda)}{p(\lambda)}} - \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{\dfrac{p(Q|O,\lambda)p(O,\lambda)}{p(O,\lambda)}}$

>$\implies$
$\ln{p(O|\lambda)} = \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{p(O|Q,\lambda)p(Q|\lambda)} - \sum\limits_{Q}p(Q|O,\bar{\lambda})\ln{p(Q|O,\lambda)}$


***
## Cross-Entropy(交叉熵)
### Entropy 
原本是用在表達化學平衡態的名詞，藉以表達原子在空間當中的分布狀況，如果原子分佈越混亂，我們則說**Entropy越高，表示資訊量越高，越多不確定性存在**，因為有更多因素去影響我們原子的分布狀況。
>例1 : 假設隨機從一個口袋裡取硬幣，口袋裡有一個藍色的，一個紅色的，一個綠色的，一個橘色的。取出一個硬幣之後，每次問一個問題，然後做出判斷，目標是，問最少的問題，得到正確答案。其中一個最好的策略如下 : 相問是不是紅或藍，分為兩組後，再問是否為紅及是否為橘，就能找出所有的球，由於問了兩道題目，$1/4(機率) * 2道題目 * 4顆球 = 2$，平均需要問$2$道題目才能找出不同顏色的球，也就是說期望值為$2$，代表entropy。

>例2 : 例1中變成了袋子中$\dfrac{1}{8}$硬幣是綠色的，$\dfrac{1}{8}$的是橘色的，$\dfrac{1}{4}$是紅色的，$\dfrac{1}{2}$是藍色的，這時最優的策略如下 : 先問是否為藍；再問是否為紅；最後問是否為橘。其中，$\dfrac{1}{2}$ 的概率是藍色，只需要$1$個問題就可以知道是或者不是，$\dfrac{1}{4}$ 的概率是紅色，需要$2$個問題，按照這個邏輯，猜中硬幣需要的問題的期望值是 $\dfrac{1}{2}*1 + \dfrac{1}{4}*2 + \dfrac{1}{8}*3 + \dfrac{1}{8}*3 = 1.75$

>總結上面的例子，假設一種硬幣出現的概率是$p$，那麼猜中該硬幣的所需要的問題數是$\log_2\dfrac{1}{p}$

>公式 : ㄉ(問題個數的期望值，其意義就是在最優化策略下，猜到顏色所需要的問題的個數，不確定性越高，entropy就越大)
![](https://i.imgur.com/HchuUXc.png)

### Cross-entropy
用意是在觀測預測的機率分佈與實際機率分布的誤差範圍，就拿下圖為例就直覺說明，cross-entropy(purple line = area under the blue curve)，我們預測的機率分佈為橘色區塊，真實的機率分佈為紅色區塊，藍色的地方就是cross-entropy區塊，紫色現為計算出來的值。我們預測值與實際值差越多，也就是代表內涵的資訊量愈大，也就是不確定越多，也就是 cross-entropy 會越高。
![](https://i.imgur.com/XEcuOch.png)
反之，如果疊合的區塊越多，就代表不確定性越少，也就是 cross-entropy 會越小，如下圖所示。
![](https://i.imgur.com/rUwjg56.png)

>例 : 如果在例2中使用例1的策略 : \dfrac{1}{8}的概率硬幣是橘色，需要$2$個問題，\dfrac{1}{2}的概率是藍色，仍然需要$2$個問題，平均來說，需要的問題數是 $\dfrac{1}{8}*2 + \dfrac{1}{8}*2 + \dfrac{1}{4}*2 + \dfrac{1}{2}*2 = 2$。因此，在例2中使用例1的策略是一個比較差的策略。其中2是這個方案中的cross-entropy。

>因此，給定一個策略，cross-entropy就是在該策略下猜中顏色所需要的問題的期望值，越好的策略，最終的cross-entropy越低。具有最低的cross-entropy的策略就是最優化策略。也就是上面定義的entropy。因此，在機器學習中，我們需要最小化cross-entropy。

>公式 : (p為實際機率分布，q為預測機率分布)
![](https://i.imgur.com/yLgynrk.png)

***
## DTW - 動態時間規整(Dynamic Time Warping)
### 簡介
給定兩向量，欲計算兩者間的距離時我們常使用歐幾里得距離(Euclidian Distance)來計算，例如欲計算$X(4,7), Y(7,11)$的距離應為
$$D(X,Y)=\sqrt{(4-7)^2+(7-11)^2}=5$$
但當兩向量長度不同時，顯然使用歐幾里得距離並不是個好選擇，如下圖，某些點距離另一點距離甚至比其對應歐幾里得距離的點來的更近。一般來說，假設兩向量的元素位置代表的是時間，由於兩向量時間軸可能會有所偏差，因此我們並不清楚點對點的對應關係，因此傳統的歐幾里得距離並不適合用於此情況，我們需要一套有效的運算方法，此方法為DTW，我們稱計算出來的數值為"*DTW distance*"，在此我們更樂意將距離稱為相似性，而所謂的距離是數學上用來量化相似性所給的名稱。
![](https://i.imgur.com/fhFC0cL.png)

### 數學描述
#### 參數定義
DTW的目的在於比對兩個序列的相似性，假設給定一個特徵空間(Feature Space)$F$和兩組序列$X=(x_1,x_2,...,x_N),Y=(y_1,y_2,...,y_M)\quad x_n,y_m\in F,\quad for \quad 1\leq n\leq N,\quad 1\leq m\leq M$，接著我們需要一個表(table)來記錄點與點之間的距離，在此定義$C$為一個$F$ X $F$的矩陣
$$C(n,m)=cost \ of \ x_n \ and \ y_m$$
一般來說當$C(n,m)$越小時表示兩點的相似度越高(*low cost*)，越大時相似度越低(*high cost*)。接著我們還需要將走過的路徑(*DTW path*)記錄下來，在此定義序列$p=(p_1,p_2,...,p_L)$，且$p_l=(n_l,m_l)\in N$ X $M \ matrix ,\quad 1\leq l \leq L$ 表示第$l$個點為$(x_{n_l},y_{m_l})$。

#### 初始條件
由於*DTW*在於對其並校正兩不同長度的序列，首先必須先將兩序列的起始位與結束位置對齊，也就是說可能會對兩序列對時間軸進行壓縮，並且時間對於點需要有單向性(*Monotonicity*)即兩序列的相似狀況不會發生順序上的對調，最後在比對的的時候必須照順序比對點不可跳過(*point by point*)才不會發生相似性遺漏的狀況，以下列出做*DTW*演算時需滿足的條件。

1. 邊界條件($Boundary \ condition$)：$p_1=(1,1),p_L=(N,M)$
2. 單向性($Monotonicity$)：$n_1\leq n_2\leq ...\leq n_L \ and \ m_1\leq m_2\leq ...\leq m_L$
3. 依序性($Step \ size \ condition$)：$p_{l+1}-p_l\in \begin{Bmatrix} (0,1),(1,0),(1,1)\end{Bmatrix}$

#### 演算過程
定義完參數且設定好初始條件後開始要來實際演算，因為*DTW*的條件有單向性和依序性，這樣的問題符合最佳子問題的性質，可以根據前一次找到的最短路徑點繼續推算下一個的最短路徑，並且未來的改變不影響過去的結果，因此可以使用動態規劃(*Dynamic Program*)來解決*DTW*問題。

定義$p^*$為最佳路徑(*optimal warping path*)
$$p_1^*=(1,1)$$
$$p_2^*=\arg \min_{(n,m)}\begin{Bmatrix} C(1,2),C(2,1),C(2,2) \end{Bmatrix}$$
$$p_3^*=\arg \min_{(n,m)}\begin{Bmatrix} C(n_2,m_2+1),C(n_2+1,m_2),C(n_2+1,m_2+1) \end{Bmatrix}$$
$$...$$
$$p_l^*=\arg \min_{(n,m)}\begin{Bmatrix} C(n_{l-1},m_{l-1}+1),C(n_{l-1}+1,m_{l-1}),C(n_{l-1}+1,m_{l-1}+1) \end{Bmatrix}$$
$$...$$
$$p_L^*=(N,M)\implies stop$$
定義$D_l$為路徑$p$上$p_1$到$p_l$的總距離，
$$D_l=\sum_{i=1}^lC(n_i,m_i)$$
由此我們可以歸納出*Dynamic Time Warping Algorithm*的步驟：

1.Initialization
$$p_1=(1,1)$$
$$D_1=C(1,1)$$
2. Rescursion
$$p_l=\arg \min_{(n,m)}\begin{Bmatrix} C(n_{l-1},m_{l-1}+1),C(n_{l-1}+1,m_{l-1}),C(n_{l-1}+1,m_{l-1}+1) \end{Bmatrix}$$
$$D_l=min\begin{Bmatrix} C(n_{l-1},m_{l-1}+1),C(n_{l-1}+1,m_{l-1}),C(n_{l-1}+1,m_{l-1}+1) \end{Bmatrix}\ + \ D_{l-1}$$
3. Terminal
$$P_L=(N,M)$$
$$D_L=C(N,M)+D_{L-1}$$

### 簡易範例
給定$F=\begin{Bmatrix}\alpha ,\beta ,\gamma \end{Bmatrix}$ ，假設$X=(\alpha,\beta,\gamma),Y=(\alpha,\alpha,\beta,\gamma,\beta,\gamma),Z=(\alpha,\gamma,\gamma,\beta,\beta)$，定義
$$cost \ between \ two \ point=\begin{cases}0, \ x_n=y_m \\ 1, \ x_n \neq y_m \end{cases}$$
找出與$X$相似的序列。

1. 首先初始化參數，可將$C_{X,Y}與C_{X,Z}$畫成以下兩圖，
![](https://i.imgur.com/PECMbgS.jpg)
可以看出$X,Y$的路徑$p_{X,Y}$與$X,Z$的路徑$p_{X,Z}$
2. 計算Warping Distance：
透過走過的路徑$p$算出$D_{X,Y}=1, \ D_{X,Z}=3$得出序列$Y$與序列$X$最為相似。
