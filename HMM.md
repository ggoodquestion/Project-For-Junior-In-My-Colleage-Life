HMM - 隱藏式馬可夫模型(Hidden Markov Model)
==

### Markov Chain(馬可夫鏈)：
Markov Chain為一隨機過程，該過程要求具備「無記憶」的性質：下一狀態的概率分布只能由當前狀態決定，在時間序列中它前面的事件均與之無關，稱作**馬可夫性質**，可用以下數學式描述 :
>$P(q_i=a|q_1...q_{i-1})=P(q_i=a|q_{i-1})$

### 模型描述
而當我們感興趣之事件並不可直接觀測時，稱其為隱藏事件(Hidden)，比如詞性標記需以上下文辨認，這時可以使用HMM。

> 參數:
> 1. $O={o_1,o_2,…,o_T}$ 
: observations(觀測序列)
> 2. $Q={q_1,q_2,…,q_N}$ 
: states(隱序列，我們感興趣的事件)
> 3. $A=a_{11}...a_{ij}...a_{NN}$ 
: transition probability matrix
   (轉移機率矩陣，紀錄當前隱藏狀態轉移到下一隱藏狀態之機率)
> 4. $B=b_i(o_t)$ 
: observation likelihoods/emission probabilities
   (發射機率分布，紀錄當前隱藏狀態影響觀測狀態之機率)
> 5. $\pi=\pi_1,\pi_2,...,\pi_N$ 
: 初始狀態
#### 範例
![](https://i.imgur.com/Rh9m0hU.png)

以上圖為例，假設$O=${$1，2，3$}表示某人吃的冰淇淋個數，$Q=${$H$(Hot)，$C$(Cold)}則為我們想知道的天氣狀況，$\pi=[0.8，0.2]$表示初始狀態為$H$的機率為$0.8$；$C$為$0.2$。
轉移機率矩陣$T=$
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
> $$P(Q)=P(q_1,q_2,...,q_T)$
$=P(q_T|q_{T-1},q_{T-2},...,q_1)P(q_{T-1},q_{T-2},...,q_1)$
$=P(q_T|q_{T-1},q_{T-2},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)P(q_{T-2},...,q_1)$
$...$
$=P(q_T|q_{T-1},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)...P(q_2|q_1)P(q_1)$
$=\prod_{t=1}^TP(q_t|q_{t-1},...,q_1)$

現在套用馬可夫性質

**1-th**

We know,
> $P(Q)=P(q_T|q_{T-1},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)...P(q_2|q_1)P(q_1)$
$\implies P(Q)=P(q_T|q_{T-1})P(q_{T-1}|q_{T-2})...P(q_2|q_1)P(q_1)$
$=\prod_{t=1}^TP(q_t|q_{t-1})$

**2-th**
> $P(Q)=P(q_T|q_{T-1},...,q_1)P(q_{T-1}|q_{T-2},...,q_1)...P(q_2|q_1)P(q_1)$
$\implies P(Q)=P(q_T|q_{T-1}.q_{T-2})P(q_{T-1}|q_{T-2},q_{T-3})...P(q_3|q_2,q_1)P(q_2|q_1)P(q_1)$
$=\prod_{t=1}^TP(q_t|q_{t-1},q_{t-2})$

**n-th**

最後可以推得在$n$階馬可夫時
> $$P(Q)=\prod_{t=1}^TP(q_t|q_{t-1},q_{t-2},...,q_{t-n})$$

#### Three fundamental problems
1. Likelihood : The Forward Algorithm
2. Decoding : The Viterbi Algorithm
3. Learning : The Forward-Backward Algorithm

### HMM基礎三問題的數學推演
- ### Likelihood(相似性問題)：Forwarding Algorithm
 本問題主要為計算發生某個觀測序列之最大機率，也就是把所有會發生此事件的路徑的機率加總即為所求。
 假設觀測序列$O$有$T$個觀測結果即$O=${$o_1,o_2,...,o_T$}，且每個$q$有$N$種狀況
> 
 目標： $P(O|\lambda)$, $\lambda$={$A,B,\pi$}
> 
> $$\implies P(O|Q)=\prod_{i=1}^T P(o_i|q_i)$$
> 
 透過一階馬可夫性質：
> $$P(O,Q)=P(O|Q)\cdot P(Q)=\prod_{i=1}^TP(o_i|q_i)\cdot \prod_{i=1}^TP(q_i|q_{i-1})$$
> $$\implies P(O) = \sum_QP(O,Q)=\sum_QP(O|Q)\cdot P(Q)$$
> $$=\sum_Q[\prod_{t=1}^TP(o_t|q_t)\cdot P(q_1)\cdot \prod_{t=2}^TP(q_t|q_{t-1})]$$
> 
得到的式子表示發生$O$的機率為找出所有$Q$的可能性並相加所有結果，但是這種遍尋的方式效率太低，因此我們將每個節點發生的機率記錄下來並用於下一次的計算，此稱為Forwarding Algorithm。
> 
 在此定義$a_t(j)$ 表示在$j$狀況下時，且觀察到$o_t$的機率
 
> $$a_t(j)=P(o_1,o_2...,o_t,q_t=j|\lambda)$$
> $$\implies a_t(j)=\sum_{i=1}^Na_{t-1}(i)a_{ij}b_j(o_t)=\sum_{i=1}^NP(o_1,o_2,...,q_{t-1}=i|\lambda)a_{ij}b_j(o_t)$$ 
並定義$b_j(o_t)$表示在$j$狀況之下發生$o_t$的機率

> $$b_j(o_t)=P(o_t|q_j)$$
得到上述式子後我們可以總結出Forwarding Algorithm的步驟

1. Initialization
> $$a_t(j)=\pi_jb_j(o_1)$$
2. Recursion
> $$a_t(j)=\sum_{i=1}^Na_{t-1}(i)a_{ij}b_j(o_t) \quad,1\leq j\leq N,\quad 1< t\leq T$$
3. Termination
> $$P(O|\lambda)=\sum_{i=1}^Na_T(i)$$
> 
經過該演算之後即可得到發生序列$O$的最大機率，為Likelihood問題的解

- #### Decoding(解碼)：Viterbi Algorithm
本問題主要在找出造成觀測序列$O$最有可能的隱序列$Q$，在每次的遞迴中記錄著當次找到的狀態$q$，在結束遞迴後trace back即可找出該隱序列。

目標：$Q^*=\arg \max_QP(O,Q|\lambda)$

首先，與Likelihood問題的式子相似
> $$a_t(j)=\sum_{i=1}^Na_{t-1}(i)a_{ij}b_j(o*-t)$$

定義$V_t(j)$為在觀測到$o_1...o_t$且經過$q_1...q_{t-1}$路徑且當前狀態為$j$的最大機率
> $$v_t(j)=\max_{q_1...q_{t-1}}P(q_1,...,q_{t-1},o_1,...o_t,q_t=j|\lambda)$$
> $$=\max_{i=1}^Nv_{t-1}(j)a_{ij}b_j(o_t)$$

不同於Likelihood是找所有可能的加總，Decoding是找最大的可能並將路徑記錄下來，即為答案(Viterbi Path)，此處為了記錄路徑，**定義$k_t(j)$為在地$t$個觀測結果時隱狀態為$j$的前一狀態**，因此歸納出以下步驟。
1. Initialization
> $$v_1(j)=\pi_jb_j(o_1) \quad 1\leq j\leq N$$
> $$k_1(j)=0 \quad 1\leq j\leq N$$
2. Recursion
> $$v_t(j)=\max_{i=1}^Nv_{t-1}(i)a_{ij}b_j(o_t) \quad 1\leq j\leq N , 1<t\leq T$$
> $$k_t(j)=\arg \max_{i=1}^Nv_t(j)=\arg \max_{i=1}^Nv_{t-1}(i)a_{ij}b_j(o_t) \quad 1\leq j\leq N , 1<t\leq T$$
3. Terminal: The best probaility and start of backtrace
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