GMM - 高斯混和模型(Gaussian Mixture Model)
==

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

