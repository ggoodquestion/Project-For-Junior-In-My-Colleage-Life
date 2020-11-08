DTW - 動態時間規整(Dynamic Time Warping)
==

### 簡介
給定兩向量，欲計算兩者間的距離時我們常使用歐幾里得距離(Euclidian Distance)來計算，例如欲計算$X(4,7), Y(7,11)$的距離應為
> $$D(X,Y)=\sqrt{(4-7)^2+(7-11)^2}=5$$
> 
但當兩向量長度不同時，顯然使用歐幾里得距離並不是個好選擇，如下圖，某些點距離另一點距離甚至比其對應歐幾里得距離的點來的更近。一般來說，假設兩向量的元素位置代表的是時間，由於兩向量時間軸可能會有所偏差，因此我們並不清楚點對點的對應關係，因此傳統的歐幾里得距離並不適合用於此情況，我們需要一套有效的運算方法，此方法為DTW，我們稱計算出來的數值為"*DTW distance*"，在此我們更樂意將距離稱為相似性，而所謂的距離是數學上用來量化相似性所給的名稱。
![](https://i.imgur.com/fhFC0cL.png)
### 參數定義
DTW的目的在於比對兩個序列的相似性，假設給定一個特徵空間(Feature Space)$F$和兩組序列$X=(x_1,x_2,...,x_N),Y=(y_1,y_2,...,y_M)\quad x_n,y_m\in F,\quad for \quad 1\leq n\leq N,\quad 1\leq m\leq M$，接著我們需要一個表(table)來記錄點與點之間的距離，在此定義$C$為一個$F$ X $F$的矩陣
> $$C(n,m)=cost \ of \ x_n \ and \ y_m$$

一般來說當$C(n,m)$越小時表示兩點的相似度越高(*low cost*)，越大時相似度越低(*high cost*)。接著我們還需要將走過的路徑(*DTW path*)記錄下來，在此定義序列$p=(p_1,p_2,...,p_L)$，且$p_l=(n_l,m_l)\in N$ X $M \ matrix ,\quad 1\leq l \leq L$ 表示第$l$個點為$(x_{n_l},y_{m_l})$。

再來需要一個表(table)來記錄每個點累積所需最短的消耗(cost)，在這個演算法中，我們需要透過每個點所累績的的最短路徑來找到最短的DTW path，因此定義
> $$D(n,m)=\min\begin{Bmatrix}D(n-1,m),D(n,m-1),D(n-1,m-1)\end{Bmatrix}+C(n,m)$$

### 初始條件
由於*DTW*在於對齊並校正兩不同長度的序列，首先必須先將兩序列的起始位與結束位置對齊，也就是說可能會對兩序列對時間軸進行壓縮，並且時間對於點需要有單向性(*Monotonicity*)即兩序列的相似狀況不會發生順序上的對調，最後在比對的的時候必須照順序比對點不可跳過(*point by point*)才不會發生相似性遺漏的狀況，以下列出做*DTW*演算時需滿足的條件。

1. 邊界條件($Boundary \ condition$)：$p_1=(1,1),p_L=(N,M)$
2. 單向性($Monotonicity$)：$n_1\leq n_2\leq ...\leq n_L \ and \ m_1\leq m_2\leq ...\leq m_L$
3. 依序性($Step \ size \ condition$)：$p_{l+1}-p_l\in \begin{Bmatrix} (0,1),(1,0),(1,1)\end{Bmatrix}$

### 演算過程
定義完參數且設定好初始條件後開始要來實際演算，因為*DTW*的條件有單向性和依序性，這樣的問題符合最佳子問題的性質，可以根據前一次找到的最短路徑點繼續推算下一個的最短路徑，並且未來的改變不影響過去的結果，因此可以使用動態規劃(*Dynamic Program*)來解決*DTW*問題。一開始我們並不知道最好的結果是什麼，而最好的結果就是最短的DTW path，也就使我們需要找到最好的$D(N,M)$，所以首先要先初始$D$，再來透過找到的$D$去backtrace找到DTW path，所以這個演算我們把它分為兩個步驟，(i)初始化$D$ (ii)找到路徑序列$p$。


由此我們可以歸納出*Dynamic Time Warping Algorithm*的步驟：

首先要先找出最佳的累積路徑距離

1. Initialization
> $$D(1,1)=C(1,1)$$
2. Rescursion
> $$D(n,m)=\min\begin{Bmatrix}D(n-1,m),D(n,m-1),D(n-1,m-1)\end{Bmatrix}+C(n,m)$$
3. Terminal
> $$D(N,M)=\min\begin{Bmatrix}D(N-1,M),D(N,M-1),D(N-1,M-1)\end{Bmatrix}+C(N,M)$$

再來從已知的$D$找最佳路徑$p$

1. Initialization
> $$p_L=(N,M)$$
2. Recursion
> $$p_l=\arg\min_{(n,m)}\begin{Bmatrix}D(n-1,m),D(n,m-1),D(n-1,m-1)\end{Bmatrix}$$
3. Termination
> $$p_1=(1,1)$$

### 簡易範例
給定$F=\begin{Bmatrix}\alpha ,\beta ,\gamma \end{Bmatrix}$ ，假設$X=(\alpha,\beta,\gamma),Y=(\alpha,\alpha,\beta,\gamma,\beta,\gamma),Z=(\alpha,\gamma,\gamma,\beta,\beta)$，定義
> $$C(n,m)=\begin{cases}0, \ x_n=y_m \\ 1, \ x_n \neq y_m \end{cases}$$
> 
找出與$X$相似的序列。

1. 首先初始化參數，可將$C_{X,Y}與C_{X,Z}$畫成以下兩圖，
以看出$X,Y$的路徑$p_{X,Y}$與$X,Z$的路徑$p_{X,Z}$
![](https://i.imgur.com/2wVNQzc.jpg)
![](https://i.imgur.com/pyj1Sir.jpg)

2. 計算Warping Distance：
透過$D_{X,Y}(N,M)=1, \ D_{X,Z}(N,M)=3$得出序列$Y$與序列$X$最為相似