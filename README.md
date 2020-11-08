Lagrange Multiplier - 拉格朗日乘數
==

### 目的
用於在數學的**最佳化**問題中，如果一個多元函數其變數受到一個或多個條件約束時求其極值。這種方法可以將有$n$個變數和$k$個約束條件的**最佳化**問題轉換成$n+k$個方程式的解。在此方法中引入一個或一組新的未知數$\lambda$(即拉格朗日乘數)當作各個約束條件方程式的梯度向量之線性組合的係數。

例如，欲求$f(x,y)$在受$g(x,y)=c$的束時的極值，在引入拉格朗日乘數$\lambda$後寫成拉格朗日函數，這時只需求該拉格朗日函數的極值：
> $$L(x,y,\lambda)=f(x,y)+\lambda \cdot \left( g(x,y)-c)\right)$$
> 
需注意用此種方式求出來的極值點會包含原問題的所有極值點，但不保證每個極值點都是原問題的極值點。

### 原理
假設有一函數$f(x,y)$受到$g(x,y)=c$約束，即欲求$f(x,y)$的極值且滿足$g(x,y)=c$，$c$為常數，假設$d_n$為$f$的等高線即$f(x,y)=d_n$，可以想像在某$d_n$時$f(x,y)$與$g(x,y)=c$會相交，而當兩函數在某相交的點時相切時，則該點為兩函數之極值。

![](https://i.imgur.com/AYHxBmD.png)

而當兩函數相切時則，兩函數之梯度向量具有以下關係：
> $$\nabla f(x,y)=-\lambda \nabla \left( g(x,y)-c\right)$$
> $$\implies \nabla \left[ f(x,y)+\lambda (g(x,y)-c) \right]=0$$

接著透過適當的方法求出$\lambda$並套入
> $$F(x,y, \lambda)=f(x,y)+\lambda \left( g(x,y)-c\right)$$

當$F(x,y, \lambda)$等於$f(x,y)$即為該問題的最佳解，因為在該點$g(x,y)-c=0$

### 證明
假設在$f(x,y)=k$時有極值$A$，且有一常數函數$g(x,y)=c$，在兩者極直接在$A$的情況下，兩者的全微分為：
> $$df=\frac{\nabla\ f}{\nabla\ x}dx+\frac{\nabla\ f}{\nabla\ y}dy=0$$
$$dg=\frac{\nabla\ g}{\nabla\ x}dx+\frac{\nabla\ g}{\nabla\ y}dy=0$$

由於兩者在有極值的情況下為相切，所以兩者該點的切線斜率呈現一個比例關係，

> $$\frac{\frac{\nabla f}{\nabla x}}{\frac{\nabla g}{\nabla x}}=\frac{\frac{\nabla f}{\nabla y}}{\frac{\nabla g}{\nabla y}}=-\lambda$$
$$\implies \frac{\nabla f}{\nabla x}+\lambda \frac{\nabla g}{\nabla x}=0$$
$$\implies \frac{\nabla f}{\nabla y}+\lambda \frac{\nabla g}{\nabla y}=0$$

最後分別乘上$dx,dy$後積分即可得到新的函數
> $$L(x,y,\lambda)=f(x,y)+\lambda g(x,y)$$

至此我們已將原本的問題轉換成該數學式的最佳解。
