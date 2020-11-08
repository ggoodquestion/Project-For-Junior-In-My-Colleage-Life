Cepstrum - 倒頻譜
==

## 動機
系統的輸出信號可以視為輸入信號與impulse response的convolution。有時，我們需要個別的信號進行研究或處理，而分離這兩個信號的過程稱為deconvolution。

當我們知道輸入信號或impulse response的話，可以通過傅立葉的convolution property得到另一個信號。而當輸入信號及impulse response都是未知的，這時可以使用倒頻譜分析。

語音由excitation source和vocal tract system(聲道系統)組成。為了獨立地分析和建模語音的excitation和system components，必須將這兩個成分分開。使用倒頻譜分析的目的是將語音分為source和system components ，而無需事先知道它們。

假設$e(n)$為excitation sequence，$h(n)$為vocal tract filter sequence，則 speech sequence $s(n)$ 可以下列表示 : 
>$s(n) = e(n) * h(n)$ $\implies S(\omega) = E(\omega)H(\omega)$

而我們想要將語音序列在時域中分解為source和system components，因此，必須將頻域中兩個分量的乘法轉換為兩個分量的線性組合。所以，倒頻譜分析用於**將頻域中相乘的兩分量轉換為倒頻譜域中兩個分量的線性組合**。
### 原理
語音序列的magnitude spectrum為 :  
>$｜S(\omega)｜ = ｜E(\omega)｜｜H(\omega)｜$

取對數後 :   
>$log｜S(\omega)｜ = log｜E(\omega)｜+log｜H(\omega)｜$

在頻率域中，vocal tract components以在低頻區域中緩慢變動的成分表示，而excitation components由高頻率區域中快速變化的成分表示。  
若想將其轉回時域，我們需使用$IDFT$，不同的是，linear spectrum的$IDFT$轉換回時域，而log spectrum的$IDFT$轉換為**quefrency**時域或**倒譜域**，類似於時域 : 
>$c(n) = IDFT\lbrace{log｜S(\omega)｜\rbrace} = IDFT\lbrace{log｜E(\omega)｜+log｜H(\omega)｜\rbrace}$

### 公式
>$cepstrum = IDFT\lbrace{spectrum\rbrace}$  
$\implies c[n] = IDFT \lbrace{ \ log|DFT\lbrace{x[n]\rbrace}|\  \rbrace} = \sum_{n=0}^{N-1}log(|\sum_{n=0}^{N-1}x[n]e^{-j\dfrac{2\pi kn}{N}}|)e^{\dfrac{j2\pi kn}{N}}$

![](https://i.imgur.com/uIdDap5.png)
![](https://i.imgur.com/ymQT2ps.png)

### Liftering
如同濾波器(filter)常使用在頻譜上，倒濾波器(lifter)就是在倒頻譜上所使用的濾波器。藉由在想要的倒頻譜區間乘以一個rectangular window來取得該區間的值，有low-time liftering和high-time liftering，分別取出vocal tract components和excitation components。
>low-time liftering
![](https://i.imgur.com/bLCe7j5.png)

>high-time liftering
![](https://i.imgur.com/j1aNDMK.png)



### 意義
時域信號利用Fourier Transform可轉換為頻率函數或功率頻譜密度函數，如果頻譜圖上呈現出複雜的週期結構而難以分辨時，將頻譜視為一種新的信號做處理 - 取對數後進行Inverse Fourier Transform，可以使週期結構呈現便於識別的譜線形式，所得圖形為倒頻譜，有複數倒頻譜及實數倒頻譜。有時為方便計算，會將原來信號的頻譜先轉成類似分貝的單位，再作逆傅立葉變換。

### 觀念
頻譜圖上的獨立變數是頻率，而倒頻譜圖上的獨立變數為倒頻率(quefrency)，倒頻率是一種時間的度量單位。假設聲音頻號取樣速率為44100赫茲，且倒頻譜上有個很大的值在倒頻率等於100，代表實際上在44100/100=441赫茲有很大的值，因其在頻譜上週期性出現而顯現在倒頻譜上。

### 應用
1. 倒頻譜可以被視為在不同頻帶上變化速率的信息，倒頻譜一開始被發明在地震或炸彈產生的地震回音，現今也被使用在分析雷達信號，以及信號處理等問題。
2. 倒頻譜在處理人聲信號以及音樂信號有非常好的效果，例如梅爾頻率倒頻譜(Mel-Frequency Cepstrum)，用來做聲音的辨認，偵測音高等。近年來梅耳倒頻譜也被應用在音樂信息的回复。
3. 倒頻譜用在處理多路徑問題時(如聲波的回音、電磁波的折、反射等)，如果將其他路徑干擾視為噪聲，為了消除噪聲，利用倒頻譜，不需測量每條多路徑的延遲時間，可以利用傳送多次信號，觀察其他路徑在倒頻譜上的效果，並且加以濾除。
4. 語音大致上是由音高、聲帶脈衝、聲門波形所組成，我們可以利用倒頻譜將這三種元素在倒頻域上分開，以利於做語音信號的分析。

### 倒濾波器 - lifter
如同濾波器(filter)常使用在頻譜上，倒濾波器就是在倒頻譜上所使用的濾波器。低通的倒濾波器跟低通濾波器類似，它藉由在倒頻譜上乘以一個window係數，使倒頻譜上的高倒頻率被壓抑，如此一來，當信號轉回時域空間時會變成一個較平滑的信號。

### MFCC- 梅爾頻率倒頻譜係數(Mel-Frequency Cepstral Coefficients)
>**Mel scale - 梅爾刻度**     
公式 : $Mel(f) = 2595log_{10}(1+\dfrac{f}{700})，f:赫茲$  
為非線性刻度單位，表示人耳對等距音高(pitch)變化的感官。其參考點定義是將1000Hz，且高於人耳聽閾值40分貝以上的聲音信號，定為1000mel。在頻率500Hz以上時，人耳每感覺到等量的音高變化，所需要的頻率變化隨頻率增加而愈來愈大。這樣的結果是，在赫茲刻度500Hz往上的四個八度(一個八度即為兩倍的頻率)，只對應梅爾刻度上的兩個八度。Mel的名字來源於單詞melody，表示這個刻度是基於音高比較而創造的。