## (title) TORE: TOken REcycling method for efficient sequential predictions with Vision Transformers


## Abstract

When running sequential decision making algorithms, it's often necesary to sequentialy query a model with increasing amount of available information. An example is Active Visual Exploration (AVE), which is an environment exploration problem based on increasing number of visual ques. Research in this area has led to the design of accurate methods utilizing emerging properties of self-supervised Vision Transformers (VITs). AVE is a commonly faced problem by autonomous robots, however existing methods focus on accuracy rather than efficiency. To address this problem we propose simple, effective and general VIT forward pass modification technique called TORE. Training a model with TORE modification doesn't require additional hyperparameter tuning, it doesn't destroy the emerging properties of self-supervised VITs used for AVE, and inference with TORE greately reduces the amount of computation used for sequential model predictions. Finally we conduct series of experiments investigating the effect of TORE on representations produced by different supervised and self-supervised models.


## 3 Filary
1. Pokazujemy, że nasza metoda działa na supervised modelach, można ją zaaplikować do algorytmów, które rozdzielają decision making od prediction accuracy. Pokazujemy zysk w FLOPsach
2. Pokazujemy, że nasza metoda nie niszczy emerging properties MAE i można dalej odpalić na nim AMEM, tylko że mamy teraz lepsze accyracy dla większych K.
3. Żeby lepiej zrozumieć jaki wpływ mają podziały KM na predykcje.Analizujemy klasy, na których różnice w accuracy deit i compdeit są największe i analizujemy mapy antencji przy predykcji. Ponadto, badamy przekrój moideli self-supervised i jak KM wpływa na uzyskiwane reprezentacje.


<span style="color:lightgreen">

# Experimental setup i motywacja
**Wprowadzamy experimental setup i argumentujemy sensowność przez active visual explorartion**

- wprowadzenie eksperymentu lin. acc. KM
- wprowadzenie eksperymentu lin. acc. seq.



# End-to-end training is possible
**Pokazujemy, że da się łatwo wytrenować model end-to-end, stabilnie i baz zmiany hiperparametrów**

- Porównanie deitsmall vs compdeitsmall na naszych dwóch eksperymentach
- Pokazanie, że na imagenetv2 wykresy zachowują się tak samo więc nie ma overfittingu
- pokazanie różnicy na flopsach


</span>

<span style="color:pink">

(*) Trzeba jeszcze resumować trening.

</span>

<span style="color:lightgreen">

# Finetuning during transfer learning
**Sprawdzamy które modele najłatwiej dotune'ować podczas transfer learningu na inne datasety**

- wprowadzenie policy treningu, z każdej metody produkujemy 2 modele. Jeden transferujemy losując KM, drugi transferujemy klasycznie
- porównanie par modeli za pomocą lin. acc. KM

*Przy transfer learningu używamy zawsze tej samej metody, wziętej z deita. Porównanie w tabelce accuracy dla KM=[(0,1), (4,16), (8,16)] pomiędzy każdą parą modeli. Dla (0,1) patrzymy, która metoda ma najmniejszy spadek, dla (4/8,16) która ma największy zysk.*

</span>

<span style="color:pink">

(*) Trzeba poodpalać te treningi, zebrać dane i zrobić tabelkę, ale generalnie wszystko jest napisane i działa.

</span>


<span style="color:red">

# Dependence of emerging properties on KM
**Sprawdzamy czy emerging properties zostają zachowane przy naszym rodzaju treningu**
	
- Fine tuning MAE, ale z treningiem MAE -> dostajemy ftMAE
- Porównujemy AMEM classifficarion head-only na SUN360 na oryginalnym MAE i ftMAE dla K=0 i K=8

</span>
<span style="color:pink">


# Qualitative analysis of features dependent on KM
**sprawdzamy dokładniej jak mają się feature'y dla K w porównaniu do K=0 dla różnych modeli**

</span>




<span style="color:lightgreen">

### Effect of self-supervised learning
**Pokazujemy, że różne metodologie uczenia dają różne kształty krzywych linacc KM**

- wprowadzenie reprezentatów różnych metod self-supervised
- narysowanie krzywych lin. acc. KM, pokazanie różnicy w modelach

<span style="color:pink">

(*) Trzeba jeszcze raz przetrenować głowy/głowę do moco.

</span>


</span>




<span style="color:lightgreen">

- średnia odległość featureów dla kolejnych K od K=0
	- mierzone przez l2, cos
	- mierzone przez accuracy na knn, gdzie treningowe są dla K=0, testowe są dla kolejnych K

</span>

<span style="color:pink">

- tSNE wykresy


- analiza klas, na których mamy największe różnice, analiza map atencji

</span>