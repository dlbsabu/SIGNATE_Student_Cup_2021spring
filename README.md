![image](https://user-images.githubusercontent.com/68833240/117968196-e4b17700-b360-11eb-9d20-c9ca02344624.png)
# SIGNATE_Student_Cup_2021spring

SIGNATE Student Cup 2021春【予測部門】のリポジトリ
- result
  - public: 0.6590873
  - private: 0.7053295
  - rank: 2nd place / 330 :congratulations: :clap:

## 目次
[Basics](#basics)  
[Features](#features)   
[Final model](#final)  
[Notebook](#notebook)    
[Solution](#solution)    

<a id="basics"></a>
## Basics
予測部門では、楽曲の長さや人気度、アコースティック度といった様々な曲の特徴から、その曲がどのジャンル（全11種類）に該当するかを推定


<a id="features"></a>
## Features
- nagiss氏のフォーラムより[url](https://signate.jp/competitions/449/discussions/lgbmknn-lb06567-1?comment_id=3493#3493)
  - num_nas, tempo, one-hot region, CountEnc region, LabelEnc region, log tempo
  - agg_zscore_[]_grpby_region
  - standardscaled_[] 
  - knn
  
- 自作特徴量
  - standardscaled_[]を2つずつ取り出し四則演算した特徴量('2つ組みの特徴量') 
  - standardscaled_[]を3つずつ取り出し和、積を計算した特徴量('3つ組みの特徴量')
  - standardscaledの代わりにrankgaussスケーリング
  - null importanceによる特徴量選択[参考](https://qiita.com/trapi/items/1d6ede5d492d1a9dc3c9)

- あんまりうまく行ってないもの、または使っていないもの
  - 次元削減（umap）
  - standardscaled_[]特徴量に対して、行の"sum", "max", "min", "mean", "median", "mad", "var", "std", "skew", "kurt"


<a id="final"></a>
## Final model
- simple_lgb_model
  - nigss氏のもの[LGBM+KNN](https://signate.jp/competitions/449/discussions/lgbmknn-lb06567-1?comment_id=3493#3493)がベース
  - '2つ組みの特徴量', '3つ組みの特徴量'をすべて追加したのち、null importanceの重要度が低い特徴量を削除（手違いでこちらをアンサンブルに組み込んでしまった）
  - '2つ組みの特徴量'をすべて追加, '3つ組みの特徴量'のうちnull importanceの重要度が高い特徴量のみ追加（よりよいCVが出ていたのでこちらを使う予定だった）

- pseudo_model
  - nigss氏のもの[Pseudo-Labeling](https://signate.jp/competitions/449/discussions/pseudo-labeling-lb06630?comment_id=3513#3513)がベース
  - 0.95, 0.925, 0.9, 0.875, 0.85とどんどんpseudo labelを増やしているが、さすがに過学習してしまうのではないか？と感じたため、0.95のみにしてみた。
  - Stratified K-foldのseed値を変えた。

- tabnet_model[参考](https://zenn.dev/sinchir0/articles/9228eccebfbf579bfdf4)
  - 欠損値補完
  - num_nas, tempo, one-hot region, CountEnc region, LabelEnc region, log tempo
  - agg_zscore_[]_grpby_region
  - standardscaled_[] 
  - standardscaledの代わりにrankgaussスケーリングしたもの
  - knn
  - '2つ組みの特徴量', '3つ組みの特徴量'のうち、null importanceが結構高いもののみ採用(特徴量増やしすぎても過学習するだけなので、lightGBMより少なめ)
  
以上3つのモデルのアンサンブル(1:1:1)  

ポイントとしては、
- train:test =　1:1とtrainが少ないのでpseudo labelingは効くだろうとおもった。
- lightGBMなどの木モデルでは特徴量間の四則演算が効く事が多い(2つ組みや3つ組みのこと)
- lightGBMは特徴量をたくさん増やしてもうまいこと特徴量を選んでくれるのでどんどん作る。
- とはいえ3つ組みまで足すと増えすぎてCVが悪くなったのである程度特徴量選択をしないとノイズになるだけ。-> 今回はumap等の次元削減よりもnull importanceがうまく行った
- アンサンブルの多様性のためにrandom seed averageもする。
- アンサンブルの多様性のためにNN系も追加。今回はtabnetをためした。
- Trust CV  
といったところか

<a id="notebook"></a>
## Notebook
logも兼ねている
**nb000**
- simple EDA

**nb001**
- Pycaretで実験
- LB:0.5347709

**nb002**
- ベースラインを構築したが、うまく行かないので、他者のベースラインを待つ

**nb003**
- nagiss氏のベースラインをそのまま
- CV:0.66055, LB:0.6567770

**nb004**
- nb003からregionのone-hotを除いた
- CV: 0.54870, LB:0.5759619と大幅に悪化
- Feature importanceが低いとはいえ、やはりlightGBMは特徴量をむやみに減らさない方がよさげ。モデルに選んでもらおう。

**nb005**
- nb003のstandardscaled_したカラムに対してadd, subtract, multiply, divideを計算したカラムを追加
- CV:0.66147, LB:  0.6505313
- CV改善、LB悪化

**nb006**
-  欠損値補完のためのnb
- /data/hokan.pklに

**nb007**
- umapで次元削減しようとした。nanの存在によりうまく行かないので、nb006で欠損補完
- 一旦umapなしで提出
- CV:0.65329, LB:0.6519563 
- CV悪化、LB改善
- 欠損値補完自体はCV悪化

**nb008**
- umapでadd, subtract, multiply, divideを５次元に次元削減。
- CV:0.66103, LB:0.6486595 
- umapの効果だけみるとCV改善、LB悪化

**nb009**
-  欠損値補完のためのnb, fold N=15に
- /data/hokan_15.pklに

**nb010**
- 欠損値補完fold N=15にしたやつでnb007と同じ処理
- CV:0.65833, LB:0.6530840
- 欠損値補完はN=15使ったほうがよい

**nb011**
- 行のmin, max等々の特徴量を追加
- CV:0.65387, LB:0.6537329
- 微妙

**nb012**
- 3つ組の特徴量（和、積）を追加
- CV:0.64870, LB:0.6501104
- umapで64次元に削減
- CV:0.65646, LB:0.6409287
- 微妙
- lgbmは特徴量増やしても選んでくれるが、さすがに特徴量が多すぎてノイズになったか？umapで減らしてみるとまぁまぁCV良くはなるが


**nb013**
- tabnetを試してみる
- CV:0.61321859, LB:0.2581344
- なんじゃこりゃ

**nb014**
- standard_まで特徴量作成、tabnet
-  CV:0.7442259, LB:0.5666629
-  CV変な値やけど、よくなってはいるか

**nb015**
- tabnetの事前学習というものをした。よーわからん。
- CV:0.62654275, LB:0.4705204

**nb016**
- nagiss氏のknn
- CV=0.652475359, LB=0.6457052


**nb017**
- nb005のknn少し変更

**nb018**
 - nb10のコピー
- 欠損値補完fold N=15にしたやつでnb007と同じ処理, minとかはつくらない
- umapで64次元に圧縮
- CV:0.65624, LB:0.6512785
- 微妙

**nb019**
- aggしたものも全部含めて
- CV:0.65727, LB6513462
- 少し改善

**nb020**
- nagiss氏のpseudoそのまま

**nb021**
- pseudo
- nagiss氏のものは0.95, 0.925, 0.9, 0.875, 0.85とどんどんpseudoを増やしているが、過学習してしまうのでは？と感じたため、0.95のみにしてみた。CVが信用できなくなるので評価のしようはないが。

**nb022**
- アンサンブルnb005, 0b014, nb021
- LB:0.6573835
- 微妙

**nb023**
- rankgaussスケールがnn系には効くらしいと聞き導入
- CV:0.7906824544494929, LB:0.5692701
- tabnetにしてはよさげ

**nb024**
- tabnet でいろいろやるが上がらず
- 最終日に改定knn追加
- CV:0.7837376657よさげ

**nb025**
- umapだけでは特徴量削減でイマイチぱっとしない
- 特徴量選択の手法についてもう少し調べてみる
- null importanceやってみる・・・？
- 必要そうな特徴量を抽出

**nb025-2**
- 続き。閾値を変えて、特徴量をいろいろ書き出してみる。
- 逆にめちゃくちゃ低い値のものも書き出す。

**nb026**
- 2つ組みの特徴量のなかで、閾値を超えたものだけ追加
- CV:0.66092, LB:0.6559823
- ええやん

**nb027**
- アンサンブルnb005, 0b023, nb021
- LB:0.6561361

**nb028**
- 2つ組を全部追加、null impoの良さげな3つ組のみ追加
- CV:0.66383, LB:6544056
- とても良い。

**nb029**
- 2, 3つ組生成、その後null importanceの悪いものを削除
- CV:0.65986
- さすがに特徴量減らし過ぎたか？

**nb030**
- tabnet
- 2, 3つ組みの特徴量のうち、null importanceの閾値が30を超えたものを採用
- (NN系は特徴量増やしすぎても過学習するだけなので、lightGBMより少なめ)
- knnも追加
- CV:0.790179448, LB:0.6279500
- めちゃくちゃいい

**nb031**
- アンサンブルnb028, nb021, nb030（後で確認したら手違いでnb029になっていた...。）
- 2:2:1の重みで
- LB:0.6559510
- えっえっ低っ
- なきそう

**nb032**
- アンサンブルnb028, nb021, nb030（後で確認したら手違いでnb029になっていた...。）
- 1:1:1の重みで
- LB:0.6590873
- いやー微妙やなこれ
- 結局最後までスコア上がらんかったわ

