# GenerateImageFromUserPreference

## 目的
「奇跡の一枚」を超えるハシカン画像の生成

## 提案手法
高解像度画像生成器「PGGAN」とランク学習器「RankNet」を組み合わせて、ユーザー好みの画像を生成する。

具体的にはPGGANのGeneratorのLossにRankNetの出力を逆数にして足す。

## 参考
### RankNet
* [RankNetを実装してランキング学習](https://qiita.com/kzkadc/items/c358338f0d8bd764f514)
* [ニューラルネットワークを用いたランク学習](https://qiita.com/sz_dr/items/0e50120318527a928407)

### PGGAN
* [PGGAN「優しさあふれるカリキュラム学習」](https://qiita.com/Phoeboooo/items/ea0e44733e2d2240879b)
* [GANを使って簡単に架空アイドル画像を自動生成](https://www.mahirokazuko.com/entry/2018/12/15/201501)
* [ニートの僕が幼女の顔生成に挑戦](https://qiita.com/pnyompen/items/412734d244d7ebb45ca7)
