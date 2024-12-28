# Echo State Network (ESN) - Minimum C Implementation

## 概要

このリポジトリには、**最小限のEcho State Network (ESN) をC言語で実装したサンプル**が含まれています。  
あくまでも動作原理をシンプルに示すことを目的としており、学術的・実用的に厳密な最適化や汎用ライブラリの利用は行っていません。

## コードの特徴

1. **パラメータ設定**  
   - 入力次元 `N_INPUT`、リザバーの次元 `N_RESERVOIR`、出力次元 `N_OUTPUT` といった基本パラメータを定義します。  
   - リーク率 `ALPHA` などのハイパーパラメータもここで定義しています。

2. **初期化**  
   - リザバーの重み行列 `W` と入力重み行列 `W_in` をランダムに生成します。  
   - 出力重み行列 `W_out` は学習前はゼロ初期化します。  
   - リザバー状態ベクトル `x` もゼロで初期化します。

3. **リザバー状態更新**  
   - 以下の更新式に基づき、1ステップごとにリザバー状態を更新します。

   <img src="https://latex.codecogs.com/svg.image?\mathbf{x}(t&plus;1)=(1-\alpha)\,\mathbf{x}(t)\;&plus;\;\alpha\,\tanh\!\bigl(\mathbf{W}_{\text{in}}\mathbf{u}(t)\;&plus;\;\mathbf{W}\,\mathbf{x}(t)\bigr)." />

4. **出力計算**  
   - 更新したリザバー状態 <img src="https://latex.codecogs.com/svg.image?\(\mathbf{x}(t)\)" /> に対し、

   <img src="https://latex.codecogs.com/svg.image?\mathbf{y}(t)=\mathbf{W}_{\text{out}}\;\mathbf{x}(t)." />

   で出力を求めます。

5. **学習 (Ridge回帰)**  
   - 出力重み <img src="https://latex.codecogs.com/svg.image?\(\mathbf{W}_{\text{out}}\)" /> はリザバーの状態と教師データからのみ学習します。  
   - 以下のRidge回帰を用いて更新します。

   <img src="https://latex.codecogs.com/svg.image?\mathbf{W}_{\text{out}}=\mathbf{D}\,\mathbf{X}^\top\bigl(\mathbf{X}\,\mathbf{X}^\top&plus;\lambda\,\mathbf{I}\bigr)^{-1}." />

   - <img src="https://latex.codecogs.com/svg.image?\(\mathbf{X}\)" /> は各時刻でのリザバー状態の履歴。  
   - <img src="https://latex.codecogs.com/svg.image?\(\mathbf{D}\)" /> は教師出力の履歴。

6. **テスト(予測)**  
   - 学習で固定した <img src="https://latex.codecogs.com/svg.image?\(\mathbf{W}_{\text{out}}\)" /> を使い、任意のテスト入力に対して1ステップごとに予測を行います。

## 実行方法

1. ターミナルでこのディレクトリに移動してください。
2. `gcc sample.c -o esn_minimal -lm` などとしてコンパイルします。  
   - `-lm` は `math.h` を使う場合に必要となります。
3. `./esn_minimal` を実行すると、簡単なサンプル予測結果が表示されます。

## 注意点・補足

- **行列演算**はすべてループベースで直接行っており、大きなネットワークには向きません。  
  実際にはBLASやEigenなどのライブラリを使うことが推奨されます。  
- **スペクトル半径**の厳密な計算は省略し、`RHO_INIT` で一律にスケールしています。本来は最大固有値を計算してスケーリングする必要があります。  
- **ウォーミングアップ(初期区間の破棄)** は行っていないため、リザバーが十分に励起されていない初期段階の状態も学習に使われます。  
- **ハイパーパラメータ** (リーク率、正則化係数など) は適宜調整する必要があります。  
- 学習データやテストデータは仮置きの `sin` や `cos` で生成しているため、実際には各種時系列タスク用のデータを用意してください。

## 参考

- Jaeger, H. (2001). *The “echo state” approach to analysing and training recurrent neural networks*. GMD-Forschungszentrum Informationstechnik.
- [リザバーコンピューティングとは？(株式会社QuantumCore)](https://www.qcore.co.jp/reservoir/)
