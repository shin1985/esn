#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ----- パラメータ設定 ----- */
#define N_INPUT      1     // 入力次元
#define N_RESERVOIR  10    // リザーバ(隠れ状態)の次元
#define N_OUTPUT     1     // 出力次元

#define TRAIN_LEN    100   // 学習サンプル数
#define TEST_LEN     50    // テストサンプル数

#define ALPHA        0.3   // リーク率 (leaking rate)
#define RHO_INIT     0.9   // リザーバ重みのランダムスケール(簡易的に使用)

#define RIDGE_PARAM  1e-2  // リッジ回帰の正則化係数(簡易)

/* 
 * 簡易乱数生成:  -1.0 ~ +1.0 の範囲の値を返す 
 * rand() を使うので実装によっては精度等注意
 */
double rand_u(void) {
    return 2.0 * rand() / (double)RAND_MAX - 1.0;
}

/* 
 * tanh 関数を使用 (隠れ状態更新用) 
 */
static inline double activation(double x) {
    return tanh(x);
}

/* 
 * ESN の構造体を用意 
 */
typedef struct {
    double W_in[N_RESERVOIR][N_INPUT];     // 入力重み
    double W[N_RESERVOIR][N_RESERVOIR];    // リザーバ内部重み
    double W_out[N_OUTPUT][N_RESERVOIR];   // 出力重み (学習で求める)

    double x[N_RESERVOIR]; // リザーバの状態ベクトル
} ESN;

/* 
 * リザーバ状態 x(t+1) = (1 - alpha)* x(t) + alpha * tanh( W_in*u(t) + W*x(t) )
 * (最もシンプルな形式として、出力からのフィードバックは省略)
 */
void esn_update_state(ESN *esn, const double *u) {
    double new_x[N_RESERVOIR];

    // (W_in * u) + (W * x) を計算し、tanh をかける
    for(int i = 0; i < N_RESERVOIR; i++) {
        double sum = 0.0;
        // 入力からの寄与
        for(int j = 0; j < N_INPUT; j++) {
            sum += esn->W_in[i][j] * u[j];
        }
        // リザーバ内部の寄与
        for(int j = 0; j < N_RESERVOIR; j++) {
            sum += esn->W[i][j] * esn->x[j];
        }
        // tanh をかけてリーク率で混合
        new_x[i] = (1.0 - ALPHA) * esn->x[i] + ALPHA * activation(sum);
    }

    // x(t+1) を更新
    for(int i = 0; i < N_RESERVOIR; i++) {
        esn->x[i] = new_x[i];
    }
}

/* 
 * 出力 y(t) = W_out * x(t)
 */
void esn_calculate_output(ESN *esn, double *y) {
    for(int i = 0; i < N_OUTPUT; i++) {
        double sum = 0.0;
        for(int j = 0; j < N_RESERVOIR; j++) {
            sum += esn->W_out[i][j] * esn->x[j];
        }
        y[i] = sum;
    }
}

/* 
 * リザーバ重みと入力重みをランダム初期化 
 * (スペクトル半径調整など簡単に済ませる)
 */
void esn_init(ESN *esn) {
    // 入力重みをランダム初期化
    for(int i = 0; i < N_RESERVOIR; i++) {
        for(int j = 0; j < N_INPUT; j++) {
            esn->W_in[i][j] = 0.5 * rand_u(); 
        }
    }

    // リザーバ内部重みをランダム初期化
    for(int i = 0; i < N_RESERVOIR; i++) {
        for(int j = 0; j < N_RESERVOIR; j++) {
            // 簡易的にスケールを小さめに
            esn->W[i][j] = RHO_INIT * 0.5 * rand_u();
        }
    }

    // 出力重みは学習前は 0 初期化
    for(int i = 0; i < N_OUTPUT; i++) {
        for(int j = 0; j < N_RESERVOIR; j++) {
            esn->W_out[i][j] = 0.0;
        }
    }

    // リザーバ状態を 0 クリア
    for(int i = 0; i < N_RESERVOIR; i++) {
        esn->x[i] = 0.0;
    }
}

/* 
 * リッジ回帰による W_out の学習
 * X: N_RESERVOIR x TRAIN_LEN のリザーバ状態履歴 (縦N_RESERVOIR, 横TRAIN_LEN)
 * D: N_OUTPUT x TRAIN_LEN の教師出力データ
 * W_out を N_OUTPUT x N_RESERVOIR で更新 
 *
 * 非常に単純化した実装: (W_out) = D * X^T * (X * X^T + λI)^(-1)
 * ただし、転置行列や逆行列は直接計算しているため、サイズが大きくなると実用的ではありません。
 */
void train_ridge_regression(ESN *esn, double **X, double **D, int train_len) {
    // 行列サイズ
    // X は (N_RESERVOIR, train_len)
    // D は (N_OUTPUT,     train_len)
    // W_out は (N_OUTPUT, N_RESERVOIR)

    // 1) M = X * X^T : (N_RESERVOIR x N_RESERVOIR)
    double M[N_RESERVOIR][N_RESERVOIR];
    for(int i=0;i<N_RESERVOIR;i++){
        for(int j=0;j<N_RESERVOIR;j++){
            double sum = 0.0;
            for(int k=0;k<train_len;k++){
                sum += X[i][k]*X[j][k];
            }
            // 正則化項を対角成分に足す
            if(i == j) sum += RIDGE_PARAM;
            M[i][j] = sum;
        }
    }

    // 2) M_inv = M^(-1) (ガウス消去法などで計算: 簡易実装)
    //   大きな行列に対してはライブラリを使うべき
    double M_inv[N_RESERVOIR][N_RESERVOIR];
    {
        // 単位行列を作る
        for(int i=0;i<N_RESERVOIR;i++){
            for(int j=0;j<N_RESERVOIR;j++){
                M_inv[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // ガウス・ジョルダン法による M の逆行列計算
        for(int i=0;i<N_RESERVOIR;i++){
            // ピボット取得
            double pivot = M[i][i];
            // pivotが0に近い場合のチェック等省略
            double inv_pivot = 1.0/pivot;
            // ピボット行を正規化
            for(int col=0; col<N_RESERVOIR; col++){
                M[i][col]     *= inv_pivot;
                M_inv[i][col] *= inv_pivot;
            }
            // ピボット列を基準に他行を0にする
            for(int row=0; row<N_RESERVOIR; row++){
                if(row != i){
                    double factor = M[row][i];
                    for(int col=0; col<N_RESERVOIR; col++){
                        M[row][col]     -= factor * M[i][col];
                        M_inv[row][col] -= factor * M_inv[i][col];
                    }
                }
            }
        }

        // ここで M_inv に M の逆行列が入る
        for(int i=0;i<N_RESERVOIR;i++){
            for(int j=0;j<N_RESERVOIR;j++){
                M_inv[i][j] = M_inv[i][j];
            }
        }
    }

    // 3) (W_out) = D * X^T * M_inv
    //    (W_out) は (N_OUTPUT x N_RESERVOIR)
    for(int i=0;i<N_OUTPUT;i++){
        for(int j=0;j<N_RESERVOIR;j++){
            double sum = 0.0;
            // sum over k
            for(int k=0;k<N_RESERVOIR;k++){
                // (D * X^T)[i][k] を先に計算
                double tmp = 0.0;
                for(int t=0;t<train_len;t++){
                    tmp += D[i][t] * X[k][t];
                }
                // それに M_inv[k][j] を掛ける
                sum += tmp * M_inv[k][j];
            }
            esn->W_out[i][j] = sum;
        }
    }
}

int main(void) {
    srand((unsigned int)time(NULL));

    /* ----- ESN 構造体を初期化 ----- */
    ESN esn;
    esn_init(&esn);

    /* ----- 学習用データ & テスト用データを用意 (サンプル: 簡単な配列) ----- */
    // ここでは例として、教師データに x(t) = sin(t) っぽいものを仮定
    // 実際にはユーザが実データを読み込んで用意する
    double train_input[TRAIN_LEN][N_INPUT];
    double train_output[TRAIN_LEN][N_OUTPUT];
    for(int t=0; t<TRAIN_LEN; t++){
        double val = sin(0.1 * t);
        train_input[t][0]  = val;        // 入力
        train_output[t][0] = cos(0.1 * t); // 出力(例)
    }

    double test_input[TEST_LEN][N_INPUT];
    for(int t=0; t<TEST_LEN; t++){
        double val = sin(0.1 * (TRAIN_LEN + t));
        test_input[t][0] = val;
    }

    /* 
     * 学習フェーズ:
     * 各時刻のリザーバ状態 x(t) を貯めておき、後で W_out を学習 
     */

    // リザーバ状態を保存する X
    // X[i][t] : i番目のリザーバニューロン, t番目の時系列サンプル
    double **X = (double**)malloc(sizeof(double*) * N_RESERVOIR);
    for(int i=0;i<N_RESERVOIR;i++){
        X[i] = (double*)malloc(sizeof(double) * TRAIN_LEN);
    }

    // 教師出力を保存する D
    // D[j][t] : j番目の出力ニューロン, t番目の時系列サンプル
    double **D = (double**)malloc(sizeof(double*) * N_OUTPUT);
    for(int i=0;i<N_OUTPUT;i++){
        D[i] = (double*)malloc(sizeof(double) * TRAIN_LEN);
    }

    // ウォーミングアップ的に最初から学習に入れる(簡易化)
    // 実際には最初の数ステップは捨てるなど工夫を入れることが多い
    for(int t=0; t<TRAIN_LEN; t++){
        esn_update_state(&esn, train_input[t]);

        // 状態を保存
        for(int i=0;i<N_RESERVOIR;i++){
            X[i][t] = esn.x[i];
        }
        // 教師信号をコピー
        for(int i=0;i<N_OUTPUT;i++){
            D[i][t] = train_output[t][i];
        }
    }

    // リザーバ状態 X, 教師出力 D から W_out をリッジ回帰で学習
    train_ridge_regression(&esn, X, D, TRAIN_LEN);

    /* メモリ解放(計算後) */
    for(int i=0;i<N_RESERVOIR;i++){
        free(X[i]);
    }
    free(X);
    for(int i=0;i<N_OUTPUT;i++){
        free(D[i]);
    }
    free(D);

    /* ----- テストフェーズ(予測) ----- */
    // 学習時のリザーバ状態をリセット (ここでは単純に0に戻すなど)
    for(int i=0; i<N_RESERVOIR; i++){
        esn.x[i] = 0.0;
    }

    printf("Test predictions:\n");
    for(int t=0; t<TEST_LEN; t++){
        esn_update_state(&esn, test_input[t]);
        double y[N_OUTPUT];
        esn_calculate_output(&esn, y);
        printf("t=%3d, input=%.3f, predict=%.3f\n", t, test_input[t][0], y[0]);
    }

    return 0;
}
