
import numpy as np

"""
パラレレパイペッド法の実装
ある土地被覆クラスにおいて, すべての教師データの入力事例(観測値)から得られる各バンドのスペクトル値の範囲(min-max)
を使うことでn次元空間上に超直方体(hyperrectangle, orthotope)が定義される.
パラレレパイペッド法とは入力事例が属する超直方体があるとき, その超直方体を定める土地被覆クラスを最終的な分類結果
とする教師付き分類手法のことである.
"""
# n次元直方体(n-dimensional rectangular box, hyperrectangle)の実装
class HyperRectangle:
    def __init__(self, intervals, category = 0):
        """
        intervals: List of tuples/lists [(a1,b1),(a2,b2),...,(an,bn)]
        """
        self.category  = category
        self.intervals = []
        for a, b in intervals:
            if a > b:
                raise ValueError(f'Invalid interval:[{a},{b}]')
            self.intervals.append((a,b))
        self.dimension = len(self.intervals)

    def contains(self,point):
        """点が超直方体に含まれているかを判定する"""
        if len(point) != self.dimension:
            raise ValueError('Point dimensionality mismatch!')
        return all(a <= x <= b for x, (a,b) in zip(point,self.intervals))

    def volume(self):
        """n次元直方体の体積(長さの積)を返す"""
        from functools import reduce
        from decimal import Decimal
        # 高階関数の一つであるreduceは,無名関数lambdaを用いることでコードが簡潔になる. ここで, accの初期値は1である.
        # 積の結果が非常に大きくなることでoverflowになることを防ぐため, decimal.Decimalを利用する.
        return reduce(lambda acc, ab: acc*(Decimal(ab[1])-Decimal(ab[0])), self.intervals, Decimal(1))

    def centroid(self):
        """n次元直方体の重心(centroid)の座標を返す"""
        return np.array(list(map(lambda interval: (interval[0] + interval[1])/2 , self.intervals)))

    def __repr__(self):
        """オブジェクトの「公式な文字列表現」を返すメソッド.
        すべてのclassはobjectを暗黙的に継承するため, クラスの定義で
        __repr__を再定義するのは, __repr__のオーバーライドに相当する.
        reprはrepresentationの頭文字をとったもの.
        """
        return f'HyperRectangle({self.intervals})'

class Parallelepiped:
    # Parallelepiped(パラレレパイペッド, パラレレ(パラレロ)ピペッド): 平行六面体
    def __init__(self, category_num = 17):
        self.category_num = category_num # defaultの値はindianpinesには有効な定数
        self.hr_list = []
        self.QDs = [] # 各バンドの四分位偏差(quartile deviation)をまとめたベクトルを各クラスに対して求める.
    def fit(self, X_train, y_train):
        # 各クラスに対して, 特徴空間における最小値~最大値の区間をバンドごとに学習する
        column_num = X_train.shape[1]
        for category in range(1, self.category_num):
            intervals = []
            QD = []
            for column_position in range(column_num):
                column_vec = X_train[np.where(y_train == category)][:,column_position]  # あるクラス(category)の教師データをまとめた行列のcolumn_position番目の列ベクトル.教師データのサンプル数をnとすると, 次元は(n,)である.
                min_val, max_val = column_vec.min(), column_vec.max()                   # あるクラスの教師データのcolumn_positionバンドの上下限を取得
                intervals.append((min_val,max_val))
                IQR_value = find_the_IQR(dist=column_vec)
                QD.append(IQR_value/2)
            hr = HyperRectangle(intervals = intervals, category = category)
            self.hr_list.append(hr)
            self.QDs.append(np.array(QD))
        self.QDs = np.array(self.QDs)

    def predict(self,
                X_test,
                remove_order_dependency = True,
                classification_strategy = 'QD'
                ):
        """
        Args:
            X_test(ndarray): 予測したい特徴量をまとめた配列.
            remove_order_dependency(bool): 順序依存性への対処を行うかを決める.
            classification_strategy(str): 順序依存性を解消した分類戦略を指定する.
        Returns:
            y_pred(ndarray): 特徴量に対する予測ラベルをまとめた配列.
        """
        y_pred = []
        if remove_order_dependency:
            """最終的な分類結果の決定に超直方体の重心を利用する"""
            for x in X_test:
                # xが属する超直方体をまとめたリストを初期化
                hr_index = [index for index, hr in enumerate(self.hr_list) if hr.contains(x)]
                hr_candidates = [self.hr_list[i] for i in hr_index]
                hr_QDcandidates = np.array([self.QDs[i] for i in hr_index])
                if len(hr_candidates) == 0:
                    label = 0
                else:
                    if classification_strategy == 'centroid':
                        """
                        xが複数の超直方体に属する場合, xと超直方体の重心との距離が最小となるような超直方体を
                        定める土地被覆クラスを最終的な分類結果とする
                        """
                        # np配列としてまとめて扱う
                        hr_centroids = np.array([hr.centroid() for hr in hr_candidates])
                        dists = np.linalg.norm(hr_centroids - x, axis = 1) # broadcastingで高速に計算
                    elif classification_strategy == 'QD':
                        """
                        xが複数の超直方体に属する場合,xと超直方体の各次元の四分位偏差(QD,quartile deviation)を
                        まとめたベクトルとの距離が最小となるような超直方体を定める土地被覆クラスを最終的な分類結果とする
                        """
                        dists = np.linalg.norm(hr_QDcandidates - x, axis = 1) # broadcastingで高速に計算

                    ans_index = np.argmin(dists)
                    if len(hr_index) == 1:
                        label = hr_candidates[0].category
                    else:
                        label = hr_candidates[ans_index].category
                y_pred.append(label)
            return np.array(y_pred)
        else:
            """
            最終的な分類結果の決定に超直方体の重心を使わない.
            初めて入力事例xが属する超直方体を定める土地被覆クラスを最終的な分類結果とする.
            """
            for x in X_test:
                label = 0
                for hr in self.hr_list:
                    # xが複数のhrに含まれる場合, hr_listにあるhrの先着順で予測ラベルが決定される
                    if hr.contains(x):
                        label = hr.category
                        break
                y_pred.append(label)
            return np.array(y_pred)

def find_the_IQR(dist, interpolation_method = 'linear'):
    """IQR(四分位範囲, interquartile)を返す
    Args:
        dist (nd.array): IQRを求める対象のデータ
        interpolation_method (str): パーセンタイル計算の補間方法. Defaults to 'linear'.
                                    ('linear', 'lower', 'higher', 'midpoint', 'nearest')
    Returns:
        float: distのIQRの値
    """
    from scipy.stats import iqr
    if len(dist) == 0:
        print('distが空のため、IQRが求められません')
        return np.nan
    # 欠損値がある場合は取り除く(より堅牢な関数にするため)
    # なおindianpinesデータセットでは欠損値を含まない教師データが扱えるが、
    # 一般のリモートセンシング画像を想定して欠損値の対処をしておく.
    dist = dist[~np.isnan(dist)]
    IQR_scipy = iqr(dist,interpolation=interpolation_method) # distの四分位範囲
    return IQR_scipy

if __name__ == '__main__':
    import s3vm_pines as s3modu
    from indianpines import load as load_pines
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    pines_parameters = {
         "pca" : 0,
         "include_background" : True,
         "recategorize_rule" : None
    }
    split_parameters = {
        "p_train" : 0.6,
        "seed_train" : 43,
        "n_train_equal" : False,
        "recategorize_rule" : None
    }
    pines = load_pines(**pines_parameters)

    X = [(1,2),(3,4),(5,6)]
    X_rectangle = HyperRectangle(intervals = X)
    print(f'X_rectangle centroid: ',X_rectangle.centroid())
    train_test_status, _ = s3modu.train_test_split(**split_parameters)

    X_train, X_test, y_train, y_test = train_test_split(
        pines.features,
        pines.target,
        test_size = 0.4,
        random_state=43
    )
    model = Parallelepiped()

    
    model.fit(X_train = X_train, y_train = y_train)
    y_pred = model.predict(X_test = X_test)
    print(accuracy_score(y_true=y_test, y_pred=y_pred))
