import numpy as np

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
