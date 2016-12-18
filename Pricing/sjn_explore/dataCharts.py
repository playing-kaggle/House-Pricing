__author__ = 'cat'
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np

class ChartPlayer:

    def show_charts(self, df, col_name):
        values = df[col_name]

        fig, ax = plt.subplots()

        if values.dtype != 'O':
    #         pd.DataFrame(values).hist()
            plt.hist(values.values, range=(values.min(),values.max()))
        else:
            counts = Counter(values)
            df = pd.DataFrame.from_dict(counts, orient='index').sort_index()

            labels = df.index.values

            N = len(labels)
            width = 0.35       # the width of the bars: can also be len(x) sequence
            ind = np.arange(N)

            def autolabel(rects,label_num):
                # attach some text labels
                for rect in rects:
                    height = int(rect.get_height())
                    count_label = str(height) if height>5 or label_num<20 else ''
                    ax.text(rect.get_x() + rect.get_width()/2., height,
                        count_label, ha='center', va='bottom')
            if N<10:
                plt.xticks(ind + width/2., tuple(labels))

            rect = plt.bar(ind, tuple(df.values[:,0]), width)
            autolabel(rect,len(labels))

        plt.ylabel('Count')
        plt.title('Count of '+col_name)
        plt.show()
