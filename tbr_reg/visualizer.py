import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QGridLayout, QTableWidget, QTableWidgetItem, QLineEdit

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ATE import Domain, UniformSamplingStrategy, SumParameterGroup, ContinuousParameter

import random


def first(lst, default=None):
    return lst[0] if len(lst) == 1 else default


class ParamWidget(QTableWidget):
    def __init__(self, domain, parent=None):
        super(ParamWidget, self).__init__(parent)

        self.domain = domain

        self.setRowCount(sum([
            (param.size if isinstance(param, SumParameterGroup) else 1)
            for param in self.domain.params
        ]))
        self.setColumnCount(3)

        self.setHorizontalHeaderLabels(['Parameter', 'Selection', 'Value'])
        self.setColumnWidth(0, 200)
        self.setColumnWidth(1, 70)
        self.setColumnWidth(2, 200)

        self.param_to_row = {}
        self.row_to_param = [None] * self.rowCount()

        param_idx = 0
        for param in self.domain.params:
            if isinstance(param, SumParameterGroup):
                rows = [(name, human_readable_name)
                        for name, human_readable_name
                        in zip(param.names, param.human_readable_names)]
            else:
                rows = [(param.name, param.human_readable_name)]

            for name, human_readable_name in rows:
                self.setItem(param_idx, 0, QTableWidgetItem(
                    human_readable_name))
                self.setItem(param_idx, 1, QTableWidgetItem('sample'))
                self.setItem(param_idx, 2, QTableWidgetItem(''))

                self.param_to_row[name] = param_idx
                self.row_to_param[param_idx] = name
                param_idx += 1

        self.setItem(0, 1, QTableWidgetItem('x'))
        self.setItem(1, 1, QTableWidgetItem('y'))

    def set_param(self, param_name, param_value):
        if param_name not in self.param_to_row:
            print('WARNING: Skip %s = %s' % (param_name, param_value))
            return

        row_idx = self.param_to_row[param_name]
        self.setItem(row_idx, 2, QTableWidgetItem(str(param_value)))

    def find_param_with_selection(self, selection):
        for row_idx in range(self.rowCount()):
            if self.item(row_idx, 1).text() == selection:
                return self.row_to_param[row_idx]
        return None


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.domain = Domain()

        self.x_param = None
        self.x_param_name = None
        self.x_granularity = None

        self.y_param = None
        self.y_param_name = None
        self.y_granularity = None

        self.tbr_params = None

        self.init_layout()

    def init_layout(self):
        layout = QGridLayout()

        self.model_fig, self.model_canv, self.model_tool = self.init_fig()
        layout.addWidget(self.model_tool, 0, 0)
        layout.addWidget(self.model_canv, 1, 0)

        self.err_fig, self.err_canv, self.err_tool = self.init_fig()
        layout.addWidget(self.err_tool, 2, 0)
        layout.addWidget(self.err_canv, 3, 0, 3, 1)

        self.true_fig, self.true_canv, self.true_tool = self.init_fig()
        layout.addWidget(self.true_tool, 0, 1, 1, 2)
        layout.addWidget(self.true_canv, 1, 1, 1, 2)

        self.param_table = ParamWidget(self.domain)
        layout.addWidget(self.param_table, 2, 1, 2, 2)

        self.x_granularity_box = QLineEdit('5')
        layout.addWidget(self.x_granularity_box, 4, 1)

        self.y_granularity_box = QLineEdit('5')
        layout.addWidget(self.y_granularity_box, 4, 2)

        self.sample_button = QPushButton('Sample values')
        self.sample_button.clicked.connect(self.perform_sample)
        layout.addWidget(self.sample_button, 5, 1)

        self.setLayout(layout)

    def init_fig(self):
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        return figure, canvas, toolbar

    def perform_sample(self):
        sampling_strategy = UniformSamplingStrategy()

        df = self.domain.gen_data_frame(sampling_strategy, 1)
        for column in df.columns:
            self.param_table.set_param(column, df.at[0, column])

        self.x_granularity = int(self.x_granularity_box.text())
        self.x_param_name = self.param_table.find_param_with_selection('x')
        print(f'X is {self.x_param_name}')

        self.y_granularity = int(self.y_granularity_box.text())
        self.y_param_name = self.param_table.find_param_with_selection('y')
        print(f'Y is {self.y_param_name}')

        if self.x_param_name is None or self.y_param_name is None:
            print('ERROR: missing X or Y axis!')
            return

        self.x_param = first([param for param in self.domain.params if param.name ==
                              self.x_param_name and isinstance(param, ContinuousParameter)])
        self.y_param = first([param for param in self.domain.params if param.name ==
                              self.y_param_name and isinstance(param, ContinuousParameter)])

        if self.x_param is None or self.y_param is None:
            print('ERROR: X or Y axis is not continuous!')
            return

        n_points = self.x_granularity * self.y_granularity
        df = pd.concat([df]*n_points)

        x_linspace = np.linspace(
            start=self.x_param.val[0], stop=self.x_param.val[1], num=self.x_granularity, endpoint=True)
        y_linspace = np.linspace(
            start=self.y_param.val[0], stop=self.y_param.val[1], num=self.y_granularity, endpoint=True)

        df[self.x_param_name] = np.c_[
            [x_linspace] * self.y_granularity].ravel('C')
        df[self.y_param_name] = np.c_[
            [y_linspace] * self.x_granularity].ravel('F')
        self.tbr_params = df

        self.plot_model()
        self.plot_true()
        self.plot_err()

    def plot_domain(self, fig, canv, z_data, z_label, symmetrical=True):
        fig.clear()
        fig.set_tight_layout(True)
        ax1 = fig.add_subplot(111)

        cmap = 'RdBu' if symmetrical else 'viridis'

        x_data = self.tbr_params[self.x_param_name].to_numpy().reshape(
            self.y_granularity, self.x_granularity)
        y_data = self.tbr_params[self.y_param_name].to_numpy().reshape(
            self.y_granularity, self.x_granularity)

        pl1 = None
        if z_data is not None:
            z_data = z_data.reshape(self.y_granularity, self.x_granularity)
            pl1 = ax1.contourf(x_data, y_data, z_data,
                               cmap=cmap)

        ax1.scatter(x_data, y_data, marker='o', c='k',
                    s=12, linewidths=0.8, edgecolors='w')
        ax1.set_xlabel(first([param.human_readable_name for param in self.domain.params
                              if param.name == self.x_param_name]))
        ax1.set_ylabel(first([param.human_readable_name for param in self.domain.params
                              if param.name == self.y_param_name]))

        if pl1 is not None:
            cb1 = fig.colorbar(pl1, orientation='vertical',
                               label=z_label, ax=ax1)
            # cb1.ax.locator_params(nbins=5)

        canv.draw()

    def plot_model(self):
        random_data = np.random.rand(
            self.x_granularity * self.y_granularity, 1) * 2
        self.plot_domain(self.model_fig, self.model_canv,
                         random_data, 'Model TBR')

    def plot_true(self):
        random_data = np.random.rand(
            self.x_granularity * self.y_granularity, 1) * 2
        self.plot_domain(self.true_fig, self.true_canv,
                         random_data, 'True TBR')

    def plot_err(self):
        random_data = np.random.rand(
            self.x_granularity * self.y_granularity, 1) * 2
        self.plot_domain(self.err_fig, self.err_canv,
                         random_data, 'Approximation error',
                         symmetrical=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
