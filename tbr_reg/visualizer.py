import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QGridLayout, QTableWidget, QTableWidgetItem, QLineEdit, QProgressBar
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
import joblib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ATE import Domain, UniformSamplingStrategy, SumParameterGroup, ContinuousParameter, Samplerun
from data_utils import encode_data_frame
from keras.models import load_model

import random


def first(lst, default=None):
    return lst[0] if len(lst) == 1 else default


class Model:
    def __init__(self, in_filename):
        self.in_filename = in_filename
        self.scaler = joblib.load('%s.scaler.pkl' % in_filename)
        self.network = load_model('%s.model.h5' % in_filename)
        self.network.summary()

    def evaluate(self, params, domain):
        # preprocess data
        X = encode_data_frame(params, domain)
        X = X.sort_index(axis=1)
        X = self.scaler.transform(X)

        # make predictions
        df = params.copy()
        df.insert(0, 'tbr_pred', -1.)
        df['tbr_pred'] = self.network.predict(X)
        return df


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
                self.setItem(param_idx, 1, QTableWidgetItem('r'))
                self.setItem(param_idx, 2, QTableWidgetItem(''))

                self.param_to_row[name] = param_idx
                self.row_to_param[param_idx] = name
                param_idx += 1

        self.setItem(9, 1, QTableWidgetItem('x'))
        self.setItem(10, 1, QTableWidgetItem('y'))

    def set_param(self, param_name, param_value):
        if param_name not in self.param_to_row:
            print('WARNING: Skip %s = %s' % (param_name, param_value))
            return

        row_idx = self.param_to_row[param_name]
        self.setItem(row_idx, 2, QTableWidgetItem(str(param_value)))

    def get_params(self):
        def graceful_float_conversion(value):
            try:
                return float(value)
            except ValueError:
                return value

        return {
            param_name: graceful_float_conversion(self.item(row_idx, 2).text())
            for row_idx, param_name in enumerate(self.row_to_param)
        }

    def find_param_with_selection(self, selection):
        for row_idx in range(self.rowCount()):
            if self.item(row_idx, 1).text() == selection:
                return self.row_to_param[row_idx]
        return None


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.domain = Domain()
        self.surrogate_model = Model('outputs/model')

        self.x_param = None
        self.x_param_name = None
        self.x_granularity = None

        self.y_param = None
        self.y_param_name = None
        self.y_granularity = None

        self.tbr_params = None
        self.tbr_true = None
        self.tbr_err = None

        self.tbr_worker = None
        self.tbr_worker_thread = None

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
        layout.addWidget(self.true_tool, 0, 1, 1, 3)
        layout.addWidget(self.true_canv, 1, 1, 1, 3)

        self.param_table = ParamWidget(self.domain)
        layout.addWidget(self.param_table, 2, 1, 2, 3)

        self.x_granularity_box = QLineEdit('5')
        layout.addWidget(self.x_granularity_box, 4, 1)

        self.y_granularity_box = QLineEdit('5')
        layout.addWidget(self.y_granularity_box, 4, 2)

        self.generate_button = QPushButton('Generate lattice')
        self.generate_button.clicked.connect(self.generate_lattice)
        layout.addWidget(self.generate_button, 4, 3)

        self.randomize_button = QPushButton('Randomize params')
        self.randomize_button.clicked.connect(self.randomize_params)
        layout.addWidget(self.randomize_button, 5, 1)

        self.query_tbr_button = QPushButton('Query true TBR')
        self.query_tbr_button.clicked.connect(self.query_tbr)
        layout.addWidget(self.query_tbr_button, 6, 1)

        self.query_tbr_progress = QProgressBar()
        layout.addWidget(self.query_tbr_progress, 6, 2, 1, 2)

        self.setLayout(layout)

    def init_fig(self):
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        return figure, canvas, toolbar

    def randomize_params(self):
        sampling_strategy = UniformSamplingStrategy()
        df = self.domain.gen_data_frame(sampling_strategy, 1)
        for column in df.columns:
            self.param_table.set_param(column, df.at[0, column])

        self.generate_lattice()

    def generate_lattice(self):
        df = pd.DataFrame(
            data={key: [value] for key, value in self.param_table.get_params().items()})

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
        self.tbr_params = df.reset_index(drop=True)

        self.tbr_true = None
        self.tbr_err = None

        self.tbr_surrogate = None
        self.evaluate_model()

        self.plot_model()
        self.plot_true()
        self.plot_err()

    def evaluate_model(self):
        if self.tbr_params is None:
            self.tbr_surrogate = None
            return

        self.tbr_surrogate = self.surrogate_model.evaluate(
            self.tbr_params, self.domain)

    def query_tbr(self):
        if self.tbr_params is None:
            print('ERROR: TBR queried with no sampled parameters')
            return

        self.query_tbr_button.setEnabled(False)
        self.query_tbr_progress.setValue(0)
        self.query_tbr_progress.setMinimum(0)
        self.query_tbr_progress.setMaximum(self.tbr_params.shape[0])

        class Worker(QObject):
            finished = pyqtSignal()
            progress = pyqtSignal(int, int)
            samples_available = pyqtSignal(pd.DataFrame)

            def __init__(self, tbr_params, x_param_name, y_param_name, parent=None):
                super(Worker, self).__init__(parent)
                self.tbr_params = tbr_params
                self.x_param_name = x_param_name
                self.y_param_name = y_param_name

            def progress_handler(self, i, n_samples):
                self.progress.emit(i, n_samples)

            @pyqtSlot()
            def query_tbr(self):
                run = Samplerun(no_docker=True)
                sampled = run.perform_sample(out_file=None,
                                             param_values=self.tbr_params,
                                             progress_handler=self.progress_handler)

                param_names = [self.x_param_name, self.y_param_name]
                sampled = pd.merge(self.tbr_params, sampled,  how='left',
                                   left_on=param_names, right_on=param_names, suffixes=('', '_dup_'))
                sampled.drop([column for column in sampled.columns
                              if column.endswith('_dup_')], axis=1, inplace=True)
                self.samples_available.emit(sampled)

                self.finished.emit()

        self.tbr_worker = Worker(
            self.tbr_params, self.x_param_name, self.y_param_name)
        self.tbr_worker_thread = QThread()

        self.tbr_worker.moveToThread(self.tbr_worker_thread)
        self.tbr_worker.finished.connect(self.tbr_worker_thread.quit)
        self.tbr_worker.samples_available.connect(
            self.tbr_worker_samples_available)
        self.tbr_worker.progress.connect(self.tbr_worker_progress)
        self.tbr_worker_thread.started.connect(self.tbr_worker.query_tbr)
        self.tbr_worker_thread.finished.connect(self.tbr_worker_finished)

        self.tbr_worker_thread.start()

    @pyqtSlot(int, int)
    def tbr_worker_progress(self, i, n_samples):
        self.query_tbr_progress.setValue(i + 1)

    @pyqtSlot(pd.DataFrame)
    def tbr_worker_samples_available(self, sampled):
        self.tbr_true = sampled
        self.plot_true()

        err = sampled.copy()
        err.insert(0, 'tbr_surrogate_err', 0.01)

        param_names = [self.x_param_name, self.y_param_name]
        err = pd.merge(err, self.tbr_surrogate, how='left',
                       left_on=param_names, right_on=param_names, suffixes=('', '_dup_'))
        err.drop([column for column in err.columns
                  if column.endswith('_dup_')], axis=1, inplace=True)
        err['tbr_surrogate_err'] = err.apply(
            lambda row: np.abs(row.tbr - row.tbr_pred), axis=1)

        self.tbr_err = err
        self.plot_err()

    @pyqtSlot()
    def tbr_worker_finished(self):
        del self.tbr_worker
        del self.tbr_worker_thread
        self.query_tbr_button.setEnabled(True)

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
        vmin, vmax = None, None
        if z_data is not None:
            vmin, vmax = np.min(z_data), np.max(z_data)

            if symmetrical:
                if np.abs(vmin - 1) > np.abs(vmax - 1):
                    vmax = 1 + np.abs(vmin - 1)
                else:
                    vmin = 1 - np.abs(vmax - 1)

            z_data = z_data.reshape(self.y_granularity, self.x_granularity)
            pl1 = ax1.contourf(x_data, y_data, z_data,
                               cmap=cmap, vmin=vmin, vmax=vmax)

        ax1.scatter(x_data, y_data, marker='o', c='k',
                    s=12, linewidths=0.8, edgecolors='w')
        ax1.set_xlabel(first([param.human_readable_name for param in self.domain.params
                              if param.name == self.x_param_name]))
        ax1.set_ylabel(first([param.human_readable_name for param in self.domain.params
                              if param.name == self.y_param_name]))

        if pl1 is not None:
            n_steps = 12
            cb1 = fig.colorbar(pl1, orientation='vertical',
                               label=z_label, ax=ax1, boundaries=np.linspace(vmin, vmax, n_steps))
            cb1.ax.locator_params(nbins=n_steps)

        canv.draw()

    def plot_model(self):
        surrogate_data = self.tbr_surrogate['tbr_pred'].to_numpy() \
            if self.tbr_surrogate is not None else None
        self.plot_domain(self.model_fig, self.model_canv,
                         surrogate_data, 'Surrogate TBR')

    def plot_true(self):
        true_data = self.tbr_true['tbr'].to_numpy() \
            if self.tbr_true is not None else None
        self.plot_domain(self.true_fig, self.true_canv,
                         true_data, 'True TBR')

    def plot_err(self):
        err_data = self.tbr_err['tbr_surrogate_err'].to_numpy() \
            if self.tbr_err is not None else None
        self.plot_domain(self.err_fig, self.err_canv,
                         err_data, 'Approximation error',
                         symmetrical=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
