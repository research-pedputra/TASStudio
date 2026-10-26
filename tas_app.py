"""Run with: python tas_app.py [project.tas]."""
import copy
import json
import sys
from pathlib import Path
import os
import uuid
import hashlib
import numpy as np
os.environ.setdefault('QT_API', 'pyside6')
from PySide6 import QtCore, QtGui, QtWidgets as W
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.artist import ArtistInspector
from matplotlib.text import Text
from matplotlib.colors import is_color_like
from tas_core import Project, KINDS, literal, render, artists, capture_views, packed, apply_artist_edit, fit_figure_labels


class ColorField(W.QWidget):
    def __init__(self, value, multiple=False):
        super().__init__()
        self.multiple = multiple
        layout = W.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = W.QLineEdit(repr(value) if multiple else str(value or ''))
        self.edit.setPlaceholderText('Use plot default' if not multiple else "['#2563eb', '#f97316']")
        layout.addWidget(self.edit)
        pick = W.QPushButton('Choose…')
        pick.setFixedWidth(84)
        pick.clicked.connect(self.choose)
        layout.addWidget(pick)

    def choose(self):
        color = W.QColorDialog.getColor(parent=self)
        if color.isValid():
            if self.multiple:
                values = literal(self.edit.text()) or []
                if isinstance(values, str):
                    values = [values]
                self.edit.setText(repr(list(values) + [color.name()]))
            else:
                self.edit.setText(color.name())

    def text(self):
        return self.edit.text()

    def setText(self, value):
        self.edit.setText(repr(value) if self.multiple else str(value or ''))


class Fields(W.QWidget):
    LABELS = {'ncols': 'Matrix columns', 'nrows': 'Matrix rows', 'figsize': 'Figure size · inches',
              'cmap': 'Color map', 'delays': 'Snapshot times · ps', 'snapshot_colors': 'Snapshot line colors',
              'number_of_components': 'Lifetime components', 'combined': 'Combine / normalize traces',
              'log_x': 'Logarithmic time axis', 'share_colorbar': 'Shared colorbar',
              'color_vmin': 'Color scale minimum', 'color_vmax': 'Color scale maximum',
              'gsb_wavelength': 'GSB wavelength · nm (None = auto)'}

    def __init__(self, values):
        super().__init__()
        self.inputs = {}
        form = W.QFormLayout(self)
        # Every caption has its own row: long names cannot collide with inputs.
        form.setRowWrapPolicy(W.QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(W.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(9)
        for key, value in values.items():
            if key in ('horizontal_spacing', 'vertical_spacing'):
                widget = W.QDoubleSpinBox()
                widget.setRange(0, 3)
                widget.setSingleStep(.05)
                widget.setValue(value)
            elif key in ('ncols', 'nrows'):
                widget = W.QSpinBox()
                widget.setRange(0 if key == 'nrows' else 1, 100)
                if key == 'nrows':
                    widget.setSpecialValueText('Auto')
                widget.setValue(value)
            elif key == 'font_family':
                widget = W.QFontComboBox()
                widget.setSizeAdjustPolicy(W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                widget.setMinimumContentsLength(14)
                widget.setCurrentFont(QtGui.QFont(value))
            elif key == 'cmap':
                widget = W.QComboBox()
                widget.setSizeAdjustPolicy(W.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                widget.setMinimumContentsLength(14)
                widget.addItems([''] + sorted(matplotlib.colormaps))
                widget.setCurrentText(value or '')
                widget.setToolTip('Choose a Matplotlib color map. Blank uses the plot default.')
            elif key in ('lifetime_data_color', 'lifetime_fit_color', 'snapshot_colors'):
                widget = ColorField(value, multiple=key == 'snapshot_colors')
            elif isinstance(value, bool):
                widget = W.QCheckBox('Enabled')
                widget.setChecked(value)
            else:
                widget = W.QLineEdit(repr(value))
            if key not in ('cmap',):
                widget.setToolTip('Use numbers, (minimum, maximum), [values], or None for automatic.')
            if key in ('horizontal_spacing', 'vertical_spacing'):
                widget.setToolTip('Larger values add space between panels. Start at 0.50.')
            self.inputs[key] = widget
            label = W.QLabel(self.LABELS.get(key, key.replace('_', ' ').capitalize()))
            label.setWordWrap(True)
            label.setMinimumWidth(235)
            form.addRow(label, widget)

    def values(self):
        result = {}
        for k,w in self.inputs.items():
            if isinstance(w, W.QCheckBox):
                result[k] = w.isChecked()
            elif isinstance(w, (W.QDoubleSpinBox, W.QSpinBox)):
                result[k] = w.value()
            elif isinstance(w, W.QFontComboBox):
                result[k] = w.currentFont().family()
            elif isinstance(w, W.QComboBox):
                result[k] = w.currentText() or None
            else:
                result[k] = literal(w.text())
        return result


def scroll(widget):
    area = W.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class ResponsiveCanvas(FigureCanvasQTAgg):
    """Scale the preview resolution, keeping the requested physical figure size."""
    def __init__(self, figure):
        self.figure_inches = tuple(figure.get_size_inches())
        super().__init__(figure)
        self.setMinimumSize(1, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = event.size().width()
        if width > 0:
            dpi = width * self.device_pixel_ratio / self.figure_inches[0]
            self.figure.set_dpi(dpi)
            self.figure.set_size_inches(self.figure_inches, forward=False)
            self.draw_idle()


class FigureViewport(W.QWidget):
    """Fit the figure inside the available area without stretching its aspect."""
    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        canvas.setParent(self)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(W.QSizePolicy.Policy.Expanding, W.QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_canvas()

    def fit_canvas(self):
        fw, fh = self.canvas.figure_inches
        scale = min(self.width() / fw, self.height() / fh)
        w, h = max(1, round(fw * scale)), max(1, round(fh * scale))
        self.canvas.setGeometry((self.width()-w)//2, (self.height()-h)//2, w, h)


class TextEditor(W.QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit plot text')
        self.resize(430, 360)
        layout = W.QFormLayout(self)
        self.content = W.QPlainTextEdit(text.get_text())
        self.content.setToolTip('Multiline text and Matplotlib math, e.g. $\\tau$, are supported.')
        layout.addRow('Text', self.content)
        self.font = W.QFontComboBox()
        self.font.setCurrentFont(QtGui.QFont(text.get_fontfamily()[0]))
        layout.addRow('Font', self.font)
        self.size = W.QDoubleSpinBox()
        self.size.setRange(1, 500)
        self.size.setValue(text.get_fontsize())
        layout.addRow('Size (points)', self.size)
        self.weight = W.QComboBox()
        self.weight.setEditable(True)
        self.weight.addItems(['normal', 'bold', 'light', 'semibold', 'heavy'])
        self.weight.setCurrentText(str(text.get_fontweight()))
        layout.addRow('Weight', self.weight)
        self.italic = W.QCheckBox()
        self.italic.setChecked(text.get_fontstyle() != 'normal')
        layout.addRow('Italic', self.italic)
        self.color = W.QLineEdit(str(text.get_color()))
        colors = W.QWidget()
        row = W.QHBoxLayout(colors)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.color)
        pick = W.QPushButton('Choose…')
        pick.clicked.connect(self.choose_color)
        row.addWidget(pick)
        layout.addRow('Color', colors)
        self.rotation = W.QDoubleSpinBox()
        self.rotation.setRange(-360, 360)
        self.rotation.setValue(text.get_rotation())
        layout.addRow('Rotation', self.rotation)
        self.error = W.QLabel()
        layout.addRow(self.error)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def choose_color(self):
        color = W.QColorDialog.getColor(parent=self)
        if color.isValid():
            self.color.setText(color.name())

    def values(self):
        return dict(text=self.content.toPlainText(), fontfamily=self.font.currentFont().family(),
                    fontsize=self.size.value(), fontweight=literal(self.weight.currentText()),
                    fontstyle='italic' if self.italic.isChecked() else 'normal',
                    color=literal(self.color.text()), rotation=self.rotation.value())

    def validate(self):
        try:
            probe = Text()
            probe.set(**self.values())
            if not is_color_like(self.values()['color']):
                raise ValueError('Choose a valid Matplotlib color.')
        except Exception as error:
            self.error.setText(str(error))
            return
        self.accept()


class MatrixDataModel(QtCore.QAbstractTableModel):
    """Fast editable view over a TAS matrix; it does not create one widget per cell."""
    def __init__(self, matrix, parent=None):
        super().__init__(parent)
        self.matrix = np.asarray(matrix, dtype=float)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.matrix.shape[0]

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else self.matrix.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.matrix[index.row(), index.column()]
        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            return '' if not np.isfinite(value) else f'{value:.12g}'
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and (index.row() == 0 or index.column() == 0):
            return QtGui.QColor('#e5f0ef')
        return None

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if role != QtCore.Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        try:
            self.matrix[index.row(), index.column()] = np.nan if not str(value).strip() else float(value)
        except ValueError:
            return False
        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole])
        return True

    def flags(self, index):
        return super().flags(index) | QtCore.Qt.ItemFlag.ItemIsEditable

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        return ('Delay / signal column ' if orientation == QtCore.Qt.Orientation.Horizontal else 'Wavelength / data row ') + str(section)


class EditableDataDialog(W.QDialog):
    def __init__(self, project, source, parent=None):
        super().__init__(parent)
        self.project, self.source = project, source
        self.setWindowTitle('Edit embedded TAS data · ' + source['name'])
        self.resize(1100, 720)
        layout = W.QVBoxLayout(self)
        note = W.QLabel('The top row is delay time and the first column is wavelength. Edit any value, paste a selection from a spreadsheet, then Save changes. The updated matrix is embedded in the .tas project.')
        note.setWordWrap(True)
        layout.addWidget(note)
        max_rows = source.get('processing', {}).get('max_rows', 513)
        self.model = MatrixDataModel(project.source_matrix(source, max_rows), self)
        self.table = W.QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.horizontalHeader().setDefaultSectionSize(105)
        self.table.verticalHeader().setDefaultSectionSize(23)
        self.table.setWordWrap(False)
        layout.addWidget(self.table, 1)
        tools = W.QHBoxLayout()
        paste = W.QPushButton('Paste selection')
        paste.clicked.connect(self.paste_safely)
        tools.addWidget(paste)
        tools.addStretch()
        layout.addLayout(tools)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def paste(self):
        index = self.table.currentIndex()
        if not index.isValid():
            raise ValueError('Select the top-left target cell before pasting.')
        rows = [line.split('\t') for line in W.QApplication.clipboard().text().strip().splitlines()]
        if not rows:
            return
        end_row, end_column = index.row() + len(rows), index.column() + max(len(row) for row in rows)
        if end_row > self.model.rowCount() or end_column > self.model.columnCount():
            raise ValueError('Pasted values extend beyond the existing data matrix.')
        for row_offset, row in enumerate(rows):
            for column_offset, value in enumerate(row):
                cell = self.model.index(index.row() + row_offset, index.column() + column_offset)
                if not self.model.setData(cell, value):
                    raise ValueError(f'Invalid numeric value: {value}')

    def paste_safely(self):
        try:
            self.paste()
        except Exception as error:
            W.QMessageBox.warning(self, 'Cannot paste data', str(error))

    def save(self):
        try:
            matrix = self.model.matrix
            if not np.isfinite(matrix[0, 1:]).any() or not np.isfinite(matrix[1:, 0]).any():
                raise ValueError('Keep at least one delay time and wavelength.')
            self.project.update_source_matrix(self.source, matrix)
            self.accept()
        except Exception as error:
            W.QMessageBox.warning(self, 'Cannot save data', str(error))


def spreadsheet_column_name(number):
    """Return Excel-style names: A, B, ..., Z, AA, AB, ... ."""
    result = ''
    number += 1
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


class ManualDataModel(QtCore.QAbstractTableModel):
    """A growing table model that stays quick for large clipboard pastes."""
    def __init__(self, manual, parent=None):
        super().__init__(parent)
        columns = manual.get('columns', []) if isinstance(manual, dict) else []
        rows = manual.get('rows', []) if isinstance(manual, dict) else []
        self.columns = [str(value) for value in columns] or ['X', 'Y1']
        self.rows = [[str(value) if value is not None else '' for value in row]
                     for row in rows if isinstance(row, list)]
        self.used_columns = max(2, len(self.columns), *( [len(row) for row in self.rows] or [0]))
        self.used_rows = len(self.rows)
        self.columns.extend(spreadsheet_column_name(i) for i in range(len(self.columns), self.used_columns))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else max(40, self.used_rows + 20)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else max(8, self.used_columns + 3)

    def _ensure(self, rows, columns):
        while len(self.columns) < columns:
            self.columns.append(spreadsheet_column_name(len(self.columns)))
        while len(self.rows) < rows:
            self.rows.append([''] * self.used_columns)
        if columns > self.used_columns:
            for row in self.rows:
                row.extend([''] * (columns - self.used_columns))
            self.used_columns = columns
        self.used_rows = max(self.used_rows, rows)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            if index.row() < self.used_rows and index.column() < self.used_columns:
                return self.rows[index.row()][index.column()]
            return ''
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return None

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if role != QtCore.Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        self._ensure(index.row() + 1, index.column() + 1)
        self.rows[index.row()][index.column()] = str(value).strip()
        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
        return True

    def flags(self, index):
        return super().flags(index) | QtCore.Qt.ItemFlag.ItemIsEditable | QtCore.Qt.ItemFlag.ItemIsSelectable

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Vertical:
            return str(section + 1)
        name = self.columns[section] if section < len(self.columns) else spreadsheet_column_name(section)
        return f'{spreadsheet_column_name(section)} · {name}'

    def paste(self, start, text):
        values = [line.rstrip('\r').split('\t') for line in text.splitlines()]
        if not values:
            return
        width = max(len(row) for row in values)
        self.beginResetModel()
        self._ensure(start.row() + len(values), start.column() + width)
        for row_offset, row in enumerate(values):
            for column_offset, value in enumerate(row):
                self.rows[start.row() + row_offset][start.column() + column_offset] = value.strip()
        self.endResetModel()

    def add_rows(self, count=100):
        self.beginResetModel()
        self._ensure(self.used_rows + count, self.used_columns)
        self.endResetModel()

    def add_column(self):
        self.beginResetModel()
        self._ensure(self.used_rows, self.used_columns + 1)
        self.endResetModel()

    def clear_indexes(self, indexes):
        changed = False
        for index in indexes:
            if index.row() < self.used_rows and index.column() < self.used_columns:
                self.rows[index.row()][index.column()] = ''
                changed = True
        if changed:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1),
                                  [QtCore.Qt.ItemDataRole.DisplayRole])

    def set_column_names(self, names):
        self.beginResetModel()
        self._ensure(self.used_rows, max(2, len(names)))
        for index, name in enumerate(names):
            self.columns[index] = name.strip() or spreadsheet_column_name(index)
        self.endResetModel()

    def serializable(self):
        last_row = 0
        for index, row in enumerate(self.rows[:self.used_rows], 1):
            if any(value.strip() for value in row[:self.used_columns]):
                last_row = index
        return dict(columns=self.columns[:self.used_columns],
                    rows=[row[:self.used_columns] for row in self.rows[:last_row]])

    def numeric_data(self):
        values = np.full((self.used_rows, self.used_columns), np.nan)
        for row_index, row in enumerate(self.rows[:self.used_rows]):
            for column_index, value in enumerate(row[:self.used_columns]):
                if value.strip():
                    try:
                        values[row_index, column_index] = float(value)
                    except ValueError:
                        pass
        return values


class ManualPlotDialog(W.QDialog):
    """Spreadsheet entry and a live, exportable Matplotlib manual plot."""
    def __init__(self, studio):
        super().__init__(studio)
        self.studio = studio
        self.settings = copy.deepcopy(studio.project.manual.get('settings', {}))
        self.setWindowTitle('Manual data table and plot')
        self.resize(1440, 820)
        layout = W.QVBoxLayout(self)
        note = W.QLabel('Paste tab-separated rows from Excel or any spreadsheet. Choose an X column and one or more Y columns, then update the preview. Save changes keeps the table and its plotting choices inside this .tas project.')
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = W.QSplitter(QtCore.Qt.Orientation.Horizontal)
        sheet = W.QWidget()
        sheet_layout = W.QVBoxLayout(sheet)
        sheet_layout.setContentsMargins(0, 0, 0, 0)
        sheet_layout.addWidget(W.QLabel('Manual data sheet'))
        self.model = ManualDataModel(studio.project.manual, self)
        self.table = W.QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.horizontalHeader().setDefaultSectionSize(120)
        self.table.verticalHeader().setDefaultSectionSize(23)
        self.table.setWordWrap(False)
        sheet_layout.addWidget(self.table, 1)
        data_tools = W.QHBoxLayout()
        for title, callback in [('Paste', self.paste), ('Copy', self.copy_selection), ('Clear selected', self.clear_selection),
                                ('Add 100 rows', self.add_rows), ('Add column', self.add_column), ('Rename columns', self.rename_columns)]:
            button = W.QPushButton(title)
            button.clicked.connect(callback)
            data_tools.addWidget(button)
        data_tools.addStretch()
        sheet_layout.addLayout(data_tools)
        splitter.addWidget(sheet)

        preview = W.QWidget()
        preview_layout = W.QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        controls = W.QWidget()
        form = W.QFormLayout(controls)
        form.setFieldGrowthPolicy(W.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.x_column = W.QComboBox()
        self.y_columns = W.QListWidget()
        self.y_columns.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.kind = W.QComboBox()
        self.kind.addItems(['Line', 'Markers', 'Line + markers', 'Scatter'])
        self.title = W.QLineEdit()
        self.xlabel = W.QLineEdit()
        self.ylabel = W.QLineEdit()
        self.colors = ColorField([], multiple=True)
        self.grid = W.QCheckBox('Show grid')
        self.legend = W.QCheckBox('Show legend')
        form.addRow('X column', self.x_column)
        form.addRow('Y columns', self.y_columns)
        form.addRow('Plot type', self.kind)
        form.addRow('Title', self.title)
        form.addRow('X label', self.xlabel)
        form.addRow('Y label', self.ylabel)
        form.addRow('Line colors', self.colors)
        form.addRow(self.grid, self.legend)
        update = W.QPushButton('Update plot preview')
        update.setObjectName('primaryButton')
        update.clicked.connect(self.render_safely)
        form.addRow(update)
        preview_layout.addWidget(controls)
        self.figure = Figure(figsize=(7.0, 5.2))
        self.canvas = ResponsiveCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        preview_layout.addWidget(self.toolbar)
        self.viewport = FigureViewport(self.canvas)
        preview_layout.addWidget(self.viewport, 1)
        splitter.addWidget(preview)
        splitter.setSizes([720, 720])
        layout.addWidget(splitter, 1)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(W.QDialogButtonBox.StandardButton.Save).setText('Save changes')
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for key, callback in [(QtGui.QKeySequence.StandardKey.Paste, self.paste),
                              (QtGui.QKeySequence.StandardKey.Copy, self.copy_selection)]:
            action = QtGui.QAction(self)
            action.setShortcut(key)
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.table.addAction(action)
        self.refresh_column_selectors()
        self.restore_settings()
        self.render_empty_preview()

    def refresh_column_selectors(self):
        x_value = self.x_column.currentData()
        selected = [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in self.y_columns.selectedItems()]
        self.x_column.blockSignals(True)
        self.x_column.clear()
        self.y_columns.clear()
        for index in range(self.model.used_columns):
            label = f'{spreadsheet_column_name(index)} · {self.model.columns[index]}'
            self.x_column.addItem(label, index)
            item = W.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, index)
            self.y_columns.addItem(item)
        self.x_column.setCurrentIndex(0 if x_value is None else max(0, min(int(x_value), self.model.used_columns - 1)))
        for index in selected:
            if 0 <= index < self.y_columns.count():
                self.y_columns.item(index).setSelected(True)
        self.x_column.blockSignals(False)

    def restore_settings(self):
        self.kind.setCurrentText(self.settings.get('kind', 'Line'))
        self.title.setText(str(self.settings.get('title', '')))
        self.xlabel.setText(str(self.settings.get('xlabel', '')))
        self.ylabel.setText(str(self.settings.get('ylabel', '')))
        self.colors.setText(self.settings.get('colors', []))
        self.grid.setChecked(bool(self.settings.get('grid', True)))
        self.legend.setChecked(bool(self.settings.get('legend', True)))
        x_index = self.settings.get('x_column', 0)
        self.x_column.setCurrentIndex(max(0, min(int(x_index), self.x_column.count() - 1)))
        for index in self.settings.get('y_columns', [1]):
            if 0 <= int(index) < self.y_columns.count():
                self.y_columns.item(int(index)).setSelected(True)
        if not self.y_columns.selectedItems() and self.y_columns.count() > 1:
            self.y_columns.item(1).setSelected(True)

    def paste(self):
        start = self.table.currentIndex()
        if not start.isValid():
            start = self.model.index(0, 0)
        text = W.QApplication.clipboard().text()
        if not text:
            return
        self.model.paste(start, text)
        self.refresh_column_selectors()

    def copy_selection(self):
        indexes = self.table.selectionModel().selectedIndexes()
        if not indexes:
            return
        rows = [index.row() for index in indexes]
        columns = [index.column() for index in indexes]
        selected = {(index.row(), index.column()) for index in indexes}
        lines = []
        for row in range(min(rows), max(rows) + 1):
            lines.append('\t'.join(self.model.data(self.model.index(row, column)) if (row, column) in selected else ''
                                   for column in range(min(columns), max(columns) + 1)))
        W.QApplication.clipboard().setText('\n'.join(lines))

    def clear_selection(self):
        self.model.clear_indexes(self.table.selectionModel().selectedIndexes())

    def add_rows(self):
        self.model.add_rows()

    def add_column(self):
        self.model.add_column()
        self.refresh_column_selectors()

    def rename_columns(self):
        old = ', '.join(self.model.columns[:self.model.used_columns])
        value, accepted = W.QInputDialog.getText(self, 'Rename columns',
                                                  'Names in left-to-right order (comma separated):', text=old)
        if accepted:
            names = [part.strip() for part in value.split(',')]
            if names:
                self.model.set_column_names(names)
                self.refresh_column_selectors()

    def selected_y_columns(self):
        return [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in self.y_columns.selectedItems()]

    def current_settings(self):
        colors = literal(self.colors.text()) or []
        if isinstance(colors, str):
            colors = [colors]
        if not isinstance(colors, (list, tuple)) or not all(is_color_like(value) for value in colors):
            raise ValueError('Line colors must be a color or list of valid Matplotlib colors.')
        return dict(x_column=self.x_column.currentData(), y_columns=self.selected_y_columns(),
                    kind=self.kind.currentText(), title=self.title.text(), xlabel=self.xlabel.text(),
                    ylabel=self.ylabel.text(), colors=list(colors), grid=self.grid.isChecked(),
                    legend=self.legend.isChecked())

    def render_empty_preview(self):
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.text(.5, .5, 'Paste or enter data, then choose columns and update the preview.',
                  ha='center', va='center', transform=axes.transAxes, wrap=True)
        axes.set_axis_off()
        self.canvas.draw_idle()

    def render_preview(self):
        settings = self.current_settings()
        x_column, y_columns = settings['x_column'], settings['y_columns']
        if x_column is None or not y_columns:
            raise ValueError('Choose one X column and at least one Y column.')
        data = self.model.numeric_data()
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        colors = settings['colors']
        plotted = 0
        for order, y_column in enumerate(y_columns):
            if y_column == x_column:
                continue
            valid = np.isfinite(data[:, x_column]) & np.isfinite(data[:, y_column])
            if not valid.any():
                continue
            kwargs = dict(label=self.model.columns[y_column])
            if colors:
                kwargs['color'] = colors[order % len(colors)]
            if settings['kind'] == 'Markers':
                axes.plot(data[valid, x_column], data[valid, y_column], linestyle='', marker='o', **kwargs)
            elif settings['kind'] == 'Line + markers':
                axes.plot(data[valid, x_column], data[valid, y_column], marker='o', **kwargs)
            elif settings['kind'] == 'Scatter':
                axes.scatter(data[valid, x_column], data[valid, y_column], **kwargs)
            else:
                axes.plot(data[valid, x_column], data[valid, y_column], **kwargs)
            plotted += 1
        if not plotted:
            raise ValueError('No numeric X/Y pairs were found in the selected columns.')
        axes.set_title(settings['title'])
        axes.set_xlabel(settings['xlabel'] or self.model.columns[x_column])
        axes.set_ylabel(settings['ylabel'] or 'Value')
        axes.grid(settings['grid'])
        if settings['legend'] and plotted:
            axes.legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def render_safely(self):
        try:
            self.render_preview()
        except Exception as error:
            W.QMessageBox.warning(self, 'Cannot plot manual data', str(error))

    def save(self):
        try:
            manual = self.model.serializable()
            manual['settings'] = self.current_settings()
            self.studio.project.manual = manual
            self.studio.dirty = True
            self.accept()
        except Exception as error:
            W.QMessageBox.warning(self, 'Cannot save manual data', str(error))


class PlotPage(W.QWidget):
    def __init__(self, window, index):
        super().__init__()
        self.window, self.index = window, index
        self.fig = None
        self.source_signature = None
        self.results = []
        outer = W.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        self.controls = W.QWidget()
        left = W.QVBoxLayout(self.controls)
        self.sections = W.QComboBox()
        self.sections.addItems(['Plot & layout', 'Typography', 'Preprocessing', 'Advanced rcParams'])
        left.addWidget(self.sections)
        sections = W.QStackedWidget()
        self.sections.currentIndexChanged.connect(sections.setCurrentIndex)
        self.fields = {}
        state = self.state
        options = dict(state['options'])
        keep = ['ncols', 'nrows', 'figsize', 'horizontal_spacing', 'vertical_spacing'] + (
            ['share_colorbar'] if index == 0 else ['delays', 'snapshot_colors', 'ylim'] if index == 1 else
            ['gsb_wavelength', 'gsb_search_wavelength_range', 'gsb_search_time_range',
             'wavelength_half_width', 'number_of_components', 'fit_range', 'ylim', 'combined', 'log_x',
             'lifetime_data_color', 'lifetime_fit_color'])
        plot_options = W.QWidget()
        plot_layout = W.QVBoxLayout(plot_options)
        self.fields['options'] = Fields({k: options[k] for k in keep})
        visible_config = ['cmap', 'color_vmin', 'color_vmax', 'xlim'] if index < 2 else []
        self.appearance = Fields({k: state['config'][k] for k in visible_config})
        plot_layout.addWidget(self.appearance)
        plot_layout.addWidget(self.fields['options'])
        plot_layout.addStretch()
        sections.addWidget(scroll(plot_options))
        for key, values in [('style', state['style']), ('config',
                            {k:v for k,v in state['config'].items() if k not in ['figsize']+visible_config})]:
            self.fields[key] = Fields(values)
            sections.addWidget(scroll(self.fields[key]))
        self.rc = W.QPlainTextEdit(repr(state['rc']))
        self.rc.setToolTip("Matplotlib defaults, e.g. {'axes.grid': True}")
        sections.addWidget(self.rc)
        left.addWidget(sections, 1)
        button = W.QPushButton('Render figure')
        button.setObjectName('primaryButton')
        button.clicked.connect(lambda: window.run(self.regenerate))
        left.addWidget(button)
        self.results_box = W.QPlainTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setMaximumHeight(120)
        if index == 2:
            left.addWidget(self.results_box)
        self.plot_host = W.QWidget()
        self.plot_host.setObjectName('plotWorkspace')
        self.plot_layout = W.QVBoxLayout(self.plot_host)
        self.placeholder = W.QLabel('Create your next figure\n\nAdd CSV files using File → Add data, then choose Render figure.')
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plot_layout.addWidget(self.placeholder)
        outer.addWidget(self.plot_host, 1)

    @property
    def state(self):
        return self.window.project.states[self.index]

    def collect(self):
        result = copy.deepcopy(self.state)
        for key, fields in self.fields.items():
            result[key].update(fields.values())
        result['config'].update(self.appearance.values())
        result['rc'] = literal(self.rc.toPlainText())
        if not isinstance(result['rc'], dict):
            raise ValueError('rcParams must be a dictionary.')
        return result

    def regenerate(self):
        old = self.state
        new = self.collect()
        if any(new['options'][key] != old['options'].get(key, 0.5)
               for key in ('horizontal_spacing', 'vertical_spacing')):
            # Explicit panel spacing takes precedence over older manual axes positions.
            for key, props in new['edits'].items():
                if key.startswith('axes/') and key.count('/') == 1:
                    props.pop('position', None)
        # Changes to plotting settings intentionally reset a previous zoom.
        if any(new[k] != old[k] for k in ['options', 'config']):
            new['views'] = []
        elif self.fig is not None and old['views']:
            new['views'] = capture_views(self.fig)
        self.window.sync_labels()
        self.window.project.states[self.index] = new
        W.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            fig, results = render(self.window.project, self.index)
        except Exception:
            self.window.project.states[self.index] = old
            raise
        finally:
            W.QApplication.restoreOverrideCursor()
        while self.plot_layout.count():
            w = self.plot_layout.takeAt(0).widget()
            if w:
                w.hide()
                w.deleteLater()
        self.fig, self.results = fig, results
        self.source_signature = self.window.source_signature()
        self.canvas = ResponsiveCanvas(fig)
        self.viewport = FigureViewport(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        fit_labels = self.toolbar.addAction('Fit labels')
        fit_labels.setToolTip('Adjust outer margins to keep labels and colorbar titles inside the figure.')
        fit_labels.triggered.connect(lambda: self.window.run(self.fit_labels))
        self.add_text_action = self.toolbar.addAction('Add annotation')
        self.add_text_action.setCheckable(True)
        self.add_text_action.setToolTip('Enable, then click inside a plot to place editable text.')
        self.canvas.mpl_connect('button_press_event', lambda event: self.window.run(lambda: self.on_plot_click(event)))
        # Persistent edits go through our inspector. Keep navigation/export only.
        for action in list(self.toolbar.actions()):
            if action.text() in ('Customize', 'Subplots'):
                self.toolbar.removeAction(action)
        self.toolbar.setVisible(self.window.navigation_action.isChecked())
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.viewport, 1)
        self.canvas.draw()
        self.results_box.setPlainText('\n'.join(self.fit_summary(r) for r in results))
        self.window.refresh_artists()
        self.window.dirty = True
        self.window.statusBar().showMessage('Click plot text to edit it. Use Add annotation, then click a plot to place new text.')

    def on_plot_click(self, event):
        if event.button != 1 or self.toolbar.mode or event.x is None or event.y is None:
            return
        if self.add_text_action.isChecked():
            if event.inaxes is None:
                return
            self.add_text_action.setChecked(False)
            ax = event.inaxes
            pos = tuple(ax.transAxes.inverted().transform((event.x, event.y)))
            text = ax.text(*pos, 'Annotation', transform=ax.transAxes)
            dialog = TextEditor(text, self)
            if dialog.exec() != W.QDialog.DialogCode.Accepted:
                text.remove()
                return
            props = dialog.values()
            text.set(**props)
            key = 'annotation/' + uuid.uuid4().hex
            text.set_gid(key)
            self.state.setdefault('annotations', []).append(dict(
                id=key, axes=self.fig.axes.index(ax), position=pos, text=props.pop('text'), style=props))
            self.canvas.draw_idle()
            self.window.refresh_artists()
            self.window.dirty = True
            return
        renderer = self.canvas.get_renderer()
        candidates = []
        for key, text in artists(self.fig).items():
            if isinstance(text, Text) and text.get_visible() and text.get_text():
                box = text.get_window_extent(renderer).expanded(1.08, 1.2)
                if box.contains(event.x, event.y):
                    candidates.append((box.width * box.height, key, text))
        if candidates:
            _, key, text = min(candidates, key=lambda item: item[0])
            self.edit_text(key, text)

    def edit_text(self, key, text):
        dialog = TextEditor(text, self)
        if dialog.exec() != W.QDialog.DialogCode.Accepted:
            return
        props = dialog.values()
        apply_artist_edit(self.fig, key, props)
        self.state['edits'].setdefault(key, {}).update(props)
        fit_figure_labels(self.fig)
        self.canvas.draw_idle()
        self.window.dirty = True
        self.window.refresh_artists()
        self.window.artist.setCurrentIndex(self.window.artist.findData(key))

    def fit_labels(self):
        fit_figure_labels(self.fig)
        self.state['views'] = capture_views(self.fig)
        self.canvas.draw_idle()
        self.window.dirty = True

    @staticmethod
    def fit_summary(r):
        n = r['number_of_components']
        amps = r['parameters'][1:1+n]
        taus = r['parameters'][1+n:]
        average = (amps * taus).sum() / amps.sum() if amps.sum() else float('nan')
        return f"{r['label']} | GSB {r['gsb_wavelength']:.2f} nm | mean lifetime {average:.4g} ps\n" + ', '.join(
            f'τ{i+1} = {t:.4g} ± {e:.2g} ps' for i, (t,e) in enumerate(zip(r['lifetimes'], r['lifetime_errors'])))

    def reset_edits(self):
        self.state['edits'] = {}
        self.state['views'] = []
        self.regenerate()


class SourceEditor(W.QDialog):
    """A single dataset's metadata and optional overrides, without changing others."""
    def __init__(self, source, parent):
        super().__init__(parent)
        self.setWindowTitle('Edit data file · ' + source['name'])
        self.resize(590, 700)
        layout = W.QVBoxLayout(self)
        heading = W.QLabel(source['name'])
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.label = W.QLineEdit(source['label'])
        layout.addWidget(W.QLabel('Display label'))
        layout.addWidget(self.label)
        hint = W.QLabel('These settings apply only to this file. Empty values / None inherit the plot settings.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        tabs = W.QTabWidget()
        layout.addWidget(tabs, 1)
        saved = source.get('plot_style', {})
        self.groups = []
        for title, defaults in [('2D heatmap', dict(cmap=None, vmin=None, vmax=None)),
                                ('Snapshots', dict(delays=None, snapshot_colors=[])),
                                ('Lifetimes', dict(components=None, fit_range=None,
                                                  lifetime_data_color='', lifetime_fit_color=''))]:
            fields = Fields({k:saved.get(k, v) for k,v in defaults.items()})
            self.groups.append(fields)
            tabs.addTab(scroll(fields), title)
        processing = W.QWidget()
        box = W.QVBoxLayout(processing)
        self.custom = W.QCheckBox('Use custom preprocessing for this file')
        self.custom.setChecked(bool(source.get('processing')))
        box.addWidget(self.custom)
        values = parent.page().collect()['config']
        keys = ['max_rows', 'background_correction', 'background_time_range',
                'background_wavelength_range', 'smooth_sigma', 'scale_to_milli', 'chirp_correction',
                'chirp_points', 'chirp_poly_coefficients', 'chirp_poly_degree',
                'chirp_reference_wavelength', 'chirp_reference_time']
        values.update(source.get('processing', {}))
        self.processing = Fields({k:values[k] for k in keys})
        box.addWidget(self.processing)
        self.processing.setEnabled(self.custom.isChecked())
        self.custom.toggled.connect(self.processing.setEnabled)
        tabs.addTab(scroll(processing), 'Processing')
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Save | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def validate(self):
        try:
            style = {}
            for group in self.groups:
                style.update(group.values())
            style = {k:v for k,v in style.items() if v is not None and v != '' and v != []}
            if style.get('cmap') and style['cmap'] not in matplotlib.colormaps:
                raise ValueError('Choose a valid color map.')
            if style.get('components') is not None and (not isinstance(style['components'], int) or style['components'] < 1):
                raise ValueError('Components must be a positive integer or None.')
            for key in ('lifetime_data_color', 'lifetime_fit_color'):
                if key in style and not is_color_like(style[key]):
                    raise ValueError('Choose a valid color for ' + key.replace('_', ' '))
            colors = style.get('snapshot_colors', [])
            colors = [colors] if isinstance(colors, str) else colors
            if not all(is_color_like(c) for c in colors):
                raise ValueError('Snapshot colors must be a list of valid colors.')
            if 'delays' in style:
                delays = style['delays']
                if not isinstance(delays, (list, tuple)):
                    delays = [delays]
                if not delays or not all(isinstance(v, (float, int)) for v in delays):
                    raise ValueError('Snapshot times must be numbers, for example [1, 5, 10].')
                style['delays'] = delays
            if style.get('fit_range'):
                start, end = style['fit_range']
                if not 0 < start < end:
                    raise ValueError('Fit range requires 0 < start < end.')
            self.result_values = dict(label=self.label.text().strip() or 'Untitled dataset',
                                      plot_style=style, processing=self.processing.values() if self.custom.isChecked() else {})
            self.accept()
        except Exception as error:
            W.QMessageBox.warning(self, 'Check file settings', str(error))


class Studio(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = Project()
        self.path = None
        self.dirty = False
        self.setWindowTitle('TAS Studio — transient absorption figure editor')
        self.resize(1500, 950)
        self.setMinimumSize(950, 640)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f5f8; color: #243247; }
            QMenuBar, QMenu, QToolBar { background: #ffffff; spacing: 8px; }
            QMenuBar::item { padding: 8px 16px; }
            QMenu::item { padding: 8px 28px; }
            QMenu::item:selected { background: #dbeeea; }
            QDockWidget::title { background: #e6ebf1; padding: 9px; font-weight: bold; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {
                background: white; border: 1px solid #cdd6e1; border-radius: 5px;
                padding: 6px; min-height: 22px;
            }
            QPushButton { background: white; border: 1px solid #cdd6e1; border-radius: 5px; padding: 8px 12px; }
            QPushButton:hover { background: #e4efed; }
            QPushButton#primaryButton { background: #167568; color: white; border: none; font-weight: bold; }
            QTabBar::tab { background: #e6ebf1; padding: 12px 22px; }
            QTabBar::tab:selected { background: white; color: #116658; border-bottom: 3px solid #167568; }
            QScrollArea { border: none; }
            QWidget#plotWorkspace { background: #e9eef3; border-radius: 8px; }
            QLabel#brandTitle { font-size: 19pt; font-weight: 700; }
            QStatusBar { background: #ffffff; }
        """)
        self.file_menu = self.menuBar().addMenu('&File')
        self.edit_menu = self.menuBar().addMenu('&Edit')
        self.preprocess_menu = self.menuBar().addMenu('&Preprocess')
        self.plot_menu = self.menuBar().addMenu('&Plot')
        self.manual_menu = self.menuBar().addMenu('&Manual')
        self.style_menu = self.menuBar().addMenu('&Style')
        self.view_menu = self.menuBar().addMenu('&View')
        self.project_bar = self.addToolBar('Quick actions')
        self.project_bar.setObjectName('quickActions')
        commands = [('New project', self.new_project, 'Ctrl+N'), ('Open…', self.open_project, 'Ctrl+O'),
                    ('Add data…', self.add_files, 'Ctrl+I'), ('Save', self.save_project, 'Ctrl+S'),
                    ('Save as…', lambda: self.save_project(True), 'Ctrl+Shift+S'),
                    ('Export figure…', self.export, 'Ctrl+E')]
        for title, fn, shortcut in commands:
            action = self.add_command(self.file_menu, title, fn, shortcut)
            if title in ('Add data…', 'Save', 'Export figure…'):
                self.project_bar.addAction(action)
        self.file_menu.addSeparator()
        self.add_command(self.file_menu, 'Import font…', self.import_font)
        self.file_menu.addSeparator()
        self.add_command(self.file_menu, 'Exit', self.close, 'Ctrl+Q')
        for title, fn in [('Edit embedded data…', self.edit_data), ('Edit selected file…', self.edit_source), ('Remove selected file', self.remove_source),
                          ('Typography', self.show_fonts), ('Add annotation', self.add_annotation),
                          ('Fit labels', self.fit_current), ('Reset figure edits', lambda: self.page().reset_edits())]:
            self.add_command(self.edit_menu, title, fn)
        self.add_command(self.preprocess_menu, 'Configure selected file…', self.edit_source)
        self.preprocess_menu.addSeparator()
        self.add_command(self.preprocess_menu, 'Enable background subtraction',
                         lambda: self.set_selected_processing('background_correction', True))
        self.add_command(self.preprocess_menu, 'Disable background subtraction',
                         lambda: self.set_selected_processing('background_correction', False))
        self.preprocess_menu.addSeparator()
        self.add_command(self.preprocess_menu, 'Enable chirp correction',
                         lambda: self.set_selected_processing('chirp_correction', True))
        self.add_command(self.preprocess_menu, 'Disable chirp correction',
                         lambda: self.set_selected_processing('chirp_correction', False))
        self.preprocess_menu.addSeparator()
        self.add_command(self.preprocess_menu, 'Clear cached processed data', self.clear_processing_cache)
        for title, tab, section in [('Render active plot', None, None), ('2D ΔA heatmap settings', 0, 0),
                                    ('Snapshot plot settings', 1, 0), ('Lifetime plot settings', 2, 0),
                                    ('Matrix layout', None, 0)]:
            self.add_command(self.plot_menu, title,
                             self.render_active if tab is None and section is None else
                             lambda tab=tab, section=section: self.show_plot_settings(tab, section))
        self.plot_menu.addSeparator()
        self.add_command(self.plot_menu, 'Manual data table and plot…', self.open_manual_plot)
        self.add_command(self.manual_menu, 'Manual data table and plot…', self.open_manual_plot, 'Ctrl+M')
        self.add_command(self.style_menu, 'Typography', self.show_fonts)
        self.add_command(self.style_menu, 'Color maps and ΔA limits', lambda: self.show_plot_settings(0, 0))
        self.add_command(self.style_menu, 'Per-file plot style…', self.edit_source)
        self.add_command(self.style_menu, 'Advanced artist inspector', self.show_inspector)
        self.settings_dock = W.QDockWidget('Figure settings', self)
        self.settings_dock.setObjectName('figureSettings')
        self.settings_dock.setMinimumWidth(310)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)
        self.view_menu.addAction(self.settings_dock.toggleViewAction())
        self.view_menu.addAction(self.project_bar.toggleViewAction())
        self.navigation_action = self.view_menu.addAction('Plot navigation toolbar')
        self.navigation_action.setCheckable(True)
        self.navigation_action.setChecked(True)
        self.navigation_action.toggled.connect(self.toggle_navigation)
        dock = W.QDockWidget('Data files', self)
        self.data_dock = dock
        dock.setObjectName('dataFiles')
        self.view_menu.addAction(dock.toggleViewAction())
        box = W.QWidget()
        v = W.QVBoxLayout(box)
        self.sources = W.QListWidget()
        self.sources.itemChanged.connect(self.mark_dirty)
        v.addWidget(self.sources)
        self.sources.itemDoubleClicked.connect(lambda item: self.run(self.edit_source))
        edit_file = W.QPushButton('Edit selected file…')
        edit_file.clicked.connect(lambda: self.run(self.edit_source))
        v.addWidget(edit_file)
        remove = W.QPushButton('Remove selected dataset')
        remove.clicked.connect(lambda: self.run(self.remove_source))
        v.addWidget(remove)
        dock.setWidget(box)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setMaximumHeight(225)
        inspector = W.QDockWidget('Advanced artist inspector', self)
        self.inspector_dock = inspector
        inspector.setObjectName('artistInspector')
        self.view_menu.addAction(inspector.toggleViewAction())
        panel = W.QWidget()
        panel.setMaximumWidth(310)
        form = W.QFormLayout(panel)
        form.setRowWrapPolicy(W.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.artist = W.QComboBox()
        self.prop = W.QComboBox()
        self.value = W.QPlainTextEdit()
        self.value.setMaximumHeight(110)
        self.help = W.QLabel('Select an object and property. Values use Python literal syntax.')
        self.help.setWordWrap(True)
        self.help.setMaximumWidth(290)
        self.artist.currentIndexChanged.connect(self.refresh_properties)
        self.prop.currentTextChanged.connect(self.show_property)
        form.addRow('Object', self.artist)
        form.addRow('Property', self.prop)
        form.addRow(self.value)
        apply = W.QPushButton('Apply property')
        apply.clicked.connect(lambda: self.run(self.apply_property))
        form.addRow(apply)
        form.addRow(self.help)
        inspector.setWidget(panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, inspector)
        inspector.hide()
        self.view_menu.addSeparator()
        self.add_command(self.view_menu, 'Restore default workspace', self.restore_workspace)
        self.rebuild_pages()
        self.resizeDocks([self.settings_dock], [350], QtCore.Qt.Orientation.Horizontal)
        self.statusBar().showMessage('Ready · Add CSV files to begin')

    def add_command(self, menu, title, function, shortcut=None):
        action = menu.addAction(title)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda checked=False: self.run(function))
        return action

    def source_signature(self):
        return [{k:v for k,v in s.items() if k != 'data'} |
                {'data_hash': hashlib.sha256(s['data']).hexdigest()}
                for s in self.project.sources]

    def selected_source(self):
        index = self.sources.currentRow()
        if index < 0:
            raise ValueError('Select a data file first.')
        return self.project.sources[index]

    def edit_data(self):
        source = self.selected_source()
        dialog = EditableDataDialog(self.project, source, self)
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            self.dirty = True
            self.statusBar().showMessage('Embedded TAS data updated. Render any plot to use the edited values.')

    def open_manual_plot(self):
        dialog = ManualPlotDialog(self)
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            self.statusBar().showMessage('Manual table and plot settings are ready to save in this editable project.')

    def set_selected_processing(self, name, value):
        source = self.selected_source()
        source.setdefault('processing', {})[name] = value
        self.project.clear_cache(source)
        self.dirty = True
        self.statusBar().showMessage(f'{name.replace("_", " ").capitalize()} updated for {source["label"]}.')

    def clear_processing_cache(self):
        self.project.clear_cache()
        self.statusBar().showMessage('Processed-data cache cleared. The next render will recalculate it.')

    def render_active(self):
        self.page().regenerate()

    def show_plot_settings(self, tab, section):
        if tab is not None:
            self.tabs.setCurrentIndex(tab)
        self.settings_dock.show()
        self.page().sections.setCurrentIndex(section)

    def show_inspector(self):
        self.inspector_dock.show()

    def toggle_navigation(self, visible):
        for page in getattr(self, 'pages', []):
            if hasattr(page, 'toolbar'):
                page.toolbar.setVisible(visible)

    def restore_workspace(self):
        for dock in (self.settings_dock, self.data_dock, self.inspector_dock):
            dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.data_dock)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)
        self.settings_dock.show()
        self.data_dock.show()
        self.inspector_dock.hide()
        self.project_bar.show()
        self.navigation_action.setChecked(True)
        self.resizeDocks([self.settings_dock], [350], QtCore.Qt.Orientation.Horizontal)

    def new_project(self):
        if self.confirm_discard():
            self.project, self.path = Project(), None
            self.fill_sources()
            self.rebuild_pages()
            self.dirty = False
            self.setWindowTitle('TAS Studio — Untitled')

    def show_fonts(self):
        self.settings_dock.show()
        self.page().sections.setCurrentIndex(1)

    def import_font(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, 'Import fonts', '', 'Font files (*.ttf *.otf)')
        for path in paths:
            font = self.project.register_font(Path(path).name, Path(path).read_bytes())
            QtGui.QFontDatabase.addApplicationFont(font['path'])
            self.page().fields['style'].inputs['font_family'].setCurrentFont(QtGui.QFont(font['family']))
            self.dirty = True
        if paths:
            self.show_fonts()
            self.statusBar().showMessage('Font imported and embedded in this project. Click Render to apply.')

    def add_annotation(self):
        if not self.page().fig:
            self.page().regenerate()
        self.page().add_text_action.setChecked(True)
        self.statusBar().showMessage('Click inside the plot to place your annotation.')

    def fit_current(self):
        if self.page().fig:
            self.page().fit_labels()

    def edit_source(self):
        index = self.sources.currentRow()
        source = self.selected_source()
        self.sync_labels()
        dialog = SourceEditor(source, self)
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            source.update(dialog.result_values)
            self.project.clear_cache(source)
            self.fill_sources()
            self.sources.setCurrentRow(index)
            self.dirty = True
            self.statusBar().showMessage('File settings updated. Render each plot to apply them.')

    def mark_dirty(self, *args):
        self.dirty = True

    def run(self, fn):
        try:
            return fn()
        except Exception as e:
            W.QMessageBox.critical(self, 'TAS Studio', str(e))

    def rebuild_pages(self):
        old = self.centralWidget()
        self.tabs = W.QTabWidget()
        for font in self.project.fonts:
            QtGui.QFontDatabase.addApplicationFont(font['path'])
        self.pages = [PlotPage(self, i) for i in range(3)]
        settings = W.QStackedWidget()
        for page in self.pages:
            settings.addWidget(page.controls)
        self.settings_dock.setWidget(settings)
        self.tabs.currentChanged.connect(settings.setCurrentIndex)
        for page, name in zip(self.pages, KINDS):
            self.tabs.addTab(page, name)
        workspace = W.QWidget()
        layout = W.QVBoxLayout(workspace)
        layout.setContentsMargins(18, 12, 18, 10)
        header = W.QHBoxLayout()
        title = W.QLabel('TAS <span style="color:#0b8585">STUDIO</span>')
        title.setObjectName('brandTitle')
        subtitle = W.QLabel('  /  Transient absorption analysis workspace')
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()
        render = W.QPushButton('Render active plot')
        render.setObjectName('primaryButton')
        render.clicked.connect(lambda: self.run(self.render_active))
        header.addWidget(render)
        layout.addLayout(header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(workspace)
        if old:
            old.deleteLater()
        self.tabs.currentChanged.connect(self.refresh_artists)
        self.refresh_artists()

    def page(self):
        return self.pages[self.tabs.currentIndex()]

    def sync_labels(self):
        for i, s in enumerate(self.project.sources):
            s['label'] = self.sources.item(i).text()

    def fill_sources(self):
        self.sources.clear()
        for s in self.project.sources:
            item = W.QListWidgetItem(s['label'])
            item.setToolTip(s['name'])
            item.setToolTip(s['name'] + '\nDouble-click to edit this file’s settings')
            self.sources.addItem(item)

    def add_files(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, 'Add transient absorption CSV', '', 'CSV (*.csv);;All files (*)')
        self.sync_labels()
        for p in paths:
            self.project.add(p)
        if paths:
            self.fill_sources()
            self.dirty = True

    def remove_source(self):
        index = self.sources.currentRow()
        if index >= 0:
            self.sync_labels()
            for page in self.pages:
                self.project.states[page.index] = page.collect()
            del self.project.sources[index]
            for s in self.project.states:
                s['edits'], s['views'] = {}, []
                s['annotations'] = []
            self.fill_sources()
            self.rebuild_pages()
            self.dirty = True

    def confirm_discard(self):
        # Pending field edits are not necessarily rendered yet.
        changed = self.dirty or any(p.collect() != p.state for p in self.pages)
        if not changed:
            return True
        choice = W.QMessageBox.question(self, 'Save project?', 'Save your current project before continuing?',
                    W.QMessageBox.StandardButton.Save | W.QMessageBox.StandardButton.Discard | W.QMessageBox.StandardButton.Cancel)
        if choice == W.QMessageBox.StandardButton.Save:
            return bool(self.save_project())
        return choice == W.QMessageBox.StandardButton.Discard

    def open_project(self):
        path, _ = W.QFileDialog.getOpenFileName(self, 'Open project', '', 'TAS projects (*.tas)')
        if path:
            loaded = Project.load(path)
            if not self.confirm_discard():
                return
            self.project, self.path = loaded, path
            self.fill_sources()
            self.rebuild_pages()
            self.page().regenerate()
            self.dirty = False
            self.setWindowTitle(f'TAS Studio — {Path(path).name}')

    def save_project(self, save_as=False):
        path = self.path
        if not path or save_as:
            path, _ = W.QFileDialog.getSaveFileName(self, 'Save editable project', path or 'Untitled.tas', 'TAS project (*.tas)')
        if not path:
            return False
        if not path.lower().endswith('.tas'):
            path += '.tas'
        self.sync_labels()
        for page in self.pages:
            signature = self.source_signature()
            if page.collect() != page.state or (page.fig is not None and page.source_signature != signature):
                if self.project.sources:
                    page.regenerate()
                else:
                    self.project.states[page.index] = page.collect()
            if page.fig:
                page.state['views'] = capture_views(page.fig)
        self.project.save(path)
        self.path, self.dirty = path, False
        self.setWindowTitle(f'TAS Studio — {Path(path).name}')
        self.statusBar().showMessage('Saved editable embedded data, preprocessing, styles, and all figure settings.')
        return True

    def export(self):
        page = self.page()
        if page.fig is None or page.collect() != page.state or page.source_signature != self.source_signature():
            page.regenerate()
        path, _ = W.QFileDialog.getSaveFileName(self, 'Export current figure', 'figure.png',
                    'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tiff);;EPS (*.eps)')
        if path:
            page.fig.savefig(path, dpi=page.state['config']['dpi'],
                             transparent=page.state['config']['transparent'], bbox_inches=None)
            self.statusBar().showMessage(f'Exported {path}')

    def refresh_artists(self, *args):
        self.artist.clear()
        page = self.page()
        if page.fig:
            for key, obj in artists(page.fig).items():
                self.artist.addItem(f'{key} · {type(obj).__name__}', key)

    def current_artist(self):
        page = self.page()
        return artists(page.fig).get(self.artist.currentData()) if page.fig else None

    def refresh_properties(self, *args):
        self.prop.clear()
        obj = self.current_artist()
        if obj is None:
            return
        # Object-valued transforms/callbacks cannot be represented in portable JSON.
        excluded = {'figure', 'transform', 'clip_path', 'clip_box', 'agg_filter', 'axes',
                    'data', 'xdata', 'ydata', 'array', 'offsets', 'paths', 'verts', 'norm',
                    'animated', 'picker', 'pickradius', 'url', 'urls', 'gid'}
        for name in ArtistInspector(obj).get_setters():
            if name not in excluded:
                self.prop.addItem(name)

    def show_property(self, name):
        obj = self.current_artist()
        if obj is None or not name:
            return
        try:
            val = getattr(obj, 'get_' + name)()
            if hasattr(val, 'tolist'):
                val = val.tolist()
            if name == 'cmap':
                val = val.name
            if name == 'position' and hasattr(val, 'bounds'):
                val = tuple(val.bounds)
            self.value.setPlainText(repr(val))
        except Exception:
            self.value.setPlainText('')
        self.help.setText(str(ArtistInspector(obj).get_valid_values(name))[:1300])

    def apply_property(self):
        obj = self.current_artist()
        if obj is None:
            return
        prop, val = self.prop.currentText(), literal(self.value.toPlainText())
        json.dumps(packed(val), allow_nan=False)
        apply_artist_edit(self.page().fig, self.artist.currentData(), {prop: val})
        fit_figure_labels(self.page().fig)
        self.page().canvas.draw()
        self.page().state['edits'].setdefault(self.artist.currentData(), {})[prop] = val
        self.dirty = True

    def closeEvent(self, event):
        try:
            if self.confirm_discard():
                event.accept()
            else:
                event.ignore()
        except Exception as e:
            W.QMessageBox.critical(self, 'Cannot save', str(e))
            event.ignore()


def main():
    app = W.QApplication(sys.argv)
    configure_font(app)
    app.setStyle('Fusion')
    window = Studio()
    window.show()
    if len(sys.argv) > 1:
        def load_argument():
            window.project = Project.load(sys.argv[1])
            window.path = sys.argv[1]
            window.fill_sources()
            window.rebuild_pages()
            window.page().regenerate()
            window.dirty = False
        window.run(load_argument)
    sys.exit(app.exec())


def configure_font(app):
    # A bundled font also makes headless Qt previews independent of OS fonts.
    path = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    font_id = QtGui.QFontDatabase.addApplicationFont(str(path))
    families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
    if families:
        app.setFont(QtGui.QFont(families[0], 9))


if __name__ == '__main__':
    main()
