"""PyQt5 可视化 Demo。

界面:
- 左侧：原始道路图片/视频；
- 右侧：检测结果；
- 下方：检测类别、置信度、FPS。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import torch
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from models.register_ultralytics import register_custom_modules


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "train" / "best.pt"


def select_device() -> str:
    """自动选择 CUDA。"""

    return "0" if torch.cuda.is_available() else "cpu"


def cv_to_pixmap(frame) -> QPixmap:
    """OpenCV BGR 图像转 QPixmap。"""

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class ImagePane(QLabel):
    """自适应显示图像的 QLabel。"""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setStyleSheet("QLabel { background: #111827; color: #e5e7eb; border: 1px solid #374151; }")
        self._pixmap: QPixmap | None = None

    def set_frame(self, frame) -> None:
        self._pixmap = cv_to_pixmap(frame)
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 方法名
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class RoadMarkDemo(QMainWindow):
    """路面标线缺失检测 GUI 主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RoadMarkMissingDetection - YOLO26")
        self.resize(1280, 760)

        register_custom_modules()
        self.device = select_device()
        self.model = self._load_default_model()
        self.cap: cv2.VideoCapture | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self.raw_view = ImagePane("原始画面")
        self.result_view = ImagePane("检测结果")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "坐标 xyxy"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fps_label = QLabel("FPS: 0.00")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._build_ui()
        self.statusBar().showMessage(f"设备: {self.device} | 模型: {self.current_model_name}")

    def _build_ui(self) -> None:
        toolbar = QToolBar("工具")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        style = self.style()
        self._add_action(toolbar, "加载模型", style.standardIcon(QStyle.SP_DialogOpenButton), self.load_model)
        self._add_action(toolbar, "打开图片", style.standardIcon(QStyle.SP_FileIcon), self.open_image)
        self._add_action(toolbar, "打开视频", style.standardIcon(QStyle.SP_MediaPlay), self.open_video)
        self._add_action(toolbar, "摄像头", style.standardIcon(QStyle.SP_ComputerIcon), self.open_camera)
        self._add_action(toolbar, "停止", style.standardIcon(QStyle.SP_MediaStop), self.stop_stream)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.raw_view)
        splitter.addWidget(self.result_view)
        splitter.setSizes([640, 640])

        bottom = QHBoxLayout()
        bottom.addWidget(self.table, stretch=5)
        bottom.addWidget(self.fps_label, stretch=1)

        layout = QVBoxLayout()
        layout.addWidget(splitter, stretch=5)
        layout.addLayout(bottom, stretch=1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    @staticmethod
    def _add_action(toolbar: QToolBar, text: str, icon, callback) -> None:
        action = QAction(icon, text, toolbar)
        action.triggered.connect(callback)
        toolbar.addAction(action)

    def _load_default_model(self) -> YOLO:
        if DEFAULT_WEIGHTS.exists():
            self.current_model_name = str(DEFAULT_WEIGHTS)
            return YOLO(str(DEFAULT_WEIGHTS))
        self.current_model_name = "yolo26n.pt"
        return YOLO("yolo26n.pt")

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(ROOT), "Weights (*.pt);;YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.model = YOLO(path)
            self.current_model_name = path
            self.statusBar().showMessage(f"设备: {self.device} | 模型: {path}")
        except Exception as exc:  # pragma: no cover - GUI 弹窗路径
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def open_image(self) -> None:
        self.stop_stream()
        path, _ = QFileDialog.getOpenFileName(
            self, "选择道路图片", str(ROOT), "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "读取失败", path)
            return
        self._process_frame(frame)

    def open_video(self) -> None:
        self.stop_stream()
        path, _ = QFileDialog.getOpenFileName(self, "选择道路视频", str(ROOT), "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if not path:
            return
        self._open_capture(path)

    def open_camera(self) -> None:
        self.stop_stream()
        self._open_capture(0)

    def _open_capture(self, source) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "打开失败", str(source))
            self.cap = None
            return
        self.timer.start(1)

    def stop_stream(self) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _next_frame(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop_stream()
            return
        self._process_frame(frame)

    def _process_frame(self, frame) -> None:
        start = time.perf_counter()
        result = self.model.predict(frame, imgsz=640, conf=0.25, device=self.device, verbose=False)[0]
        fps = 1.0 / max(time.perf_counter() - start, 1e-6)
        annotated = result.plot()

        self.raw_view.set_frame(frame)
        self.result_view.set_frame(annotated)
        self._update_table(result, fps)

    def _update_table(self, result, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.2f}")
        boxes = result.boxes
        if boxes is None:
            self.table.setRowCount(0)
            return
        self.table.setRowCount(len(boxes))
        for row, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = ", ".join(f"{v:.1f}" for v in box.xyxy[0].tolist())
            self.table.setItem(row, 0, QTableWidgetItem(result.names[cls_id]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{conf:.3f}"))
            self.table.setItem(row, 2, QTableWidgetItem(xyxy))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 方法名
        self.stop_stream()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = RoadMarkDemo()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
