import sys
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".ts"}


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def check_ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return result.returncode == 0
    except Exception:
        return False


class VideoListWidget(QListWidget):
    """支持拖拽多个视频文件/文件夹（文件夹会递归查找视频）"""

    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        videos = []

        for path in paths:
            if path.is_dir():
                for sub in path.rglob("*"):
                    if is_video_file(sub):
                        videos.append(str(sub))
            elif is_video_file(path):
                videos.append(str(path))

        if videos:
            self.filesDropped.emit(videos)

        event.acceptProposedAction()


class FolderLineEdit(QLineEdit):
    """支持拖拽文件夹到输出路径输入框"""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText("拖拽输出文件夹到这里，或点击“选择输出文件夹”")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return

        path = Path(urls[0].toLocalFile())
        if path.exists():
            self.setText(str(path if path.is_dir() else path.parent))

        event.acceptProposedAction()


@dataclass
class Job:
    video_path: str
    out_dir: str


class VideoSplitWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished_all = Signal()
    failed = Signal(str)

    def __init__(
        self,
        jobs: list[Job],
        detector: str,
        threshold: float,
        min_scene_seconds: float,
    ):
        super().__init__()
        self.jobs = jobs
        self.detector = detector
        self.threshold = threshold
        self.min_scene_seconds = min_scene_seconds
        self._stop_requested = False
        self._current_process: subprocess.Popen | None = None

    def stop(self):
        self._stop_requested = True
        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def run(self):
        try:
            import cv2
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector, ThresholdDetector
        except Exception as exc:
            self.failed.emit(f"无法导入依赖：{exc}")
            return

        total = len(self.jobs)

        for index, job in enumerate(self.jobs, start=1):
            if self._stop_requested:
                self.log.emit("========== 已停止 ==========")
                self.finished_all.emit()
                return

            video_path = Path(job.video_path)
            out_sub = Path(job.out_dir) / video_path.stem
            out_sub.mkdir(parents=True, exist_ok=True)

            self.log.emit("")
            self.log.emit(f"[{index}/{total}] 处理：{video_path.name}")
            self.log.emit(f"输出：{out_sub}")

            try:
                fps = self._get_video_fps(video_path, cv2)
                min_scene_len = max(1, int(round(self.min_scene_seconds * fps)))

                self.log.emit(
                    f"检测参数：detector={self.detector}, threshold={self.threshold:.3f}, "
                    f"min_scene={self.min_scene_seconds:.3f}s, fps={fps:.3f}, min_scene_len={min_scene_len} frames"
                )

                video = open_video(str(video_path))
                scene_manager = SceneManager()

                if self.detector == "content":
                    scene_manager.add_detector(
                        ContentDetector(
                            threshold=self.threshold,
                            min_scene_len=min_scene_len,
                        )
                    )
                elif self.detector == "threshold":
                    scene_manager.add_detector(
                        ThresholdDetector(
                            threshold=self.threshold,
                            min_scene_len=min_scene_len,
                        )
                    )
                else:
                    raise ValueError(f"未知检测器：{self.detector}")

                self.log.emit("正在检测镜头边界...")
                scene_manager.detect_scenes(video)
                scene_list = scene_manager.get_scene_list()

                if not scene_list:
                    self.log.emit("[i] 未检测到明显镜头切换，将整个视频作为一个片段处理。")
                    duration = self._get_video_duration(video_path, cv2)
                    scene_ranges = [(0.0, duration)]
                else:
                    scene_ranges = [
                        (start.get_seconds(), end.get_seconds())
                        for start, end in scene_list
                    ]

                self.log.emit(f"检测完成，共 {len(scene_ranges)} 个片段。")
                self._split_with_ffmpeg(video_path, out_sub, scene_ranges)

            except Exception as exc:
                self.log.emit(f"[!] 处理失败：{exc}")

            self.progress.emit(int((index / total) * 100))

        self.log.emit("========== 全部完成 ✅ ==========")
        self.finished_all.emit()

    def _get_video_fps(self, video_path: Path, cv2_module) -> float:
        cap = cv2_module.VideoCapture(str(video_path))
        fps = cap.get(cv2_module.CAP_PROP_FPS)
        cap.release()

        if not fps or fps <= 0:
            return 25.0

        return float(fps)

    def _get_video_duration(self, video_path: Path, cv2_module) -> float:
        cap = cv2_module.VideoCapture(str(video_path))
        fps = cap.get(cv2_module.CAP_PROP_FPS)
        frames = cap.get(cv2_module.CAP_PROP_FRAME_COUNT)
        cap.release()

        if not fps or fps <= 0 or not frames or frames <= 0:
            return 0.0

        return float(frames / fps)

    def _split_with_ffmpeg(
        self,
        video_path: Path,
        out_sub: Path,
        scene_ranges: list[tuple[float, float]],
    ):
        for scene_index, (start_sec, end_sec) in enumerate(scene_ranges, start=1):
            if self._stop_requested:
                return

            if end_sec <= start_sec:
                continue

            output_file = out_sub / f"{video_path.stem}_scene_{scene_index:03d}{video_path.suffix}"

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-ss",
                f"{start_sec:.3f}",
                "-to",
                f"{end_sec:.3f}",
                "-c",
                "copy",
                str(output_file),
            ]

            self.log.emit("FFmpeg： " + quote_cmd(cmd))

            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=creationflags,
            )

            if self._current_process.stdout:
                for line in self._current_process.stdout:
                    line = line.rstrip()
                    if line:
                        self.log.emit(line)

            return_code = self._current_process.wait()
            self._current_process = None

            if return_code == 0:
                self.log.emit(f"[✓] 已输出：{output_file.name}")
            else:
                self.log.emit(f"[!] FFmpeg 退出码：{return_code}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Scene Splitter")
        self.resize(980, 720)

        self.worker: VideoSplitWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        info = QLabel(
            "拖拽视频到列表；拖拽输出文件夹到路径框；点击开始后将按画面/镜头变化切分视频。\n"
            "需要：程序包含 PySceneDetect / OpenCV；系统需要安装 FFmpeg，并确保 ffmpeg 命令可用。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        list_box = QGroupBox("视频列表（可拖拽多个视频/文件夹进来）")
        list_layout = QVBoxLayout(list_box)

        self.video_list = VideoListWidget()
        list_layout.addWidget(self.video_list)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("添加视频…")
        self.btn_remove = QPushButton("移除选中")
        self.btn_clear = QPushButton("清空")

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)

        list_layout.addLayout(btn_row)
        layout.addWidget(list_box)

        cfg_box = QGroupBox("输出与检测参数")
        cfg_layout = QFormLayout(cfg_box)

        out_row = QHBoxLayout()
        self.out_edit = FolderLineEdit()
        self.btn_out = QPushButton("选择输出文件夹…")

        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.btn_out)

        cfg_layout.addRow("输出文件夹：", out_row)

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["content", "threshold"])
        cfg_layout.addRow("检测器：", self.detector_combo)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(27.0)
        cfg_layout.addRow("阈值：", self.threshold_spin)

        self.min_scene_spin = QDoubleSpinBox()
        self.min_scene_spin.setRange(0.0, 60.0)
        self.min_scene_spin.setSingleStep(0.5)
        self.min_scene_spin.setValue(1.0)
        cfg_layout.addRow("最短镜头（秒）：", self.min_scene_spin)

        layout.addWidget(cfg_box)

        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton("开始处理")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)

        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch(1)

        layout.addLayout(ctrl_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self.btn_add.clicked.connect(self.add_videos_dialog)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_out.clicked.connect(self.choose_out_dir)
        self.btn_start.clicked.connect(self.start_all)
        self.btn_stop.clicked.connect(self.stop_all)
        self.video_list.filesDropped.connect(self.add_video_paths)

    def append_log(self, text: str):
        self.log.appendPlainText(text.rstrip("\n"))

    def _all_video_paths(self) -> list[str]:
        return [self.video_list.item(i).text() for i in range(self.video_list.count())]

    def add_video_paths(self, paths: list[str]):
        existing = set(self._all_video_paths())
        added = 0

        for raw in paths:
            path = str(Path(raw).resolve())
            if path not in existing:
                self.video_list.addItem(QListWidgetItem(path))
                existing.add(path)
                added += 1

        if added:
            self.append_log(f"[+] 添加 {added} 个视频")
        else:
            self.append_log("[i] 没有新增视频（可能重复或不是支持的视频格式）")

    def add_videos_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts);;All Files (*.*)",
        )

        if files:
            self.add_video_paths(files)

    def remove_selected(self):
        items = self.video_list.selectedItems()
        if not items:
            return

        for item in items:
            row = self.video_list.row(item)
            self.video_list.takeItem(row)

        self.append_log(f"[-] 移除 {len(items)} 项")

    def clear_all(self):
        self.video_list.clear()
        self.append_log("[~] 已清空列表")

    def choose_out_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if directory:
            self.out_edit.setText(directory)

    def start_all(self):
        if self.worker and self.worker.isRunning():
            return

        videos = self._all_video_paths()
        if not videos:
            QMessageBox.warning(self, "提示", "请先添加至少一个视频。")
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请设置输出文件夹。")
            return

        if not check_ffmpeg_available():
            QMessageBox.critical(
                self,
                "缺少 FFmpeg",
                "无法启动 FFmpeg。\n\n"
                "请先安装 FFmpeg，并确保 ffmpeg.exe 已加入系统 PATH。\n\n"
                "可以在命令行运行下面命令测试：\n"
                "ffmpeg -version",
            )
            return

        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        jobs = [Job(video_path=v, out_dir=str(out_path)) for v in videos]

        self.worker = VideoSplitWorker(
            jobs=jobs,
            detector=self.detector_combo.currentText().strip(),
            threshold=self.threshold_spin.value(),
            min_scene_seconds=self.min_scene_spin.value(),
        )

        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_all.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        self.append_log("========== 开始批处理 ==========")
        self.worker.start()

    def stop_all(self):
        if self.worker and self.worker.isRunning():
            self.append_log("[!] 正在停止当前任务…")
            self.worker.stop()
        else:
            self._on_worker_finished()

    def _on_worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_worker_failed(self, message: str):
        QMessageBox.critical(self, "错误", message)
        self.append_log("[x] " + message)
        self._on_worker_finished()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
