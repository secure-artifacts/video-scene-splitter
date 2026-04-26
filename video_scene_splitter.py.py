import sys
import shlex
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, Signal
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySceneDetect 镜头分段工具 (PySide6)")
        self.resize(980, 720)

        self.jobs: list[Job] = []
        self.current_job_index = -1
        self.user_stopped = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        info = QLabel(
            "拖拽视频到列表；拖拽输出文件夹到路径框；点击开始后将按镜头切分。\n"
            "需要：当前 Python 环境已安装 PySide6 / scenedetect / opencv-python，且系统 PATH 中有 ffmpeg。"
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

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)

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

    def _check_runtime_dependencies(self) -> bool:
        try:
            import scenedetect  # noqa: F401
        except Exception as exc:
            QMessageBox.critical(
                self,
                "缺少依赖",
                f"当前 Python 环境无法导入 scenedetect：\n{exc}\n\n请先执行：pip install scenedetect opencv-python",
            )
            return False

        ffmpeg_proc = QProcess(self)
        ffmpeg_proc.start("ffmpeg", ["-version"])
        if not ffmpeg_proc.waitForStarted(1500):
            QMessageBox.critical(
                self,
                "缺少 ffmpeg",
                "无法启动 ffmpeg。请先安装 ffmpeg，并确保它在系统 PATH 中。",
            )
            return False
        ffmpeg_proc.kill()
        ffmpeg_proc.waitForFinished(1000)
        return True

    def start_all(self):
        if self.proc.state() != QProcess.NotRunning:
            return

        videos = self._all_video_paths()
        if not videos:
            QMessageBox.warning(self, "提示", "请先添加至少一个视频。")
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请设置输出文件夹。")
            return

        if not self._check_runtime_dependencies():
            return

        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        self.jobs = [Job(video_path=v, out_dir=str(out_path)) for v in videos]
        self.current_job_index = -1
        self.user_stopped = False

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.append_log("========== 开始批处理 ==========")
        self._start_next_job()

    def stop_all(self):
        self.user_stopped = True
        if self.proc.state() != QProcess.NotRunning:
            self.append_log("[!] 正在停止当前任务…")
            self.proc.kill()
        else:
            self._reset_after_stop()
            self.append_log("========== 已停止 ==========")

    def _reset_after_stop(self):
        self.jobs = []
        self.current_job_index = -1
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(0)

    def _build_command(self, video_path: Path, out_sub: Path) -> list[str]:
        detector = self.detector_combo.currentText().strip()
        threshold = self.threshold_spin.value()
        min_scene = self.min_scene_spin.value()

        cmd = [
            sys.executable,
            "-m",
            "scenedetect",
            "-i",
            str(video_path),
            "-m",
            f"{min_scene:.3f}s",
        ]

        if detector == "content":
            cmd += ["detect-content", "-t", f"{threshold:.3f}"]
        elif detector == "threshold":
            cmd += ["detect-threshold", "-t", f"{threshold:.3f}"]
        else:
            raise ValueError(f"未知检测器: {detector}")

        cmd += ["split-video", "-o", str(out_sub)]
        return cmd

    def _start_next_job(self):
        if self.user_stopped:
            self._reset_after_stop()
            return

        self.current_job_index += 1
        if self.current_job_index >= len(self.jobs):
            self.append_log("========== 全部完成 ✅ ==========")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress.setValue(100)
            self.jobs = []
            self.current_job_index = -1
            return

        job = self.jobs[self.current_job_index]
        video_path = Path(job.video_path)
        out_sub = Path(job.out_dir) / video_path.stem
        out_sub.mkdir(parents=True, exist_ok=True)

        try:
            cmd = self._build_command(video_path, out_sub)
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"构造命令失败：\n{exc}")
            self.stop_all()
            return

        self.append_log("")
        self.append_log(f"[{self.current_job_index + 1}/{len(self.jobs)}] 处理：{video_path.name}")
        self.append_log(f"输出：{out_sub}")
        self.append_log("命令： " + " ".join(shlex.quote(x) for x in cmd))

        self.proc.start(cmd[0], cmd[1:])
        if not self.proc.waitForStarted(2000):
            QMessageBox.critical(
                self,
                "错误",
                "无法启动 scenedetect。请确认当前 Python 环境已安装 scenedetect。",
            )
            self.stop_all()

    def _on_proc_output(self):
        data = self.proc.readAllStandardOutput().data().decode(errors="ignore")
        if data:
            self.append_log(data.rstrip("\n"))

    def _on_proc_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        if self.user_stopped:
            self.append_log("========== 已停止 ==========")
            self._reset_after_stop()
            self.user_stopped = False
            return

        if self.jobs:
            done = self.current_job_index + 1
            total = len(self.jobs)
            self.progress.setValue(int((done / total) * 100))

        if exit_status == QProcess.CrashExit:
            self.append_log(f"[x] 任务崩溃退出（exit_code={exit_code}）")
        elif exit_code == 0:
            self.append_log("[✓] 完成")
        else:
            self.append_log(f"[!] 退出码 {exit_code}（可能是 ffmpeg/scenedetect 参数问题，或视频文件损坏）")

        self._start_next_job()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
