import sys
import itertools
import os
import subprocess
import tempfile
import shutil
import re
import queue
import threading
import atexit
import time
import json
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QSpinBox, QPushButton, 
                               QTextEdit, QProgressBar, QGroupBox, QLineEdit,
                               QTabWidget, QStatusBar, QFileDialog, QSplitter,
                               QGridLayout, QFrame, QPlainTextEdit)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter

# ==========================================
# 1. THREAD-SAFE SHARED STATE & SIGNALS
# ==========================================

class SharedState:
    def __init__(self):
        self.queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.producer_done = threading.Event()
        self.lock = threading.Lock()
        self.best_result = {"allocations": 0, "code": "", "details": "", "language": ""}
        self.processed_count = 0
        self.success_count = 0
        self.total_perms = 0
        self.start_time = time.time()
        # Graph data buffers
        self.allocation_history = []  # (x, y) points for best-allocation-over-time
        self.allocation_samples = []  # all successful allocation values for histogram

class WorkerSignals(QObject):
    progress = Signal(int)
    log = Signal(str)
    stats_update = Signal(dict)
    graph_update = Signal(list, list)

# ==========================================
# 2. PYTHON SYNTAX HIGHLIGHTER
# ==========================================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(QColor("#569CD6"))
        keywords = ["def", "class", "if", "else", "elif", "for", "while", "return", 
                    "import", "from", "as", "try", "except", "finally", "with", "pass",
                    "break", "continue", "lambda", "yield", "True", "False", "None"]
        for kw in keywords:
            self.rules.append((re.compile(rf'\b{kw}\b'), keyword_fmt))
        
        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#CE9178"))
        self.rules.append((re.compile(r'"[^"]*"|\'[^\']*\''), string_fmt))
        
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#6A9955"))
        self.rules.append((re.compile(r'#.*$'), comment_fmt))
        
        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor("#B5CEA8"))
        self.rules.append((re.compile(r'\b\d+\b'), number_fmt))

class GenericHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(QColor("#569CD6"))
        keywords = ["int", "float", "double", "char", "void", "return", "if", "else", 
                    "for", "while", "class", "struct", "fn", "let", "mut", "pub", "mod",
                    "use", "impl", "trait", "where", "match", "public", "private", "static"]
        for kw in keywords:
            self.rules.append((re.compile(rf'\b{kw}\b'), keyword_fmt))
        
        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#CE9178"))
        self.rules.append((re.compile(r'"[^"]*"'), string_fmt))
        
        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#6A9955"))
        self.rules.append((re.compile(r'//.*$|/\*.*?\*/|#.*$'), comment_fmt))

# ==========================================
# 3. COMPILER ANALYZERS (unchanged core logic, abbreviated for space)
# ==========================================

class BaseCompilerAnalyzer:
    EXT = ".txt"
    BOILERPLATE_START = ""
    BOILERPLATE_END = ""
    LANGUAGE_NAME = "Unknown"

    def __init__(self, temp_dir):
        self.temp_dir = temp_dir

    def wrap_code(self, raw_permutation):
        return f"{self.BOILERPLATE_START}\n{raw_permutation}\n{self.BOILERPLATE_END}"

    def compile_and_trace(self, raw_perm, filename_base):
        raise NotImplementedError

    def safe_delete(self, *filepaths):
        for fp in filepaths:
            try:
                if fp and os.path.exists(fp):
                    os.remove(fp)
            except OSError:
                pass

class CAnalyzer(BaseCompilerAnalyzer):
    EXT = ".c"
    BOILERPLATE_START = "int main() {"
    BOILERPLATE_END = "return 0; }"
    LANGUAGE_NAME = "C"

    def compile_and_trace(self, raw_perm, filename_base):
        src_file = os.path.join(self.temp_dir, f"{filename_base}{self.EXT}")
        obj_file = os.path.join(self.temp_dir, "temp.o")
        su_file = os.path.join(self.temp_dir, f"{filename_base}.su")
        try:
            with open(src_file, "w") as f: f.write(self.wrap_code(raw_perm))
            cmd = ["gcc", "-c", "-fstack-usage", "-o", obj_file, src_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(su_file):
                return {"success": False, "allocations": 0, "error": result.stderr[:200]}
            total_bytes, details = 0, []
            with open(su_file, "r") as f:
                for line in f:
                    if line.startswith(src_file):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            size = int(parts[4])
                            total_bytes += size
                            details.append(f"Function {parts[3]}: {size} bytes")
            return {"success": True, "allocations": total_bytes, "details": "\n".join(details), "error": None}
        finally:
            self.safe_delete(src_file, obj_file, su_file)

class CppAnalyzer(CAnalyzer):
    EXT = ".cpp"
    LANGUAGE_NAME = "C++"
    def compile_and_trace(self, raw_perm, filename_base):
        src_file = os.path.join(self.temp_dir, f"{filename_base}{self.EXT}")
        obj_file = os.path.join(self.temp_dir, "temp.o")
        su_file = os.path.join(self.temp_dir, f"{filename_base}.su")
        try:
            with open(src_file, "w") as f: f.write(self.wrap_code(raw_perm))
            cmd = ["g++", "-c", "-fstack-usage", "-o", obj_file, src_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(su_file):
                return {"success": False, "allocations": 0, "error": result.stderr[:200]}
            total_bytes, details = 0, []
            with open(su_file, "r") as f:
                for line in f:
                    if line.startswith(src_file):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            size = int(parts[4])
                            total_bytes += size
                            details.append(f"Function {parts[3]}: {size} bytes")
            return {"success": True, "allocations": total_bytes, "details": "\n".join(details), "error": None}
        finally:
            self.safe_delete(src_file, obj_file, su_file)

class RustAnalyzer(BaseCompilerAnalyzer):
    EXT = ".rs"
    BOILERPLATE_START = "fn main() {"
    BOILERPLATE_END = "}"
    LANGUAGE_NAME = "Rust"

    def compile_and_trace(self, raw_perm, filename_base):
        src_file = os.path.join(self.temp_dir, f"{filename_base}{self.EXT}")
        obj_file = os.path.join(self.temp_dir, f"{filename_base}.o")
        try:
            with open(src_file, "w") as f: f.write(self.wrap_code(raw_perm))
            cmd = ["rustc", "--crate-type=staticlib", "--emit=obj", "-o", obj_file, src_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(obj_file):
                return {"success": False, "allocations": 0, "error": result.stderr[:200]}
            cmd_size = ["size", obj_file]
            size_res = subprocess.run(cmd_size, capture_output=True, text=True)
            total_bytes, details = 0, "Object file generated. "
            if size_res.returncode == 0:
                lines = size_res.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 3:
                        total_bytes = int(parts[1]) + int(parts[2])
                        details += f"Data: {parts[1]}B, BSS: {parts[2]}B"
            return {"success": True, "allocations": total_bytes, "details": details, "error": None}
        finally:
            self.safe_delete(src_file, obj_file)

class PythonAnalyzer(BaseCompilerAnalyzer):
    EXT = ".py"
    BOILERPLATE_START = ""
    BOILERPLATE_END = ""
    LANGUAGE_NAME = "Python"

    def compile_and_trace(self, raw_perm, filename_base):
        src_file = os.path.join(self.temp_dir, f"{filename_base}{self.EXT}")
        try:
            with open(src_file, "w") as f: f.write(self.wrap_code(raw_perm))
            cmd = [sys.executable, "-m", "dis", src_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {"success": False, "allocations": 0, "error": result.stderr[:200]}
            dis_output = result.stdout
            local_allocs = len(re.findall(r'\bSTORE_FAST\b', dis_output))
            global_allocs = len(re.findall(r'\bSTORE_GLOBAL\b', dis_output))
            name_allocs = len(re.findall(r'\bSTORE_NAME\b', dis_output))
            total_allocs = local_allocs + global_allocs + name_allocs
            total_bytes = total_allocs * 8
            details = f"Local: {local_allocs}, Global: {global_allocs}, Name: {name_allocs}"
            return {"success": True, "allocations": total_bytes, "details": details, "error": None}
        finally:
            self.safe_delete(src_file)
            pycache = os.path.join(self.temp_dir, "__pycache__")
            if os.path.exists(pycache):
                shutil.rmtree(pycache, ignore_errors=True)

# Minimal stubs for other languages (full implementations from previous version)
class GoAnalyzer(RustAnalyzer):
    EXT = ".go"
    BOILERPLATE_START = "package main\nfunc main() {"
    LANGUAGE_NAME = "Go"
    def compile_and_trace(self, raw_perm, filename_base):
        src_file = os.path.join(self.temp_dir, f"{filename_base}{self.EXT}")
        obj_file = os.path.join(self.temp_dir, f"{filename_base}.o")
        try:
            with open(src_file, "w") as f: f.write(self.wrap_code(raw_perm))
            cmd = ["go", "tool", "compile", "-o", obj_file, src_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(obj_file):
                return {"success": False, "allocations": 0, "error": result.stderr[:200]}
            cmd_size = ["size", obj_file]
            size_res = subprocess.run(cmd_size, capture_output=True, text=True)
            total_bytes, details = 0, "Object file generated. "
            if size_res.returncode == 0:
                lines = size_res.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 3:
                        total_bytes = int(parts[1]) + int(parts[2])
                        details += f"Data: {parts[1]}B, BSS: {parts[2]}B"
            return {"success": True, "allocations": total_bytes, "details": details, "error": None}
        finally:
            self.safe_delete(src_file, obj_file)

class JavaAnalyzer(BaseCompilerAnalyzer):
    EXT = ".java"
    BOILERPLATE_START = "public class Temp {\n    public static void main(String[] args) {"
    BOILERPLATE_END = "    }\n}"
    LANGUAGE_NAME = "Java"
    def compile_and_trace(self, raw_perm, filename_base):
        src_file = os.path.join(self.temp_dir, "Temp.java")
        class_file = os.path.join(self.temp_dir, "Temp.class")
        try:
            with open(src_file, "w") as f: f.write(self.wrap_code(raw_perm))
            cmd = ["javac", "-d", self.temp_dir, src_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(class_file):
                return {"success": False, "allocations": 0, "error": result.stderr[:200]}
            cmd_javap = ["javap", "-v", class_file]
            javap_res = subprocess.run(cmd_javap, capture_output=True, text=True)
            total_bytes, details = 0, []
            if javap_res.returncode == 0:
                matches = re.findall(r"max_locals\s+(\d+)", javap_res.stdout)
                for m in matches:
                    slots = int(m)
                    total_bytes += slots * 8
                    details.append(f"Method allocated {slots} local slots ({slots * 8} bytes)")
            return {"success": True, "allocations": total_bytes, "details": "\n".join(details), "error": None}
        finally:
            self.safe_delete(src_file, class_file)

# ==========================================
# 4. PARALLEL WORKER THREADS
# ==========================================

class ProducerThread(QThread):
    def __init__(self, charset, max_n, max_perms, task_queue, shared_state):
        super().__init__()
        self.charset = charset
        self.max_n = max_n
        self.max_perms = max_perms
        self.task_queue = task_queue
        self.shared_state = shared_state

    def run(self):
        processed = 0
        for length in range(1, self.max_n + 1):
            if self.shared_state.stop_event.is_set() or processed >= self.max_perms: break
            for perm in itertools.product(self.charset, repeat=length):
                if self.shared_state.stop_event.is_set() or processed >= self.max_perms: break
                self.task_queue.put("".join(perm))
                processed += 1
        self.shared_state.producer_done.set()

class CompilerWorker(QThread):
    worker_done = Signal()
    
    def __init__(self, task_queue, shared_state, analyzer_class, signals):
        super().__init__()
        self.task_queue = task_queue
        self.shared_state = shared_state
        self.analyzer_class = analyzer_class
        self.signals = signals
        self.temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def run(self):
        analyzer = self.analyzer_class(self.temp_dir)
        try:
            while not self.shared_state.stop_event.is_set():
                try:
                    code = self.task_queue.get(timeout=0.1) 
                except queue.Empty:
                    if self.shared_state.producer_done.is_set() and self.task_queue.empty(): break
                    continue

                result = analyzer.compile_and_trace(code, "temp_perm")
                
                with self.shared_state.lock:
                    self.shared_state.processed_count += 1
                    if result["success"]:
                        self.shared_state.success_count += 1
                        self.shared_state.allocation_samples.append(result["allocations"])
                        if result["allocations"] > self.shared_state.best_result["allocations"]:
                            self.shared_state.best_result = {
                                "allocations": result["allocations"],
                                "code": analyzer.wrap_code(code),
                                "details": result["details"],
                                "language": analyzer.LANGUAGE_NAME
                            }
                            self.shared_state.allocation_history.append(
                                (self.shared_state.processed_count, result["allocations"])
                            )
                            self.signals.log.emit(f"[+] New best! {result['allocations']} bytes. Code: {code[:30]}...")
                    
                    # Throttled UI updates
                    if self.shared_state.processed_count % 10 == 0:
                        elapsed = time.time() - self.shared_state.start_time
                        rate = self.shared_state.processed_count / max(elapsed, 0.001)
                        remaining = max(self.shared_state.total_perms - self.shared_state.processed_count, 0)
                        eta = remaining / max(rate, 0.001)
                        success_rate = (self.shared_state.success_count / self.shared_state.processed_count) * 100
                        
                        self.signals.stats_update.emit({
                            "processed": self.shared_state.processed_count,
                            "total": self.shared_state.total_perms,
                            "success_rate": success_rate,
                            "rate": rate,
                            "eta": eta,
                            "best": self.shared_state.best_result["allocations"]
                        })
                        
                        prog = int((self.shared_state.processed_count / self.shared_state.total_perms) * 100)
                        self.signals.progress.emit(prog)
                        
                        # Throttled graph updates
                        if self.shared_state.processed_count % 50 == 0:
                            self.signals.graph_update.emit(
                                list(self.shared_state.allocation_history),
                                list(self.shared_state.allocation_samples[-500:])
                            )
                
                self.task_queue.task_done()
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.worker_done.emit()

# ==========================================
# 5. LIVE GRAPH PANEL
# ==========================================

class GraphPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # Best Allocation Over Time
        self.plot1 = pg.PlotWidget(title="Best Allocation Over Time")
        self.plot1.setLabel('left', 'Bytes Allocated')
        self.plot1.setLabel('bottom', 'Permutations Tested')
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        self.plot1_curve = self.plot1.plot(pen=pg.mkPen('#4FC3F7', width=2))
        self.plot1.setMaximumHeight(250)
        
        # Allocation Distribution Histogram
        self.plot2 = pg.PlotWidget(title="Allocation Distribution")
        self.plot2.setLabel('left', 'Frequency')
        self.plot2.setLabel('bottom', 'Bytes Allocated')
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.plot2.setMaximumHeight(250)
        
        layout.addWidget(self.plot1)
        layout.addWidget(self.plot2)

    def update_graphs(self, history, samples):
        if history:
            x = [h[0] for h in history]
            y = [h[1] for h in history]
            self.plot1_curve.setData(x, y)
        
        if samples and len(samples) >= 2:
            # Create histogram bins
            arr = np.array(samples)
            hist, bins = np.histogram(arr, bins=20)
            self.plot2.clear()
            bg = pg.BarGraphItem(x=bins[:-1], height=hist, width=bins[1]-bins[0], 
                                 brush=pg.mkBrush('#81C784'))
            self.plot2.addItem(bg)

# ==========================================
# 6. STATS BAR WIDGET
# ==========================================

class StatsBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setStyleSheet("background-color: #252526; border-radius: 4px;")
        
        self.labels = {}
        for key, label in [("processed", "Tested: 0"), ("success", "Success: 0%"),
                           ("rate", "Rate: 0/s"), ("eta", "ETA: --"), ("best", "Best: 0 B")]:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #D4D4D4; font-weight: bold; padding: 4px 10px;")
            layout.addWidget(lbl)
            self.labels[key] = lbl
        
        layout.addStretch()

    def update_stats(self, stats):
        self.labels["processed"].setText(f"Tested: {stats['processed']:,} / {stats['total']:,}")
        self.labels["success"].setText(f"Success: {stats['success_rate']:.1f}%")
        self.labels["rate"].setText(f"Rate: {stats['rate']:.1f}/s")
        eta_str = f"{stats['eta']:.0f}s" if stats['eta'] < 3600 else f"{stats['eta']/60:.1f}m"
        self.labels["eta"].setText(f"ETA: {eta_str}")
        self.labels["best"].setText(f"Best: {stats['best']:,} B")

# ==========================================
# 7. MAIN WINDOW
# ==========================================

class MainWindow(QMainWindow):
    CHARSET_PRESETS = {
        "Custom": "",
        "Alphanumeric": "abcdefghijklmnopqrstuvwxyz0123456789",
        "ASCII Operators": "=+;{}()[]",
        "Python Vars": "a=1;b=2;x=y;",
        "Minimal C": "int x=0;",
        "Binary": "01",
        "Hex": "0123456789ABCDEF"
    }
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 Parallel Multi-Language Compiler Tracer")
        self.resize(1300, 850)
        self.workers = []
        self.producer = None
        self.shared_state = None
        self.active_workers = 0
        self.apply_dark_theme()
        self.init_ui()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1E1E1E; color: #D4D4D4; }
            QGroupBox { border: 1px solid #3E3E42; border-radius: 6px; margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4FC3F7; }
            QPushButton { background-color: #0E639C; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #1177BB; }
            QPushButton:disabled { background-color: #3E3E42; color: #6E6E6E; }
            QPushButton#stopBtn { background-color: #C72E2E; }
            QPushButton#stopBtn:hover { background-color: #E04040; }
            QLineEdit, QSpinBox, QComboBox { background-color: #2D2D30; border: 1px solid #3E3E42; padding: 6px; border-radius: 4px; color: #D4D4D4; }
            QTabWidget::pane { border: 1px solid #3E3E42; background-color: #1E1E1E; }
            QTabBar::tab { background-color: #2D2D30; color: #D4D4D4; padding: 8px 16px; border: 1px solid #3E3E42; border-bottom: none; }
            QTabBar::tab:selected { background-color: #1E1E1E; color: #4FC3F7; border-bottom: 2px solid #4FC3F7; }
            QTextEdit, QPlainTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #3E3E42; font-family: Consolas, monospace; }
            QProgressBar { border: 1px solid #3E3E42; border-radius: 4px; text-align: center; background-color: #2D2D30; }
            QProgressBar::chunk { background-color: #4FC3F7; }
            QLabel { color: #D4D4D4; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # --- Controls ---
        controls_group = QGroupBox("Configuration")
        grid = QGridLayout()
        
        grid.addWidget(QLabel("Language:"), 0, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["C", "C++", "Rust", "Go", "Java", "Python"])
        grid.addWidget(self.lang_combo, 0, 1)
        
        grid.addWidget(QLabel("Charset Preset:"), 0, 2)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.CHARSET_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        grid.addWidget(self.preset_combo, 0, 3)
        
        grid.addWidget(QLabel("Charset:"), 1, 0)
        self.charset_input = QLineEdit("a=1;b=2;x=y;")
        self.charset_input.textChanged.connect(self.update_charset_count)
        grid.addWidget(self.charset_input, 1, 1, 1, 3)
        
        grid.addWidget(QLabel("Max Len (n):"), 2, 0)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 4)
        self.n_spin.setValue(2)
        grid.addWidget(self.n_spin, 2, 1)
        
        grid.addWidget(QLabel("Max Perms:"), 2, 2)
        self.max_perms_spin = QSpinBox()
        self.max_perms_spin.setRange(100, 1000000)
        self.max_perms_spin.setValue(5000)
        self.max_perms_spin.setSingleStep(1000)
        grid.addWidget(self.max_perms_spin, 2, 3)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start Parallel Trace")
        self.start_btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_analysis)
        btn_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(self.export_btn)
        
        grid.addLayout(btn_layout, 3, 0, 1, 4)
        
        self.charset_count_label = QLabel("Chars: 0 | Permutations: 0")
        self.charset_count_label.setStyleSheet("color: #81C784;")
        grid.addWidget(self.charset_count_label, 4, 0, 1, 4)
        
        controls_group.setLayout(grid)
        main_layout.addWidget(controls_group)
        self.update_charset_count()

        # --- Stats Bar ---
        self.stats_bar = StatsBar()
        main_layout.addWidget(self.stats_bar)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # --- Tabbed Content ---
        self.tabs = QTabWidget()
        
        # Tab 1: Results
        results_tab = QWidget()
        results_layout = QHBoxLayout(results_tab)
        splitter = QSplitter(Qt.Horizontal)
        
        code_group = QGroupBox("Best Source Code")
        code_layout = QVBoxLayout(code_group)
        self.code_display = QPlainTextEdit()
        self.code_display.setReadOnly(True)
        self.code_display.setFont(QFont("Consolas", 10))
        self.code_highlighter = PythonHighlighter(self.code_display.document())
        code_layout.addWidget(self.code_display)
        splitter.addWidget(code_group)
        
        trace_group = QGroupBox("Mathematical Allocation Trace")
        trace_layout = QVBoxLayout(trace_group)
        self.trace_display = QPlainTextEdit()
        self.trace_display.setReadOnly(True)
        self.trace_display.setFont(QFont("Consolas", 10))
        trace_layout.addWidget(self.trace_display)
        splitter.addWidget(trace_group)
        
        splitter.setSizes([600, 600])
        results_layout.addWidget(splitter)
        self.tabs.addTab(results_tab, "📊 Results")
        
        # Tab 2: Live Graphs
        self.graph_panel = GraphPanel()
        self.tabs.addTab(self.graph_panel, "📈 Live Graphs")
        
        # Tab 3: Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setMaximumBlockCount(500)
        log_layout.addWidget(self.log_display)
        self.tabs.addTab(log_tab, "📝 Execution Log")
        
        main_layout.addWidget(self.tabs)

        # --- Status Bar ---
        self.statusBar().showMessage("Ready. Configure parameters and start tracing.")
        self.statusBar().setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")

    def on_preset_changed(self, preset):
        if preset in self.CHARSET_PRESETS:
            self.charset_input.setText(self.CHARSET_PRESETS[preset])

    def update_charset_count(self):
        charset = self.charset_input.text()
        n = self.n_spin.value()
        total = sum(len(charset)**i for i in range(1, n + 1))
        self.charset_count_label.setText(f"Chars: {len(charset)} | Total Permutations: {total:,}")

    def start_analysis(self):
        charset = self.charset_input.text()
        if not charset:
            self.log_display.appendPlainText("❌ Error: Character set cannot be empty.")
            return

        lang = self.lang_combo.currentText()
        compiler_map = {"C": "gcc", "C++": "g++", "Rust": "rustc", "Go": "go", "Java": "javac"}
        compiler_cmd = compiler_map[lang]
        if lang != "Python" and not shutil.which(compiler_cmd):
            self.log_display.appendPlainText(f"❌ Error: '{compiler_cmd}' not found in PATH.")
            return

        n = self.n_spin.value()
        max_perms = self.max_perms_spin.value()
        total_perms = sum(len(charset)**i for i in range(1, n + 1))
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_display.clear()
        self.trace_display.clear()
        self.code_display.clear()
        self.log_display.appendPlainText(f"🚀 Initializing {os.cpu_count()} parallel workers for {lang}...")
        self.statusBar().showMessage(f"Running {lang} analysis...")

        self.shared_state = SharedState()
        self.shared_state.total_perms = min(total_perms, max_perms)
        
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.progress_bar.setValue)
        self.signals.log.connect(self.log_display.appendPlainText)
        self.signals.stats_update.connect(self.stats_bar.update_stats)
        self.signals.graph_update.connect(self.graph_panel.update_graphs)

        self.task_queue = queue.Queue(maxsize=1000)
        
        analyzer_map = {"C": CAnalyzer, "C++": CppAnalyzer, "Rust": RustAnalyzer, 
                        "Go": GoAnalyzer, "Java": JavaAnalyzer, "Python": PythonAnalyzer}
        analyzer_class = analyzer_map[lang]

        self.producer = ProducerThread(charset, n, max_perms, self.task_queue, self.shared_state)
        self.producer.start()

        num_workers = min(os.cpu_count() or 4, 8)
        self.active_workers = num_workers
        self.workers = []
        
        for _ in range(num_workers):
            worker = CompilerWorker(self.task_queue, self.shared_state, analyzer_class, self.signals)
            worker.worker_done.connect(self.on_worker_done)
            self.workers.append(worker)
            worker.start()
        
        self.tabs.setCurrentIndex(1)  # Switch to graphs tab

    def stop_analysis(self):
        if self.shared_state:
            self.shared_state.stop_event.set()
            self.log_display.appendPlainText("\n⏹ Stopping all threads...")
            self.reset_ui()

    def on_worker_done(self):
        self.active_workers -= 1
        if self.active_workers <= 0:
            self.on_analysis_finished()

    def on_analysis_finished(self):
        self.reset_ui()
        result = self.shared_state.best_result
        
        self.trace_display.clear()
        self.trace_display.appendPlainText(f"✅ {result['language']} Parallel Trace Complete\n")
        
        if not result["code"]:
            self.trace_display.appendPlainText("❌ No valid variable allocations compiled.")
            self.code_display.setPlainText("No valid code compiled.")
            self.statusBar().showMessage("Analysis complete. No valid allocations found.")
            return

        self.code_display.setPlainText(result["code"])
        # Apply syntax highlighting based on language
        if result['language'] == "Python":
            self.code_highlighter = PythonHighlighter(self.code_display.document())
        else:
            self.code_highlighter = GenericHighlighter(self.code_display.document())
        
        self.trace_display.appendPlainText(f"📐 Total Mathematical Allocation: {result['allocations']:,} bytes\n")
        self.trace_display.appendPlainText("Compiler Trace Details:")
        self.trace_display.appendPlainText("-" * 40)
        self.trace_display.appendPlainText(result["details"])
        
        self.tabs.setCurrentIndex(0)  # Switch to results tab
        self.statusBar().showMessage(f"✅ Analysis complete. Best allocation: {result['allocations']:,} bytes")

    def export_results(self):
        if not self.shared_state or not self.shared_state.best_result["code"]:
            self.log_display.appendPlainText("❌ No results to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "trace_results.json", "JSON (*.json)")
        if path:
            data = {
                "language": self.shared_state.best_result["language"],
                "total_allocations": self.shared_state.best_result["allocations"],
                "code": self.shared_state.best_result["code"],
                "details": self.shared_state.best_result["details"],
                "processed": self.shared_state.processed_count,
                "success_rate": (self.shared_state.success_count / max(self.shared_state.processed_count, 1)) * 100,
                "allocation_history": self.shared_state.allocation_history
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.log_display.appendPlainText(f"💾 Results exported to {path}")

    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

if __name__ == "__main__":
    pg.setConfigOptions(antialias=True, background='#1E1E1E', foreground='#D4D4D4')
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())