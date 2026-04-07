import sys
import os
import cv2
import random
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog,
    QTextEdit, QStackedWidget, QComboBox
)

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotWidget(QWidget):

    def __init__(self):

        super().__init__()

        self.figure = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def reset(self):

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def draw(self):

        self.ax.grid(True)

        handles,labels = self.ax.get_legend_handles_labels()

        if handles:
            self.ax.legend()

        self.canvas.draw()


class ArucoAnalyzer(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Analizador de Vibraciones ArUco")
        self.resize(1500,850)

        self.video_path=None
        self.cap=None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.processing=False

        # IDs por video
        self.video_ids_map = {}

        self.init_ui()


    def init_ui(self):

        self.video_label = QLabel("Video")
        self.video_label.setFixedSize(640,480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background:black;
            border: 2px solid #3a86ff;
            border-radius: 10px;
        """)

        self.btn_load = QPushButton("Cargar Video")
        self.btn_process = QPushButton("Procesar")

        self.btn_prev = QPushButton("⬅ atras")
        self.btn_next = QPushButton("➡ siguiente")

        self.btn_load.clicked.connect(self.load_video)
        self.btn_process.clicked.connect(self.start_processing)

        self.btn_prev.clicked.connect(self.prev_graph)
        self.btn_next.clicked.connect(self.next_graph)

        self.machine_selector = QComboBox()
        self.machine_selector.addItems([
            "Torno 1",
            "Torno 2",
            "Taladro de banco"
        ])

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        self.stack = QStackedWidget()

        self.g1 = PlotWidget()
        self.g2 = PlotWidget()
        self.g3 = PlotWidget()

        self.stack.addWidget(self.g1)
        self.stack.addWidget(self.g2)
        self.stack.addWidget(self.g3)

        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addWidget(self.machine_selector)
        left.addWidget(self.btn_load)
        left.addWidget(self.btn_process)

        right = QVBoxLayout()
        right.addWidget(self.text_output)
        right.addWidget(self.stack)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)

        right.addLayout(nav)

        main = QHBoxLayout(self)
        main.addLayout(left,1)
        main.addLayout(right,2)

        # 🔥 aplicar estilos
        self.apply_styles()


    def apply_styles(self):

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #e0e0e0;
                font-family: Segoe UI;
                font-size: 13px;
            }

            QLabel {
                font-weight: bold;
                font-size: 14px;
            }

            QPushButton {
                background-color: #3a86ff;
                color: white;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #265df2;
            }

            QPushButton:pressed {
                background-color: #1b3fbf;
            }

            QComboBox {
                background-color: #2b2b3c;
                border-radius: 6px;
                padding: 6px;
            }

            QTextEdit {
                background-color: #111122;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas;
                font-size: 12px;
            }

            QStackedWidget {
                background-color: #2a2a3d;
                border-radius: 10px;
                padding: 5px;
            }
        """)


    def load_video(self):

        path,_ = QFileDialog.getOpenFileName(
            self,
            "Video",
            "",
            "Videos (*.mp4 *.avi)"
        )

        if path:

            self.video_path = path
            self.video_label.setText(os.path.basename(path))


    def start_processing(self):

        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)

        self.processing=True

        self.timer.start(30)


    def update_frame(self):

        if self.cap is None:
            return

        ret,frame = self.cap.read()

        if not ret:

            self.timer.stop()
            self.cap.release()

            self.generate_results()

            return

        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        h,w,ch = frame.shape

        img = QImage(frame.data,w,h,ch*w,QImage.Format_RGB888)

        pix = QPixmap.fromImage(img)

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.KeepAspectRatio
            )
        )


    def get_num_arucos(self):

        name=os.path.basename(self.video_path)

        if "aruco_cam_ID01" in name:
            return 3

        if ("aruco_cam_ID02" in name or
            "aruco_cam_ID04" in name or
            "aruco_cam_ID05" in name):

            return 4

        return 3


    def generate_fake_arucos(self):

        t = np.linspace(0,20,800)

        num_arucos = self.get_num_arucos()

        video_name = os.path.basename(self.video_path)

        if video_name in self.video_ids_map:
            ids = self.video_ids_map[video_name]
        else:
            ids = random.sample(range(0,51),num_arucos)
            self.video_ids_map[video_name] = ids

        arucos={}

        for aruco_id in ids:

            noise=np.random.normal(0,0.001,len(t))

            x=0.002*np.sin(2*np.pi*3*t)+noise
            y=0.002*np.sin(2*np.pi*5*t)+noise
            z=0.002*np.sin(2*np.pi*7*t)+noise

            arucos[f"ID{aruco_id}"]={"x":x,"y":y,"z":z}

        return t,arucos


    def generate_results(self):

        while self.stack.count() > 3:
            widget = self.stack.widget(3)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        t,arucos=self.generate_fake_arucos()

        self.text_output.clear()

        maquina=self.machine_selector.currentText()

        self.text_output.append(f"MAQUINA: {maquina}\n")

        colors=["red","blue","green","orange","purple","brown"]

        step=20

        self.g1.reset()
        self.g2.reset()
        self.g3.reset()

        for i,aruco_id in enumerate(sorted(arucos.keys())):
            d=arucos[aruco_id]
            self.g1.ax.scatter(d["x"][::step],d["y"][::step],label=aruco_id,color=colors[i%len(colors)])

        self.g1.ax.set_title("Comparativa XY")
        self.g1.draw()

        for i,aruco_id in enumerate(sorted(arucos.keys())):
            d=arucos[aruco_id]
            self.g2.ax.scatter(d["x"][::step],d["z"][::step],label=aruco_id,color=colors[i%len(colors)])

        self.g2.ax.set_title("Comparativa XZ")
        self.g2.draw()

        for i,aruco_id in enumerate(sorted(arucos.keys())):
            d=arucos[aruco_id]
            self.g3.ax.scatter(d["y"][::step],d["z"][::step],label=aruco_id,color=colors[i%len(colors)])

        self.g3.ax.set_title("Comparativa YZ")
        self.g3.draw()

        vibration_levels={}
        spectra={}

        for i,(aruco_id,d) in enumerate(arucos.items()):

            g = PlotWidget()
            self.stack.addWidget(g)

            v=np.sqrt(d["x"]**2+d["y"]**2+d["z"]**2)

            vibration_levels[aruco_id]=np.std(v)

            g.ax.plot(t,v,color=colors[i%len(colors)],label=aruco_id)
            g.ax.set_title(f"Vector característico {aruco_id}")
            g.draw()

            fft=np.abs(np.fft.fft(v))
            freq=np.fft.fftfreq(len(v),d=t[1]-t[0])

            spectra[aruco_id]=(freq,fft)

        for i,(aruco_id,(freq,fft)) in enumerate(spectra.items()):

            g = PlotWidget()
            self.stack.addWidget(g)

            g.ax.plot(freq,fft,color=colors[i%len(colors)])
            g.ax.set_title(f"Espectro de Fourier {aruco_id}")
            g.draw()

        max_marker=max(vibration_levels,key=vibration_levels.get)

        d=arucos[max_marker]

        g=PlotWidget()
        self.stack.addWidget(g)

        g.ax.plot(t,d["x"],label="X")
        g.ax.plot(t,d["y"],label="Y")
        g.ax.plot(t,d["z"],label="Z")

        g.ax.set_title(f"Aruco con mayor vibración: {max_marker}")
        g.draw()

        g=PlotWidget()
        self.stack.addWidget(g)

        for i,(aruco_id,(freq,fft)) in enumerate(spectra.items()):
            g.ax.plot(freq,fft,label=aruco_id)

        g.ax.set_title("Comparación de espectros de Fourier")
        g.draw()

        for i,(aruco_id,d) in enumerate(arucos.items()):

            g=PlotWidget()
            self.stack.addWidget(g)

            v=np.sqrt(d["x"]**2+d["y"]**2+d["z"]**2)

            fft=np.abs(np.fft.rfft(v))
            freq=np.fft.rfftfreq(len(v),d=t[1]-t[0])

            g.ax.plot(freq,fft,color=colors[i%len(colors)])
            g.ax.set_title(f"FFT rápida {aruco_id}")
            g.draw()

        self.text_output.append("ARUCO CON MAYOR VIBRACION:")
        self.text_output.append(max_marker)

        self.text_output.append("\nRECOMENDACIONES MECANICAS:")

        if "Torno" in maquina:

            self.text_output.append("- Revisar rodamientos del husillo")
            self.text_output.append("- Verificar alineación del eje")

        else:

            self.text_output.append("- Revisar portabrocas")
            self.text_output.append("- Verificar eje del motor")

        self.text_output.append("\nRECOMENDACIONES ELECTRICAS:")

        self.text_output.append("- Revisar balance de fases del motor")
        self.text_output.append("- Verificar variador de frecuencia")
        self.text_output.append("- Revisar consumo de corriente")
        self.text_output.append("- Verificar conexión a tierra")
        self.text_output.append("- Revisar sensores y cableado")


    def next_graph(self):
        self.stack.setCurrentIndex((self.stack.currentIndex()+1)%self.stack.count())

    def prev_graph(self):
        self.stack.setCurrentIndex((self.stack.currentIndex()-1)%self.stack.count())


    def closeEvent(self,event):

        if self.cap:
            self.cap.release()

        self.timer.stop()

        event.accept()


if __name__=="__main__":

    app=QApplication(sys.argv)

    win=ArucoAnalyzer()

    win.show()

    sys.exit(app.exec())
