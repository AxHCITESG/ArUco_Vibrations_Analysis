import sys
import os
import cv2
import numpy as np
import math

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap


class ArucoCameraUI(QWidget):
    def __init__(self):
        super().__init__()

        # TAMAÑO EXACTO CÁMARA 
        self.cam_width = 640
        self.cam_height = 480

        self.setWindowTitle("Visión ArUco | Coordenadas Reales")
        self.setFixedSize(self.cam_width + 40, self.cam_height + 120)

        # VIDEO =
        self.video_label = QLabel("Cámara apagada")
        self.video_label.setFixedSize(self.cam_width, self.cam_height)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                color: #AAAAAA;
                font-size: 18px;
                border: 2px solid #2ecc71;
            }
        """)
        
        # ===== BOTONES =====
        self.btn_start = QPushButton("▶ Activar")
        self.btn_stop = QPushButton("⏹ Detener")
        self.btn_record = QPushButton("⏺ Grabar")
        self.btn_exit = QPushButton("❌ Cerrar")

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_exit.clicked.connect(self.close_app)

        # ===== ESTILO BOTONES =====
        btn_style = """
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 8px;
                color: black;
            }
        """

        self.btn_start.setStyleSheet(btn_style + "background-color:  green ")
        self.btn_stop.setStyleSheet(btn_style + "background-color: red")
        self.btn_record.setStyleSheet(btn_style + "background-color: black;")
        self.btn_exit.setStyleSheet(btn_style + "background-color:  black;")

        # ===== LAYOUT =====
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_stop)
        buttons_layout.addWidget(self.btn_record)
        buttons_layout.addWidget(self.btn_exit)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

        # ===== TIMER =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ===== ESTADOS =====
        self.cap = None
        self.recording = False
        self.video_writer = None

        # ===== RUTA GRABACIÓN =====
        self.save_path = r"C:\Users\David Padilla\Documents\Archivos de estadias python\grabaciones_aruco"
        os.makedirs(self.save_path, exist_ok=True)
        self.video_id = 0

        # ===== ARUCO CONFIG =====
        self.marker_length = 0.145  

        self.cameraMatrix = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1]
        ], dtype=np.float32)

        self.distCoeffs = np.zeros((5, 1))

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_50
        )
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, cv2.aruco.DetectorParameters()
        )

        self.obj_points = np.array([
            [-self.marker_length/2,  self.marker_length/2, 0],
            [ self.marker_length/2,  self.marker_length/2, 0],
            [ self.marker_length/2, -self.marker_length/2, 0],
            [-self.marker_length/2, -self.marker_length/2, 0]
        ], dtype=np.float32)

    # ===============================
    # FUNCIONES
    # ===============================

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
            self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.recording:
            self.toggle_record()

        self.video_label.setText("Cámara apagada")

    def toggle_record(self):
        if not self.recording:
            filename = f"aruco_cam_ID{self.video_id:02d}.avi"
            full_path = os.path.join(self.save_path, filename)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                full_path, fourcc, 20.0,
                (self.cam_width, self.cam_height)
            )

            self.recording = True
            self.btn_record.setText("⏹ Stop")

        else:
            self.recording = False
            self.video_writer.release()
            self.btn_record.setText("⏺ Grabar")
            self.video_id += 1

    def close_app(self):
        self.stop_camera()
        self.close()

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                img_points = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    img_points,
                    self.cameraMatrix,
                    self.distCoeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    cv2.drawFrameAxes(
                        frame,
                        self.cameraMatrix,
                        self.distCoeffs,
                        rvec,
                        tvec,
                        self.marker_length
                    )

                    x, y, z = tvec.flatten()
                    d = math.sqrt(x*x + y*y + z*z)

                    c = img_points[0]
                    cv2.putText(
                        frame,
                        f"X:{x:.3f}m  Y:{y:.3f}m  Z:{z:.3f}m",
                        (int(c[0]), int(c[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"D:{d:.3f} m",
                        (int(c[0]), int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 0),
                        2
                    )

        if self.recording:
            self.video_writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(
            rgb.data,
            self.cam_width,
            self.cam_height,
            self.cam_width * 3,
            QImage.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(img))


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ArucoCameraUI()
    window.show()
    sys.exit(app.exec())
