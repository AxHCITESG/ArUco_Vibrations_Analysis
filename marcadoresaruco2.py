import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error cámara")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, id in enumerate(ids):
            cv2.putText(frame,
                        f"ID: {id[0]}",
                        tuple(corners[i][0][0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

    cv2.imshow("Vision ArUco", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
