import cv2
import mediapipe as mp
import math

cam = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        left_eye_x, left_eye_y = int(landmarks[263].x * frame_w), int(landmarks[263].y * frame_h)
        right_eye_x, right_eye_y = int(landmarks[130].x * frame_w), int(landmarks[130].y * frame_h)

        eye_center_x, eye_center_y = (left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2

        cv2.circle(frame, (left_eye_x, left_eye_y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye_x, right_eye_y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (eye_center_x, eye_center_y), 3, (0, 0, 255), -1)

        # calcula a direção do olhar
        if eye_center_x < frame_w // 2 and eye_center_y < frame_h // 2:
            cv2.putText(frame, 'Superior esquerda', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif eye_center_x >= frame_w // 2 and eye_center_y < frame_h // 2:
            cv2.putText(frame, 'Superior direita', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif eye_center_x < frame_w // 2 and eye_center_y >= frame_h // 2:
            cv2.putText(frame, 'Inferior esquerda', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Inferior direita', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("WebCam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
