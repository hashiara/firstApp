import cv2
import mediapipe as mp
import math
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 線の太さを設定
# drawing_spec = mp_drawing.DrawingSpec(thickness = 10, circle_radius = 1)

# 赤色の描画スタイルを設定
red_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=1)

# カウンターの初期値と閾値を設定
counter = 0
threshold_distance = 0.3  # この値を調整して閾値を設定

# 時刻を保持する変数を追加
last_increment_time = time.time()

# 使用カメラ番号　カメラソフト:0, PC:1, Webカメラ:2
cap = cv2.VideoCapture(1)

# 動画ファイルの設定
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # ビデオコーデックの設定
out = cv2.VideoWriter('kintore.avi', fourcc, fps, (width, height))  # ファイル名、コーデック、フレームレート、解像度を指定
# フォントの設定
font = cv2.FONT_HERSHEY_SIMPLEX  # フォント設定
font_scale = 1  # フォントスケール
font_color = (255, 255, 255)  # フォントカラー
font_thickness = 2  # フォント太さ

with mp_pose.Pose(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as pose:
    # while cap.isOpened():
    #     success, image = cap.read()
    #     if not success:
    #         print("Ignoring empty camera frame.")
    #         # If loading a video, use 'break' instead of 'continue'.
    #         continue
    while True:
        ret, image = cap.read()  # フレームをキャプチャ
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 各ランドマークを赤色で描画
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=red_drawing_spec)
        
        # 手首と肩の関節の距離を計算
        if results.pose_landmarks.landmark and len(results.pose_landmarks.landmark) >= 12 and len(results.pose_landmarks.landmark) >= 11:
            wrist_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            shoulder_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # 手首と肩の関節の距離を計算
            distance = math.sqrt((wrist_landmark.x - shoulder_landmark.x) ** 2 + (wrist_landmark.y - shoulder_landmark.y) ** 2)
            
            # 閾値未満の場合、カウンターをインクリメント
            if distance < threshold_distance:
                # 前回のインクリメントから1.5秒以上経過している場合にのみインクリメント
                if time.time() - last_increment_time >= 1.5:
                    counter += 1
                    last_increment_time = time.time()  # インクリメント時刻を更新
                    # print("カウント: ", counter)
        
        # 画像上にカウンターの値を表示
        cv2.putText(image, f'Count: {counter}', (20, 50), font, font_scale, font_color, font_thickness)
        cv2.imshow('MediaPipe Pose', image)

        out.write(image)  # フレームを動画に書き込む

        # ctrl and c to close a window
        if cv2.waitKey(5) & 0xFF == 27:
            break

# キャプチャと書き込みを解放
cap.release()
out.release()

# OpenCV のウィンドウを全て閉じる
cv2.destroyAllWindows()
