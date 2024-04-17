import cv2

face_ref = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
camera = cv2.VideoCapture(0)  # 0 adalah kamera bawaan


def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1)
    return faces


def drawer_box(frame):
    for x, y, width, height in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)


def close_window():
    camera.release()  # tutup kamera
    cv2.destroyAllWindows()  #  menghentikan eksekusi cv2
    exit()


def main():
    while True:
        _, frame = camera.read()  # ambil kamera system
        drawer_box(frame)
        cv2.imshow("Face Detection Dasar", frame)  # munculkan box camera

        if cv2.waitKey(1) & 0xFF == ord("q"):
            close_window()


if __name__ == "__main__":
    main()

