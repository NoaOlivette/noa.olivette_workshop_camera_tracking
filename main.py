import cv2
import mediapipe as mp
import pynput


def open_image():
    img = cv2.imread("nico.jpg", 0)
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(20) == ord("q"):
            break
    cv2.destroyAllWindows()


def open_camera():
    cam = cv2.VideoCapture(0)
    if not (cam.isOpened()):
        print("Unable to open camera\n")
    else:
        while True:
            ret, frame = cam.read()
            cv2.imshow('preview', frame)
            if cv2.waitKey(20) == ord("q"):
                break
    cam.release()
    cv2.destroyAllWindows()


def detect_hand():
    cam = cv2.VideoCapture(0)
    mp_hand = mp.solutions.hands
    other_hand = mp_hand.Hands()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("unable to open camera")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = other_hand.process(rgb)
        print(results.multi_hand_landmarks)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


def draw_plan():
    cam = cv2.VideoCapture(0)
    mpHand = mp.solutions.hands
    hands = mpHand.Hands()
    mpDraw = mp.solutions.drawing_utils
    while True:
        ret, frame = cam.read()
        if not ret:
            print("unable to open camera")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    h,w,c = frame.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    mouse = pynput.mouse.Controller
                    print(x, y, "\n")
                mpDraw.draw_landmarks(frame, handlms)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    draw_plan()
    # detect_hand()
    # open_image()
    # open_camera()
