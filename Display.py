import cv2
from Extract import Extract
#usbipd attach --busid 2-6 --wsl

class Display():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        cv2.namedWindow("Scanner")
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    def read(self):
        return self.video.read()

    def end(self):
        self.video.release()
        cv2.destroyAllWindows()

#initiates display
display = Display()

#handles processing
extract = Extract()

while True:
    ret, frame = display.read()
    if not ret:
        break

    face_positions = extract.get_face_positions(frame, draw=True)

    for (i, face) in enumerate(face_positions):
        startX, startY, endX, endY = face_positions[i]
        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        vec = extract.get_embedding(face)
        name = extract.predict_face(vec)

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text = "{}".format(name)
        if name != "Unknown":
            print(f"{name} recognized!")
        cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Scanner", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

display.end()
