import sys
import dlib
import cv2

pula_quadros = 30
captura = cv2.VideoCapture(0)
contadorQuadros = 0

detector = dlib.simple_object_detector("dataset_mascaras1.svm")


while captura.isOpened():
    conectado, frame = captura.read()
    contadorQuadros += 1
    if contadorQuadros % pula_quadros == 0:
        objetosDetectados = detector(frame, 1)
        for o in objetosDetectados:
            e, t, d, f = (int(o.left()), int(o.top()), int(o.right()), int(o.bottom()))
            cv2.rectangle(frame, (e, t), (d, f), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Mascara nao usada corretamente", (0, 130), font, 1, (300, 250), 2, cv2.LINE_AA)

        cv2.imshow("Preditor de Objetos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

captura.release()
cv2.destroyAllWindows()
sys.exit(0)