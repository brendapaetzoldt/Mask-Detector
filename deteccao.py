import dlib
import glob
import cv2
import os

opcoes = dlib.simple_object_detector_training_options()
opcoes.add_left_right_image_flips = True
opcoes.C = 5

dlib.train_simple_object_detector("dlib-19.21/tools/imglab/build/dataset.xml", "dataset_mascaras.svm", opcoes)
dlib.train_simple_object_detector("dlib-19.21/tools/imglab/build/dataset1.xml", "dataset_mascaras1.svm", opcoes)


detector = dlib.simple_object_detector("dataset_mascaras.svm")
detector1 = dlib.simple_object_detector("dataset_mascaras1.svm")

for imagem in glob.glob(os.path.join("delirium", "*.jpg")):
    img = cv2.imread(imagem)
    objetosDetectados = detector(img)
    objetosDetectados = detector1(img)
    for d in objetosDetectados:
        e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (e,t), (d, b), (0,0,255), 2)

    cv2.imshow("Detector Delirium", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()