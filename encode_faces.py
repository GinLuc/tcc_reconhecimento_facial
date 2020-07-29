# importa os pacotes necessários
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# constrói o analisador (parser) de argumentos e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# pega os caminhos das imagens de entrada da pasta dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# inicializa a lista de codificações e nomes conhecidos
knownEncodings = []
knownNames = []


def encoded(fileName):
    if(fileName.startswith('encoded-')):
        return False
    else:
        return True

# passa pelos caminhos das imagens
for (i, imagePath) in enumerate(imagePaths):
    # extrai o nome da pessoa do caminho da imagem
    name = imagePath.split(os.path.sep)[-2]
    fileName = imagePath.split(os.path.sep)[-1]

    if(encoded(fileName)):
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))

        # carrega a imagem de entrada e a converte de BGR (ordenação do OpenCV)
        # para RGB (ordenação do dlib)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detecta as coordenadas (x, y) das caixas delimitadoras
        # correspondentes para cada face nas imagens de entrada
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])

        # processa as incorporações (embedding) faciais para a face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # passa pelas codificações (encodings)
        for encoding in encodings:
            # adiciona cada codificação + nome para nosso conjunto de
            # codificações e nomes conhecidos
            knownEncodings.append(encoding)
            knownNames.append(name)
            print(f"Name: {name}  -  Encoding: {encoding}")

        newFileName = 'encoded-' + fileName

        newImagePath = os.path.split(imagePath)[0] + '/' + newFileName

        os.rename(imagePath, newImagePath)

# grava as codificações faciais + nomes no disco
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

if(len(data) > 0):
    if(os.path.exists(args["encodings"])):
        with open(args["encodings"], "ab") as f:
            f.write(pickle.dumps(data))

    else:
        f = open(args["encodings"], "wb")
        f.write(pickle.dumps(data))
        f.close()
