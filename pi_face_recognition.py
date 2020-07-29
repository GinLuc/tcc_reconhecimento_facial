# importa os pacotes necessários
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# constrói o analisador (parser) de argumentos e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="caminho no qual path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# carrega as faces conhecidas e incorporações (embeddings) junto
# da detecção de faces do OpenCV Haarcascade
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(args["encodings"], "rb"). read())
detector = cv2.CascadeClassifier(cv2.data.haarcascades + args["cascade"])

# inicializa o stream de vídeo e dá tempo para a câmera aquecer
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# inicia o contador de quadros por segundo (FPS)
fps = FPS().start()

# passa pelos frames do stream de vídeo
while True:
    # pega os frames do stream de vídeo e os redimensiona
    # para 500px (para acelerar o processamento)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # converte o frame de entrada de (1) BGR para escala de cinza (para
    # a detecção de faces) e (2) de BGR para RGB (para o reconhecimento de faces)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detecta as faces no frame em escala de cinza
    rects = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # o OpenCV retorna coordenadas da caixa delimitadora na ordem 
    # (x, y, w, h) mas precisamos delas na ordem (superior, direita,
    # inferior, esquerda), então é necessário fazer uma reordenação
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # processa as incorporações faciais para cada caixa delimitadora de face
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # passa pelas incorporações faciais
    for encoding in encodings:
        # compara cada face na imagem de entrada com nossas codificações (encodings)
        # conhecidas para encontrar correspondências
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"

        # verifica se alguma correspondência foi encontrada
        if True in matches:
            # encontra as indexações de todas as faces correspondentes 
            # e inicia um dicionário (dictionary do python) para contar 
            # o número total de vezes que cada face foi correspondida
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # passa pelas indexações correspondidas e mantém um
            # contador para cada face reconhecida
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determina a face reconhecida com o maior número de votos 
            # (nota: no improvável evento de um empate o Python 
            # escolherá o primeiro valor no dicionário)
            name = max(counts, key=counts.get)

        # atualiza a lista de nomes
        names.append(name)

    # passa pelas faces reconhecidas
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # escreve o nome da face prevista na imagem
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    # mostra a imagem em nossa tela
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # se a tecla `q` for pressionada, sai do laço (loop)
    if key == ord("q"):
        break

    # atualiza o contador de quadros por segundo (FPS)
    fps.update()

# para o temporizador e mostra informações sobre a taxa
# de quadros por segundo
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# faz uma limpeza
cv2.destroyAllWindows()
vs.stop()
