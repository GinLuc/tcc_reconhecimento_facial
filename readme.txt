Crie uma pasta chamada 'dataset' no mesma pasta dos arquivos encode_faces.py e pi_face_recognition.py 
contendo pastas com os nomes das pessoas e fotos dentro das respectivas pastas.


Para realizar o processamento das imagens e gerar o arquivo de encodings:

	python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog


Para reconhecer os rostos:

	python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

