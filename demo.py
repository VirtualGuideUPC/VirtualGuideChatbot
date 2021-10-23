import cv2 as cv
from skimage import io

from chatbot import ChatBot

def show_image(image_url):
    image = io.imread(image_url)
    while True:
        cv.imshow('Demo - Pulsa Q para salir', image) #...Imprime el frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break #...............SALE DEL BUCLE

AVT = ChatBot("Hola", " ")

print(AVT)
print("Start!")

while True:
    msg = input("...").lower()
    AVT.message = msg
    place_candidates = AVT.set_message()
    #print("Debug: >> candidates: ", place_candidates)
    AVT.select_candidate(place_candidates)
    AVT.create_response()
    AVT.select_response()
    print(">>", AVT.res)
    if AVT.show_image:
        print("* URL de la imagen:", AVT.get_url_image())
    if AVT.intencion == "despedida":
        break

if len(AVT.img_attachments):
    url_img = AVT.get_url_image()
    show_image(url_img)