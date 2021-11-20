import cv2 as cv
from numpy.lib.function_base import place
from skimage import io

from chatbot import ChatBot

def show_image(image_url):
    image = io.imread(image_url)
    while True:
        cv.imshow('Demo - Pulsa Q para salir', image) #...Imprime el frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break #...............SALE DEL BUCLE

USER_ID = 2
AVT = ChatBot("Hola", USER_ID)

print(AVT)
print("Start!")

while True:
    msg = input("...").lower()
    AVT.message = msg
    many_candidates = AVT.set_message()
    print("---", AVT.intencion)
    # AVT.select_candidate(place_candidates)
    if many_candidates:
        print("Quiero asegurarme...?")
        for i in range(len(AVT.place_candidates)):
            print("%s: %s"%(i,AVT.place_candidates[i]))
        index = int(input(">> Ingresa el número"))
        AVT.selec_from_candidates(index)
    else:
        AVT.confirm_candidate()
    AVT.save_context(USER_ID)
    AVT.create_response(USER_ID)
    AVT.select_response()
    print(">>", AVT.res)
    if AVT.show_image:
        print("* URL de la imagen:", AVT.get_url_image())
    if AVT.intencion == "despedida":
        break

if len(AVT.img_attachments):
    url_img = AVT.get_url_image()
    show_image(url_img)