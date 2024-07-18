import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils       #en boy oranını yenıden hesaplayan kütüphane

import numpy as np
from ultralytics import YOLO
from collections import defaultdict  #takip değişkeni için kütüphane

color = (0,255,0)
color_red = (0,0,255)
thickness = 2

font_scale = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

video_path = "inference/test.mp4"
model_path = "models/yolov8n.pt"

cap = cv2.VideoCapture(video_path)  #videoyu okuyup oynatma
model = YOLO(model_path)   #yolov8 in nano modelini yukarıda tanımlayıp burada çağırdım

width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')  #frameleri kaydetmek için fourcc koduna çevir
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))  #yapılan frame ve videoyu kaydet


vehicle_ids = [2, 3, 5, 7]   #nesne modelinin içerisindeki sadece bu indexlere sahip olan id leri takip et
track_history = defaultdict(lambda: [])   #takip geçmişi değişkeni

up = {}   #referans çizgi üstü
down = {}  #referans çizgi altı
threshold = 450

while True:    #görüntüyü okuyup işleme döngüsü
    ret, frame = cap.read()    #görüntüyü okuma
    if ret == False:     #görüntü hata verdiğnde döngüyü kır
        break

    frame = imutils.resize(frame, width=1280)
    results = model.track(frame, persist=True, verbose=False)[0]  #nesne tanıma modelinden nesneleri tanıyıp burada takip ettiriyoruz
    

    bboxes = np.array(results.boxes.data.tolist(), dtype="int") #xyxy float değerinde verilen kordinat değerlerini int tipine çevirme
    
    cv2.line(frame, (0, threshold), (1280, threshold), color_red, thickness)   #görüntü üzerine referans çizgisi tanımlama
    cv2.putText(frame, "Reference Line", (620, 445), font, 0.7, color_red, thickness)
    
    
    for box in bboxes:   # track id leri dolaşmak için döngü
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)   #kuyruk bırakmak için nesne merkezi hesaplama
        cy = int((y1+y2)/2)
        
        if class_id in vehicle_ids:   #modelin içindeki idlere eşitse aşağıdaki işlemleri yap
            class_name = results.names[int(class_id)].upper() # car --> CAR  #ulaşılan class_name çağırma büyük küçük harf uyumuna duyarlı

            
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 15:  #kuyruk uzunluk değeri
                track.pop(0)
            
            points = np.hstack(track).astype("int32").reshape((-1,1,2))  #alınan diziye tip dönüşümü yap
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)  #frame üzerine kuyruk çizimi
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)   #nesne etrafında takip alanı belirleme çizgisi
            
            text = "ID:{} {}".format(track_id, class_name)   #nesne framelerini tanıma
            cv2.putText(frame, text, (x1, y1-5), font, font_scale, color, thickness)  #tanılılan nesneyi tickness çizgisi üzerine yazdırma
            
            if cy>threshold-5 and cy<threshold+5 and cx<670:   #down hesaplanması
                down[track_id] = x1, y1, x2, y2
                
            if cy>threshold-5 and cy<threshold+5 and cx>670:   #up hesaplanması
                up[track_id] = x1, y1, x2, y2
                
        
        print("UP Dictionary Keys: ", list(up.keys()))    #terminalde giden araçları yazdır
        print("DOWN Dictionary Keys: ", list(down.keys()))  #terminalde gelen araçları yazdır
        
        up_text = "Giden:{}".format(len(list(up.keys())))    #görüntü işlenen ekrana giden araç sayısını yazdır
        down_text = "Gelen:{}".format(len(list(down.keys())))  #görüntü işlenen ekrana gelen araç sayısını yazdır
        
        cv2.putText(frame, up_text, (1150, threshold-5), font, 0.8, color_red, thickness)
        cv2.putText(frame, down_text, (0, threshold-5), font, 0.8, color_red, thickness)
    
    writer.write(frame)   #her bir döngüde frameleri sıkıştır
    cv2.imshow("Test", frame)
    if cv2.waitKey(10) & 0xFF==ord("q"):    #videoyu kapatmak için q tuşuna bağlı hex kodunu girdim
        break

cap.release()
writer.release() #serbest bırak videoyu
cv2.destroyAllWindows()

print("[INFO].. The video was successfully processed/saved !")