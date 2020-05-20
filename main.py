import cv2 as cv
import dlib
from scipy.spatial import distance as dist
import numpy as np
from model import load_model

model = load_model()

def predict_letter(img,expand,c_w,c_h,model):
    characters = ['O','I','2','3','A','S','G','T','8','9',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    img_c = img[(c_h-expand):(c_h+expand),
            (c_w-expand):(c_w+expand),:]
    roi = img_c
    roi = cv.resize(roi, dsize=(28,28), interpolation=cv.INTER_CUBIC)
    roi = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
    
    roi = np.array(roi)
    t = np.copy(roi)
    t = t / 255.0
    t = 1-t
    t = t.reshape(1,784)
    #m.append(roi)
    pred = model.predict_classes(t)
    return characters[pred[0]]

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

#def sound_alarm(alarm_path):
#    # play an alarm sound
#    playsound.playsound(alarm_path)

#alarm_path = 'alarm.wav'
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 5
NORMAL_BLINK = 3
COUNTER = 0
def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def realtime_test(model):
    dat = 'models/shape_predictor_68_face_landmarks.dat'
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dat)
    
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS,29)
    frames = 0
    chars_img = []
    noses_points = []
    chars_char = []
    ALARM_ON = False
    COUNTER = 0
    expand = 64
    c_w = 640
    c_h = 360
    while True:
        ret, img = cap.read()
        cv.rectangle(img, (c_w-expand,c_h+expand),
                         (c_w+expand,c_h-expand),
                        (0,255,0),
                        4)
        
        img = cv.flip(img,1)
        frames+=1
        if frames==1:
            write_image = 255*np.ones(shape=img.shape, dtype=np.uint8)
        
        image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rects = detector(image_gray, 0)
        
        if len(rects)>0:
            for rect in rects:
                shape = predictor(image_gray, rect)
                shape = shape_to_np(shape)
        
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                
                ear = (leftEAR + rightEAR) / 2.0
                
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        
                        if COUNTER ==EYE_AR_CONSEC_FRAMES:
                            noses_point = []
                            chars_img.append(write_image)
                            try:
                                chars_char.append(predict_letter(write_image,expand,c_w,c_h,model))
                            except:
                                chars_char.append('-')
                            
                        
                        #if not ALARM_ON:
                        #    ALARM_ON = True
    #
                        #    if alarm_path != "":
                        #        t = Thread(target=sound_alarm,
                        #            args=(alarm_path,))
                        #        t.deamon = True
                        #        t.start()
                        ## draw an alarm on the frame
                        cv.putText(img, "Ready for new letter", (10, 30),
                            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                # 
                else:
                    if COUNTER>0 and COUNTER<NORMAL_BLINK:
                        #let the user blink normally
                        COUNTER = 0
                        ALARM_ON = False
                    elif COUNTER>0:
                        write_image = 255*np.ones(shape=img.shape, dtype=np.uint8)
                        COUNTER = 0
                        ALARM_ON = False
            
            nose = shape[30]
            noses_points.append(nose)
            
            if len(noses_points)==1:
                #print first nose point
                cv.circle(write_image, tuple(nose), 3, (0, 0, 0), 13)
            if len(noses_points)>1:
                cv.line(write_image, tuple(noses_points[-1]),tuple(noses_points[-2]),(0, 0, 0), 13)

        if len(chars_char)>0:
            cv.putText(img, ''.join(chars_char), (600, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        
        cv.imshow('Mirror', img)
        cv.imshow("Paper", write_image)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            return print(''.join(chars_char))


word = realtime_test(model)

              