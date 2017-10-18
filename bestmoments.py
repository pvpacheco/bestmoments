from scipy import misc
import numpy as np
import pandas as pd
import cv2
import sys, getopt

def main(argv):
    # Getting arguments
    inputFile = ''
    inputSharpness = 0

    try:
        opts, args = getopt.getopt(argv,"hi:ms:",["input-file=","min-sharpness="])
    except getopt.GetoptError:
        print ('bestmoments.py -i <input-file> -ms <min-sharpness>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('bestmoments.py -i <input-file> -ms <min-sharpness>')
            sys.exit()
        elif opt in ("-i", "--input-file"):
            inputFile = arg
        elif opt in ("-ms", "--min-sharpness"):
            inputSharpness = arg

    hcFaces = "haarcascades/haarcascade_frontalface_default.xml"
    hcSmiles = "haarcascades/haarcascade_smile.xml"

    faceCascade = cv2.CascadeClassifier(hcFaces)
    smileCascade = cv2.CascadeClassifier(hcSmiles)

    cap = cv2.VideoCapture(inputFile)

    if not (cap.isOpened()):
        sys.exit()

    #cap.set(cv2.CAP_PROP_POS_FRAMES, 800)

    bestMoments = []
    lastFrameResult = 0
    frameCount = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        if (ret == False):
            break

        # Skipping frames after some result is found
        frameCount += 1
        if frameCount < (lastFrameResult + 24):
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.15,
            minNeighbors = 10,
            minSize = (100, 100),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # Highlighting faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roiGray = gray[y:y+h, x:x+w]
            roiGolor = frame[y:y+h, x:x+w]


        if(len(faces) > 0):
            smile = smileCascade.detectMultiScale(
                roiGray,
                scaleFactor = 1.15,
                minNeighbors = 18,
                minSize = (15, 15),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            # Highlighting smiles
            for (x, y, w, h) in smile:
                gy, gx = np.gradient(gray)
                gnorm = np.sqrt(gx**2 + gy**2)
                sharpness = np.average(gnorm)

                cv2.rectangle(roiGolor, (x, y), (x+w, y+h), (0, 255, 0), 1)

            if(len(smile) > 0):

                # Resizing
                r = 640.0 / frame.shape[1]
                dim = (640, int(frame.shape[0] * r))
                resizedFrame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

                # Getting frame sharpness
                gy, gx = np.gradient(gray)
                gnorm = np.sqrt(gx**2 + gy**2)
                sharpness = np.average(gnorm)

                # If sharpness is bellow 50, just ignore it
                if(sharpness > inputSharpness):
                    bestMoments.append([resizedFrame, len(smile), sharpness])
                    lastFrameResult = frameCount


        # Resizing
        r = 640.0 / frame.shape[1]
        dim = (640, int(frame.shape[0] * r))
        resizedFrame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # Output
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.imshow('Output', resizedFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if (bestMoments and len(bestMoments) > 0):

        # Order by number of smiles and image sharpness
        df = pd.DataFrame(bestMoments, columns=['resized_frame', 'smiles', 'sharpness'])
        df.sort_values(['smiles', 'sharpness'], ascending=False)

        if (len(df) == 1):
            message = "Your happy moment worth of sharing is:"
        else:
            message = "Your top happy moments worth of sharing are:"

        # Limiting results to top 3 only
        quantity = len(df)
        if(quantity > 5):
            quantity = 5;

        print(message)

        for i in range(0, quantity):
            cv2.imwrite("results/moment_%d.jpg" % (i+1), df.at[i, 'resized_frame'])
            print("results/moment_%d.jpg" % (i+1))


    else:
        message = "Sorry, we didnt't found any highlights on this video file. Can you try another one?"
        print(message)

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
   main(sys.argv[1:])
