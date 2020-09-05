import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import msvcrt

file_path_out = r'C:\Users\Jacob\CODE\verdatumai\ML\data\handwriting generation\croped labeled'
file_names=[file for _,_,files in os.walk(file_path_out) for file in files]
index = int(file_names[-1].split()[0])
print("index :", index)

file_path = r'C:\Users\Jacob\CODE\verdatumai\ML\data\handwriting generation\raw'
file_names=[file for _,_,files in os.walk(file_path) for file in files]

for file_name in file_names:
    print('File Loaded :',file_name)
    #load file
    src = cv2.imread(file_path+'\\'+file_name, cv2.IMREAD_GRAYSCALE)
    #blur the picture
    dst = cv2.GaussianBlur(src,(21,21),cv2.BORDER_DEFAULT)
    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(dst,215,255,0)
    #Negative
    binary_map = 255 - binary_map

    reshaped = cv2.resize(binary_map, (400, 400))
    
    # connected components processing
    nlabels, labels = cv2.connectedComponents(binary_map, 8, cv2.CV_32S)
    print("No of CC extracted :",np.unique(labels)[-1]+1)
    size = labels.shape

    labels = labels.reshape(-1)
    for cluster in np.unique(labels):
        x_min = 100000000
        y_min = 100000000
        x_max = -1
        y_max = -1 
        #print(len(labels.reshape(-1)))
        for i in range(len(labels)):
            if labels[i] == cluster :
                if i/size[1] < y_min :
                    y_min = i/size[1]
                if i%size[1] < x_min :
                    x_min = i%size[1]
                if i/size[1] > y_max :
                    y_max = i/size[1]
                if i%size[1] > x_max :
                    x_max = i%size[1]
        #Display
        ratio = 64/(y_max-y_min)
        print(int(y_min)-int(y_max), int(x_min)-int(x_max))
        print(64, int(ratio * (x_max - x_min)))
        reshaped = cv2.resize(src[int(y_min):int(y_max), int(x_min):int(x_max)], (int(ratio * (x_max - x_min)), 64))
        cv2.imshow('window',reshaped)
        
        print("Should this be Labeled? (Y): ")
        cv2.waitKey(50)
        choice = msvcrt.getch().decode("utf-8").lower()
        print(choice)
        if choice == 'y':
            index += 1
            name = input("Label : ")
            new_file_name = file_path_out+"\\"+str(index)+" "+name+".jpg"
            cv2.imwrite(new_file_name, reshaped)
            print('inserted')
        else :
            print("not inserted")
        cv2.destroyAllWindows()