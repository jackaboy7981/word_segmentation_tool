import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import msvcrt

#file path to save the croped images
file_path_out = r'C:\Users\Jacob\CODE\verdatumai\ML\data\handwriting generation\croped labeled' 
file_names=[file for _,_,files in os.walk(file_path_out) for file in files]
if file_names == []:
    index = 100000
else :
    index = int(file_names[-1].split()[0])
    print("index :", index)
print("Index :",index)
#file path to get images
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

        #Final Reshape
        ratio = 64/(y_max-y_min)
        reshaped = cv2.resize(src[int(y_min):int(y_max), int(x_min):int(x_max)], (int(ratio * (x_max - x_min)), 64))
        #Display
        cv2.imshow('window',reshaped)
        
        print("Should this be Labeled? (Y): ")
        cv2.waitKey(50)
        choice = msvcrt.getch().decode("utf-8").lower()
        print(choice)
        if choice == 'y':
            #index += 1
            name = input("Label : ")
            while name == '':
                print("Label should be entered !!")
                name = input("Label : ")

            reshape_choice = input('reshape :')
            try:
                reshape_choice = int(reshape_choice)
            except ValueError:
                reshape_choice = 0
            if reshape_choice == 1:
                temp_reshaped = src[int(y_min):int(y_max), int(x_min):int(x_max)]
                mid = int((y_max-y_min)/2)
                currnet_mid = 0
                max = 0
                for i in range(int(y_max-y_min)):
                    if temp_reshaped[i].sum() > max:
                        max = temp_reshaped[i].sum()
                        current_mid = i
                offset = currnet_mid - mid
                print(offset)
                if offset > 0 :
                    if y_min - offset < -1 :
                        y_min -= offset
                else :
                    if y_max - offset < size[0] :
                        y_max -= offset
            elif reshape_choice == 2 :
                offset = y_max - y_min
                y_min = y_min - offset 
                y_max = y_max + offset
            elif reshape_choice == 3 :
                offset = (y_max - y_min)/2
                y_max = y_max + offset
            
            ratio = 64/(y_max-y_min)
            reshaped = cv2.resize(src[int(y_min):int(y_max), int(x_min):int(x_max)], (int(ratio * (x_max - x_min)), 64))
        
            cv2.destroyAllWindows()
            cv2.imshow('window',reshaped)
            cv2.waitKey(50)
            g = msvcrt.getch()
            new_file_name = file_path_out+"\\"+str(index)+" "+name+".jpg"
            cv2.imwrite(new_file_name, reshaped)
            print('inserted')
        else :
            print("not inserted")
        cv2.destroyAllWindows()