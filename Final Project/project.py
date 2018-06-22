import cv2
import numpy as np
import os
import matplotlib.pyplot

IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "output"



class MHIKnn():
    def __init__(self,train_data, y):
        self.classifier = cv2.KNearest()
        self.train_data = train_data
        self.y = y

    #leave one out
    def leave_one_out(self):
        #train the model and predict the test y
        cnfmt = np.zeros((6,6))
        for i in range(len(self.train_data)):
            j = range(0, i) + range(i+1, self.train_data.shape[0])
            # train dataset
            Xt = self.train_data[j]
            yt = self.y[j]
            # train knn classifier
            self.classifier.train(Xt, yt)

            Xte = np.array([self.train_data[i]])
            yte = np.array([self.y[i]])
            # predict test data labels, res is an (1,1) array, but only 1 element, so extract it
            retval, res, neighborResponses, dists = self.classifier.find_nearest(Xte, 1)
            cnfmt[yte-1, int(res[0])-1] += 1
        pltcnf(cnfmt, actions=['boxing','handclapping','handwaving','jogging','running','walking'])

    def train(self):
        #train the model and predict the test y
        self.classifier = cv2.KNearest()
        # train dataset
        Xt = self.train_data
        yt = self.y
        # train knn classifier
        self.classifier.train(Xt, yt)


    def predict(self, Xte):
        # predict test data labels, res is an (1,1) array, but only 1 element, so extract it
        Xte = np.array([Xte])
        try:
            retval, res, neighborResponses, dists = self.classifier.find_nearest(Xte, 1)
        except Exception, e:
            print str(e)

        result = res[0,0]
        return result

actions = {'boxing':1, 'handclapping':2, 'handwaving':3, 'jogging':4, 'running':5, 'walking':6}

class MHIClassifier():
    def __init__(self):
        self.testvideo = "IMG_1337.avi"

    def build_classifier(self):
        # createbinary sequence and MHI params for each person

        actions = [('boxing',1), ('handclapping',2), ('handwaving',3), ('jogging',4), ('running',5), ('walking',6)]
        #the total number of frames used for training, length of video * number of frames per second
        # frames = {'boxing':(0,108,36),
        #           'handclapping':(0,81,27),
        #           'handwaving':(0,144,48),
        #           'jogging':[(0,300,100),(),()],
        #           'running':(0,225,75),
        #           'walking':(0,525,175)}

        frames = {'boxing':[(0, 36),(36, 72),(72, 108)],
                  'handclapping':[(0, 27),(27, 54),(54, 81)],
                  'handwaving':[(0, 48),(48, 96),(96, 144)],
                  'jogging':[(15, 70),(145, 200),(245, 300)],
                  'running':[(15, 37),(114, 137),(192, 216)],
                  'walking':[(18, 88),(242, 320),(441, 511)]}
        #Th needs to be chosen for each action
        Th = [8,8,10,40,5,5]
        #T needs to be chosen and differs with actions
        T = [50,30,50,50,30,10]

        #create all the action MHI, MEI, and labels
        MotionHistoryImgs = []
        y = []

        frame_ids = [5, 12, 20]
        MotionEnergyImgs = []

        for i, (key, action) in enumerate(actions):

            # start, end, number = frames[key]
            # for j in range(start, end, number):
            for j, (start, end) in enumerate(frames[key]):
                number = end - start
                videofile = 'person15_'+str(key)+'_d1_uncomp.avi'
                #create createbinary squence of each video frame, the KSz needs not too large
                bimg,rel = createbinary(BSz=(5,)*2, sgm=0, video_name=videofile, KSz=(3,)*2, start=start, length=number, Th=Th[i])
                # calculate the motion history image,each image represents each action,and each image represents the whole frames
                MHI = createmhi(length=len(bimg), bimg=bimg, T=T[i]).astype(np.float)
                # normalize the motion history image
                cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
                MotionHistoryImgs.append(MHI)
                MEI = (255*MHI>0).astype(np.uint8)
                MotionEnergyImgs.append(MEI)
                y.append(action)

                # start+=number
            #save createbinary image per each action
            for j in frame_ids:
                out_str = "createbinary" + "-{}-{}.png".format(key,j)
                image = bimg[j]

                cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
                save_image(out_str, image)

            for j in frame_ids:
                out_str = "rel" + "-{}-{}.png".format(key, j)
                image = rel[j]


                save_image(out_str, image)

            #save the sample MHI

            out_str = "MHI" + "-{}.png".format(key)
            save_image(out_str, MHI)

            out_str = "MEI" + "-{}.png".format(key)
            save_image(out_str, MEI)

        #motion energy image
        # MotionEnergyImgs = [(255*M>0).astype(np.uint8) for M in MotionHistoryImgs]

        #calculate createhu moments,central moments and scale invariant moment
        cms = []
        sims = []
        #MotionHistoryImgs and MotionEnergyImgs are list, number of elements are equal to the number of actions inputed
        for MHI, MEI in zip(MotionHistoryImgs, MotionEnergyImgs):
            #cm1 is a 8 elements list stored 8 float values reflecting the moments and it's the statistical description of the image data
            cm1, sim1 = createhu(MHI)
            cm2, sim2 = createhu(MEI)
            #append doesn't change array shape, add more elements to the array, the cms is a list contains actions number of elements, and each element
            #cms is a array shape with (16,)
            cms.append(np.append(cm1, cm2))
            sims.append(np.append(sim1, sim2))

        #now change cms and sims to be array, row:number of actions, column: number of moments
        # y is the dependent, shape is (16,) vector
        y = np.array(y).astype(np.int)
        sims = np.array(sims).astype(np.float32)
        self.classifier = MHIKnn(sims,y)
        self.classifier.leave_one_out()
        self.classifier.train()

    def action_recognation(self):


        frame_start = [203, 380, 520, 809, 1028, 1228]
        length = [70, 67, 80, 91, 34, 43]
        # Th needs to be chosen for each action
        Th = [5, 55, 50, 50, 70, 60]
        # T needs to be chosen and differs with actions
        T = [40, 60, 60, 60, 60, 60]

        # frame_start = [156, 349, 510, 761, 982, 1167]
        # length = [67, 48, 55, 97, 34, 66]
        # # Th needs to be chosen for each action
        # Th = [40, 50, 25, 50, 70, 40]
        # # T needs to be chosen and differs with actions
        # T = [60, 60, 20, 60, 60, 40]

        # frame_start = [277, 463, 631, 943, 1164, 1665]
        # length = [76, 40, 71, 127, 35, 135]
        # # Th needs to be chosen for each action
        # Th = [80, 80, 20, 50, 80, 80]
        # # T needs to be chosen and differs with actions
        # T = [40, 40, 20, 20, 40, 40]

        #create all the action MHI, MEI, and labels
        MotionHistoryImgs = []
        MotionEnergyImgs = []
        # labels = []

        frame_ids = [10,20,30]

        #construct the MHI images
        for i in range(len(frame_start)):
            #create createbinary squence of each video frame, the KSz needs not too large
            bimg,rel = createbinary(BSz=(15,)*2, sgm=0, video_name=self.testvideo, KSz=(9,)*2, start=frame_start[i], length=length[i], Th=Th[i])
            # calculate the motion history image,each image represents each action
            MHI = createmhi(length=len(bimg), bimg=bimg,T=T[i]).astype(np.float)
            # normalize the motion history image
            cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)
            MotionHistoryImgs.append(MHI)
            MEI = (255*MHI>0).astype(np.uint8)
            MotionEnergyImgs.append(MEI)
            # labels.append(action)
            # save createbinary image per each action
            for j in frame_ids:
                out_str = "Test_binary" + "-action{}-frame{}.png".format(i,j)
                image = bimg[j]
                cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
                save_image(out_str, image)

            # save the sample MHI

            out_str = "Test_MHI" + "-{}.png".format(i)
            save_image(out_str, MHI)

            out_str = "Test_MEI" + "-{}.png".format(i)
            save_image(out_str, MEI)

        #calculate createhu moments,central moments and scale invariant moment
        cms = []
        sims = []
        #MotionHistoryImgs and MotionEnergyImgs are list, number of elements are equal to the number of actions inputed
        for MHI, MEI in zip(MotionHistoryImgs, MotionEnergyImgs):
            #cm1 is a 8 elements list stored 8 float values reflecting the moments
            cm1, sim1 = createhu(MHI)
            cm2, sim2 = createhu(MEI)
            #append doesn't change array shape, add more elements to the array, the cms is a list contains actions number of elements, and each element
            #cms is a array shape with (16,)
            cms.append(np.append(cm1, cm2))
            sims.append(np.append(sim1, sim2))

        #now change cms and sims to be array, row:number of actions, column: number of moments
        sims = np.array(sims).astype(np.float32)

        #now using the pretrained classifer to predict the label of the frames

        res = [self.classifier.predict(sim) for sim in sims]
        ends = [start+num for start, num in zip(frame_start,length)]

        #start generate the output frames
        video_create(video_name=self.testvideo, res=res, frame_starts=frame_start, frame_ends=ends)


def createbinary(BSz, sgm, video_name, KSz,start,length,Th):
    video = os.path.join(VID_DIR, video_name)

    image_gen = video_frame_generator(video)
    imgo = image_gen.next()
    imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
    imgo = cv2.GaussianBlur(imgo, BSz, sgm)
    seq_count = 0
    while imgo is not None:

        if seq_count == start:
            imgn = image_gen.next()
            imgn = cv2.cvtColor(imgn, cv2.COLOR_BGR2GRAY)
            imgn = cv2.GaussianBlur(imgn, BSz, sgm)
            seq=[]
            rel=[]
            K = np.ones(KSz, dtype=np.uint8)
            end = start + length
            for i in range(start, end):
                #generate newer image
                #create createbinary img, compute the frame difference
                bimg = np.abs(cv2.subtract(imgn, imgo)) >= Th
                bimg = bimg.astype(np.uint8)
                # "clean up" the createbinary images
                bimg = cv2.morphologyEx(bimg, cv2.MORPH_OPEN, K)
                rel.append(imgn)
                seq.append(bimg)
                imgo = imgn
                seq_count +=1
                imgn = image_gen.next()
                imgn = cv2.cvtColor(imgn, cv2.COLOR_BGR2GRAY)
                imgn = cv2.GaussianBlur(imgn, BSz, sgm)
            break
        #these two lines of code make sure if the image not start from the begining
        imgo = image_gen.next()
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
        imgo = cv2.GaussianBlur(imgo, BSz, sgm)
        seq_count+=1
    return seq,rel

def createmhi(length, bimg, T):
    Itau = np.zeros(bimg[0].shape, dtype=np.float)
    for i, Bimg in enumerate(bimg):
        if i == length:
            break
        mask1 = Bimg == 1
        mask0 = Bimg == 0
        Itau = T * mask1 + np.clip(np.subtract(Itau, np.ones(Itau.shape)), 0, 255) * mask0
    mhi = Itau.astype(np.uint8)
    return mhi


def createhu(image):
    # pqpair features
    pqpair = [(2,0), (0,2), (1,1), (1,2), (2,1), (2,2), (3,0), (0,3)]
    # np.arange(w) is a (160,) vector, so multiply every column in the image
    # an unknown dimension and we want numpy to figure it out
    #np.arange(h) is a (120,) vector, reshape((-1,1)) makes it to be (120,1), and
    u00 = image.sum()
    h,w = image.shape
    xm = np.sum(np.arange(w) * image) / u00
    ym = np.sum(np.arange(h).reshape((-1,1)) * image) / u00
    cm = np.zeros(len(pqpair))
    sim = np.zeros(len(pqpair))
    for i,(p,q) in enumerate(pqpair):
        diffx = np.arange(w) - xm
        diffy = np.arange(h) - ym
        #calculate the central moments
        upq = np.sum(((diffy ** q).reshape((-1,1))) * (diffx ** p) * image)
        cm[i] = upq
        #calculate the scale invariance moments
        sim[i] = cm[i] / u00 ** (1+(p+q)/2)
    return cm, sim

#this code ref at https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def pltcnf(cm, actions):
    cm = (cm * 100 / cm.sum()).astype(np.uint) / 100.0
    th = cm.max() / 2.
    matplotlib.pyplot.figure(figsize=(8, 6.5))
    matplotlib.pyplot.imshow(cm, interpolation='nearest', cmap=matplotlib.pyplot.cm.Oranges)
    matplotlib.pyplot.title('Confusion matrices')
    matplotlib.pyplot.ylabel('Actual')
    matplotlib.pyplot.xlabel('Predicted')
    tm = np.arange(len(actions))
    matplotlib.pyplot.xticks(tm, actions)
    matplotlib.pyplot.yticks(tm, actions)
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.tight_layout()
    filename = 'confusion_matrices.png'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            matplotlib.pyplot.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > th else "black")
    
    matplotlib.pyplot.savefig(os.path.join(OUT_DIR, filename))

#this code is ref from the assignment 3
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:

            yield frame
        else:
            break
    video.release()
    yield None


def video_create(video_name, res, frame_starts, frame_ends):

    # frame_ids = [220, 270, 400, 430, 540, 570]

    # frame_ids = [176, 369, 520, 781, 1002, 1187]

    # frame_start = [203, 380, 520, 809, 1033, 1221]

    frame_ids = [223, 400, 540, 829, 1044, 1241]


    fps = 40

    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.next()
    h, w, d = image.shape

    out_path = "output/mhi_{}".format(video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    counter_init = 1
    output_counter = counter_init
    save_image_counter = 1

    frame_num = 1

    while image is not None:

        # print "Processing fame {}".format(frame_num)
        #the result is updated upon the output_counter
        result = res[(output_counter - 1) % len(res)]
        frame_start = frame_starts[(output_counter - 1) % len(res)]
        frame_end = frame_ends[(output_counter - 1) % len(res)]

        if frame_num == frame_end:
            output_counter += 1

        if frame_num >= frame_start and frame_num <= frame_end:
            mark_location(image, result)

        current_id = frame_ids[(save_image_counter - 1) % len(frame_ids)]

        if current_id == frame_num:
            out_str = "Test_frame_output" + "-{}.png".format(current_id)
            save_image(out_str, image)
            save_image_counter += 1

        video_out.write(image)

        image = image_gen.next()

        frame_num += 1

    video_out.release()


#code ref from assignment3
def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

#code ref from assignment3
def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)

#code ref from assignment3
def mark_location(image, result):

    color = (205, 0, 0)
    h, w, d = image.shape
    p1 = [int(w/7), int(h/5)]
    p2 = [p1[0]+350, p1[1]+80]
    # cv2.rectangle(image, tuple(p1), tuple(p2), (0,255,0), 2)
    # cv2.imshow('rec',image)

    for key, val in actions.items():
        if val == result:
            txt = key

    # cv2.circle(image, p1, 3, color, -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "(predicated:{})".format(txt), (p1[0]-5, p1[1]+20), font, 2.5, color, 1)
    # cv2.imshow('show',image)



if __name__=='__main__':
    print "--- Final Project ---"
    cls = MHIClassifier()
    cls.build_classifier()
    cls.action_recognation()




