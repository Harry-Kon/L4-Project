#this includes some of the important functions used in this project

import numpy as np
import cv2
from tqdm import tqdm

#functions relevant to intersection over union
def calcarea(point1,point2): #finds the area between two points
    return abs(point1[0]-point2[0])*abs(point1[1]-point2[1])
def findintersection(rectangle1,rectangle2): #finds the intersection of two rectangles given that they intersect
    x1,y1,w1,h1 = rectangle1
    x2,y2,w2,h2 = rectangle2
    corners1 = [[x1,y1],[x1+w1,y1],[x1,y1+h1],[x1+w1,y1+h1]]
    corners2 = [[x2,y2],[x2+w2,y2],[x2,y2+h2],[x2+w2,y2+h2]]
    for i in range(len(corners1)):
        if corners1[i][0] >= x2 and corners1[i][0] <= x2+w2 and corners1[i][1] >= y2 and corners1[i][1] <= y2+h2:
            corner1 = corners1[i]
            corner2 = corners2[3-i]
            return min(calcarea(corner1,corner2),calcarea(corner1,(corner2[0],corners1[3-i][1])),calcarea(corner1,(corners1[3-i][0],corner2[1])),calcarea(corner1,corners1[3-i]))
    for i in range(len(corners2)):
        if corners2[i][0] >= x1 and corners2[i][0] <= x1+w1 and corners2[i][1] >= y1 and corners2[i][1] <= y1+h1:
            corner2 = corners2[i]
            corner1 = corners1[3-i]
            return min(calcarea(corner2,corner1),calcarea(corner2,(corner1[0],corners2[3-i][1])),calcarea(corner2,(corners2[3-i][0],corner1[1])),calcarea(corner2,corners2[3-i]))
    return min(calcarea([x1,y2],[x1+w1,y2+h2]),calcarea([x2,y1],[x2+w2,y1+h1]))
def IoU(rectangle1,rectangle2): #calculates IoU, uses findintersection function 
    x1,y1,w1,h1 = rectangle1
    x2,y2,w2,h2 = rectangle2
    if x1 > x2 +w2 or x2 > x1+w1 or y1> y2+h2 or y2 > y1+h1:
        return 0 
    intersection = findintersection(rectangle1,rectangle2)
    union = calcarea([x1,y1],[x1+w1,y1+h1]) + calcarea([x2,y2],[x2+w2,y2+h2]) - intersection
    return (intersection/union)

#functions relevant to epipolar matching
def importmatrix(filename): #imports a matrix from a .txt file
    with open(filename) as file:
        data1 = file.readlines()
        file.closed
    data2 = []
    for  i in range((len(data1))):
        data2.append([])
        number =''
        for j in range(len(data1[i])):
            if ord(data1[i][j]) != 32:
                   number = number + data1[i][j]
            else:
                 data2[i].append(float(number))
                 number =''
        data2[i].append(float(number))
    return np.matrix(data2)
def skew(x): #finds the skew symmetric matrix from a vector
    return np.matrix([[0, -x[2], x[1]], [x[2], 0, -x[0]],[-x[1], x[0], 0]])    
def calcF(matrix1, matrix2, C): #calculates the fundamental matrix F, C is the position of camera 1's centre, matrix 1 and 2 are the camera projection matrices
    pseudoinvmatrix1 = np.linalg.pinv(matrix1)
    cross1 = skew(np.matmul(matrix2,C))
    cross2 = np.matmul(matrix2,pseudoinvmatrix1)
    return np.matrix(np.matmul(cross1,cross2))
def epipolarlinefrommatrix(epipixel,F): #Calculates the epipolar line using F
    line = np.matmul(F,[epipixel[0],epipixel[1],1])
    intercept = -line.item(2)/line.item(1)
    gradient = -line.item(0)/line.item(1)
    return ([intercept,gradient])
def rectangleonline(rectangle,line): #checks if the epipolar line intersects a rectangle
    gradient = line[1]
    height1 = line[0] + gradient*rectangle[0]
    height2 = gradient*rectangle[2]
    if max(height1,height1+height2) < rectangle[1] or min(height1,height1+height2) > rectangle[1] + rectangle[3]:
        return False
    else:
        return True
def rectanglestorectangles(rectangles1,rectangles2, F): #generates the epipolar pairs for two sets of corresponding rectangles
    dist = []
    epipolarpairs = []
    for t in tqdm(range(len(rectangles1))):
        epipolarpairs.append([])
        for i in range(len(rectangles1[t])):
            epipolarpairs[t].append([])
            centre = centrerectangle(rectangles1[t][i])
            line = epipolarlinefrommatrix(centre, F)
            for j in range(len(rectangles2[t])):
                if rectangleonline(rectangles2[t][j],line) == True:
                    dist.append(j)
                    epipolarpairs[t][i].append(j)
    return epipolarpairs
    
#functions relevant to template matching
def templatematchrectangles(img,template,threshold): #performs the template matching over a whole image, threshold here is matching threshold
    img2 = img.copy()
    img_grey  = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    w,h = template.shape[::-1]
    res = cv2.matchTemplate(img_grey, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res>threshold)
    rectangles = []
    for pt in zip(*loc[::-1]):
        rectangles.append([pt[0],pt[1],w,h])
    return rectangles
def nonmaxsuppression(rectangles,threshold): #performs non-max supression on a set of rectangles, threshold here is NMS threshold
    rectanglescopy = rectangles.copy()
    for i in range(len(rectangles)):
        removelist = []
        for j in range(i+1,len(rectangles)):
            if IoU(rectangles[i],rectangles[j]) > threshold:
                removelist.append(j)
        for k in removelist:
            rectanglescopy[k] = [0,0,0,0]
    return rectanglescopy
def cullbackground(rectangles, imagerectangles, threshold): #removes the rectangles in the background by comparing with the contour bounding rectangles, threshold here should be set to 0
    rectanglescopy = []
    for i in range(len(rectangles)):
        keep = False
        for j in range(len(imagerectangles)):
            if IoU(rectangles[i],imagerectangles[j]) > threshold:
                keep = True
        if keep == True:
            rectanglescopy.append(rectangles[i])
            keep = False
    return rectanglescopy
def findrectangles(img,imagerectangles,template,threshold1,threshold2,threshold3): #performs the three sections of template matching in a row
    rectangles = templatematchrectangles(img,template, threshold1)
    suppressed = nonmaxsuppression(rectangles, threshold2)
    culled = cullbackground(suppressed, imagerectangles, threshold3)
    return culled

#functions relevant to image processing
def mediantemporalfilter(frames): #calculates and returns the median background frame
    medianframe = np.median(frames, axis=0).astype(dtype=np.uint8)
    return medianframe
def removebackground(frames, medianframe): #performs the bitwise XOR comparison between frames and medianframe, doesn't perform threshold cut
    removedBackground = []
    for i in range(len(frames)):
        removedBackground.append(cv2.bitwise_xor(frames[i],medianframe))
    return removedBackground
def thresholdcut(frames, cutsize): #performs the threshold cut into the BGR cube of with thickness cutsize
    mask = []
    for i in range(len(frames)):
        mask.append(cv2.inRange(frames[i],np.array([cutsize,cutsize,cutsize]), np.array([255-cutsize,255-cutsize,255-cutsize])))
    return mask
def getrectanglesimage(img):#gets rectangles for one binary image (not video)
    saverectangles = []
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>5:
                peri =cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x,y,w,h = cv2.boundingRect(approx)
                saverectangles.append([x,y,w,h])
    return saverectangles

#function relevant to one view tracking
def checkrectangle(t,x,y,rectangles): #returns which rectangles in a list are at a given time space coord
    checkrectlist = []
    for j in range(len(rectangles[t])):
        if x > rectangles[t][j][0] and x < rectangles[t][j][0] + rectangles[t][j][2] and y > rectangles[t][j][1] and y < rectangles[t][j][1] + rectangles[t][j][3]:
            checkrectlist.append(j)
    return checkrectlist
def drawarc(rectangle,velocity): #returns the arc around a velocity
    r = (velocity[0]**2 + velocity[1]**2)**(1/2)
    angle = np.angle(velocity[0]+(1j*velocity[1]))
    arcpoints = []
    for k in range(-8,8):
        arcpoints.append([rectangle[0] + rectangle[2]/2 +(r*np.cos(angle+k*1/32*np.pi)),rectangle[1]+ rectangle[3]/2 +(r*np.sin(angle+k*1/32*np.pi))])
    return arcpoints
def velocityrectangle(rectangle1,rectangle2): # returns 'velocity' between 2 rectangles
    x1 ,y1, w1, h1 = rectangle1
    x2, y2, w2, h2 = rectangle2
    midx1, midy1 = x1+(w1/2), y1+(h1/2)
    midx2, midy2 = x2+(w2/2), y2+(h2/2)
    return [midx2-midx1,midy2-midy1]
def centrerectangle(rectangle): #returns centre point of a rectangle format(x,y,w,h) WARNING ISNT IN AN INTEGER FORMAT
    x,y,w,h = rectangle
    return [x+(w/2),y+(h/2)]
def centreofmasspath(rectangles): #returns the centre of mass path
    centreofmass = []
    for t in range(len(rectangles)):
        centreofmass.append([0,0])
        xsum = 0
        ysum = 0
        rectanglecount = len(rectangles[t])
        for i in range(rectanglecount):
            x,y = centrerectangle(rectangles[t][i])
            xsum += x
            ysum += y
        centreofmass[t] = [int(xsum/rectanglecount),int(ysum/rectanglecount)]
    return centreofmass
def smoothvel(path): #returns a smoothed set of velocities from path(used primarily for the centre of mass path)
    comvel = []
    for t in range(len(path)-1):
        comvel.append([path[t+1][0]-path[t][0],path[t+1][1]-path[t][1]])
    avgcomvel = []
    avgrange = 30
    for i in range(len(comvel)):
        if i > avgrange/2 and i < len(comvel)-avgrange/2:
            count = 0
            for j in range(avgrange):
                count = count+comvel[int(i-j+(avgrange/2))][0]
            avgx = count/avgrange
            count = 0
            for j in range(avgrange):
                count = count+comvel[int(i-j+(avgrange/2))][1]
            avgy = count/avgrange
            avgcomvel.append([avgx,avgy])
        else:
            if i <= avgrange/2:
                count = 0
                for j in range(avgrange):
                    count = count +comvel[int(i+j)][0]
                avgx = count/avgrange
                count = 0
                for j in range(avgrange):
                    count = count+comvel[int(i+j)][1]
                avgy = count/avgrange
                avgcomvel.append([avgx,avgy])
            if i >= len(comvel)-avgrange/2:
                count = 0
                for j in range(avgrange):
                    count = count +comvel[int(i-j)][0]
                avgx = count/avgrange
                count = 0
                for j in range(avgrange):
                    count = count+comvel[int(i-j)][1]
                avgy = count/avgrange
                avgcomvel.append([avgx,avgy])
    avgcomvel.append([0,0])
    return avgcomvel
def createvelocitytable(rectangles): #returns a list with shape of the rectangles, for the velocity to be stored in, currently filled with False showing unknown velocities
    velocities = []
    for t in range(len(rectangles)):
        velocities.append([])
        for i in range(len(rectangles[t])):
            velocities[t].append(False)
    return velocities
def findtrajectories(rectangles,velocitytable, comvel): #the function that puts it all together, given rectangles, an empty velocity table and the centre of mass velocities
    trajectories = []
    for t in tqdm(range(len(rectangles)-1)):
        currenttrajectories = []
        currenttrajectoriesdict = {}
        for m in range(t):
            for n in range(len(trajectories[m])):
                if len(trajectories[m][n]) + m == t+1:
                    if trajectories[m][n][-1] not in currenttrajectories:
                        currenttrajectories.append(trajectories[m][n][-1])
                        currenttrajectoriesdict[int(trajectories[m][n][-1])]=[m,n]
        trajectories.append([])
        steps = []
        for i in range(len(rectangles[t])):
            if velocitytable[t][i] != False:    
                arc = drawarc(rectangles[t][i],velocitytable[t][i])
            else:
                arc = drawarc(rectangles[t][i],comvel[t])
            nextrect = []
            for k in range(len(arc)):
                a = checkrectangle(t+1,arc[k][0],arc[k][1],rectangles)
                if a != []:
                    for l in a:
                        if l not in (nextrect):
                            nextrect.append(l)
            steps.append(nextrect)
        for l in range(len(steps)):
            if len(steps[l]) == 1 and steps[l] != []:
                if l in currenttrajectories:
                    m,n = currenttrajectoriesdict[l]
                    trajectories[m][n].append(steps[l][0])
                    velocitytable[t][l]= velocityrectangle(rectangles[t][l], rectangles[t+1][steps[l][0]])
                else:
                    trajectories[t].append([l,steps[l][0]])
                    velocitytable[t][l]= velocityrectangle(rectangles[t][l], rectangles[t+1][steps[l][0]])
            else:
                trajectories[t].append([l])
    return trajectories 

#functions relevant to 3D matching/recontruction
def reformatpairlist(pairlist,reformattedtrajectories, starttime): #writes pairlist as list of trajectories, not as rectangle ids
    reformatpairs = []
    for i in range(len(pairlist)):
        reformatpairs.append([])
        for j in range(len(pairlist[i])):
            reformatpairs[i].append(reformattedtrajectories[starttime + i][pairlist[i][j]])
    return reformatpairs
def findlongestchain(trajpairs): #finds the longest time a trajectory is an epipolar pair
    trajlengths = []
    for i in (range(len(trajpairs))):
        for j in range(len(trajpairs[i])):
            innextstep = True
            count = 1
            while innextstep == True:
                if i+count >= len(trajpairs):
                    innextstep = False
                    break
                if trajpairs[i][j] in trajpairs[i+count]:
                    innextstep = True
                    count += 1
                else:
                    innextstep = False
                    break
            trajlengths.append([i,j,count])
    return trajlengths
def partitiontrajpairs(trajpairs,trajlengths,trajectories,starttime): #partitions trajectories into secitons of the longest matched trajectories starttime is when trajectory1 started
    fullypartitioned = False
    partitionedtrajectory = []
    for i in range(len(trajpairs)):
        partitionedtrajectory.append([])
    while fullypartitioned == False:
        maxlength =  0
        longesttraj = []
        for i in range(len(trajlengths)):
            if trajlengths[i][2] > maxlength:
                longesttraj = trajlengths[i]
                maxlength = longesttraj[2]
        if maxlength < 3:
            fullypartitioned = True
            break   
        trajectory = trajectories[trajpairs[longesttraj[0]][longesttraj[1]][0]][trajpairs[longesttraj[0]][longesttraj[1]][1]]
        for i in range(longesttraj[2]):
            partitionedtrajectory[longesttraj[0]+i-1] = trajectory[starttime + longesttraj[0] - trajpairs[longesttraj[0]][longesttraj[1]][0] + i-1]
        for i in range(len(trajlengths)):
            if trajlengths[i][0] in range(longesttraj[0],longesttraj[0]+longesttraj[2]) or trajlengths[i][0]+trajlengths[i][2] in range(longesttraj[0],longesttraj[0]+longesttraj[2]):
                trajlengths[i] = [0,0,0]
    return partitionedtrajectory
def combineviews(trajectory1,trajectory2): #combines the trajectories from each camera view
    combined = []
    for i in range(min(len(trajectory1),len(trajectory2))):
        combined.append([trajectory1[i],trajectory2[i]])
    return combined
def reprojection(combinedtrajectories,P1,P2,starttime,rectangles1,rectangles2): #reprojects into 3D, using cv2.triangulatePoints
    projpoints1 = []
    projpoints2 = []
    for i in range(len(combinedtrajectories)):
        if combinedtrajectories[i][0] == [] or combinedtrajectories[i][1] == []:
            continue
        else:
            projpoints1.append(np.array([[centrerectangle(rectangles1[starttime + i][combinedtrajectories[i][1]])[0]],[ centrerectangle(rectangles1[starttime + i][combinedtrajectories[i][1]])[1]]],dtype=np.float))
            projpoints2.append(np.array([[centrerectangle(rectangles2[starttime + i][combinedtrajectories[i][1]])[0]],[ centrerectangle(rectangles2[starttime + i][combinedtrajectories[i][1]])[1]]],dtype=np.float))
    points = []
    for i in range(len(projpoints1)):
        points.append(cv2.triangulatePoints(P1, P2, projpoints1[i], projpoints2[i]))
    return points
def normalise3dcoords(reprojection): #converts the coords from homogeneous 4 vectors to Euclidean 3 vectors
    coords = []
    for i in range(len(reprojection)):
        scale = 1/reprojection[i][3]
        coords.append([float(scale*reprojection[i][0]),float(scale*reprojection[i][1]),float(scale*reprojection[i][2])])
    return coords
def twowaymatching(epipolarpairs1, epipolarpairs2, trajectories1, trajectories2, reformattedtraj1, reformattedtraj2, rectangles1, rectangles2, P1, P2): #performs the 3D reconstruction process, both ways at once
    trajpairs1 = []
    for i in tqdm(range(len(trajectories1))):
        trajpairs1.append([])
        for j in range(len(trajectories1[i])):
            pairlist1 = []
            for k in range(len(trajectories1[i][j])):
                pairlist1.append(epipolarpairs1[i+k][trajectories1[i][j][k]])
            trajpairs1[i].append(reformatpairlist(pairlist1, reformattedtraj2, i))
    trajpairs2 = []
    for i in tqdm(range(len(trajectories2))):
        trajpairs2.append([])
        for j in range(len(trajectories2[i])):
            pairlist2 = []
            for k in range(len(trajectories2[i][j])):
                pairlist2.append(epipolarpairs2[i+k][trajectories2[i][j][k]])
            trajpairs2[i].append(reformatpairlist(pairlist2, reformattedtraj1, i))
    fullcoords1 = []
    for i in tqdm(range(len(trajpairs1))):
        fullcoords1.append([])
        for j in range(len(trajpairs1[i])):
            trajlengths1 = findlongestchain(trajpairs1[i][j])
            partitionedtrajectory1 = partitiontrajpairs(trajpairs1[i][j], trajlengths1, trajectories2, starttime = i)
            combined1 = combineviews(trajectories1[i][j],partitionedtrajectory1)
            reprojection1 = reprojection(combined1, P1, P2,i,rectangles1,rectangles2)
            normalised = normalise3dcoords(reprojection1)
            fullcoords1[i].append(normalised)
    fullcoords2 = []
    for i in tqdm(range(len(trajpairs2))):
        fullcoords2.append([])
        for j in range(len(trajpairs2[i])):
            trajlengths2 = findlongestchain(trajpairs2[i][j])
            partitionedtrajectory2 = partitiontrajpairs(trajpairs2[i][j], trajlengths2, trajectories1, starttime = i)
            combined2 = combineviews(trajectories2[i][j],partitionedtrajectory2)
            reprojection2 = reprojection(combined2, P1, P2,i,rectangles1,rectangles2)
            normalised = normalise3dcoords(reprojection2)
            fullcoords2[i].append(normalised)
    return trajpairs1, trajpairs2, fullcoords1, fullcoords2