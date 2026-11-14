"""Current-image optical-flow evidence. Extrapolation is for searching, not output."""
import cv2
import numpy as np


def flow_box(previous, current, box):
    if previous is None or previous.shape != current.shape:
        return None
    h,w=previous.shape
    x1,y1,x2,y2=box
    mask=np.zeros((h,w),np.uint8)
    mask[max(0,int(y1)):min(h,int(y2)),max(0,int(x1)):min(w,int(x2))]=255
    points=cv2.goodFeaturesToTrack(previous,maxCorners=100,qualityLevel=.01,minDistance=4,mask=mask)
    if points is None or len(points)<6:return None
    next_points,ok,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(31,31),maxLevel=3)
    if next_points is None or ok is None:return None
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(current,previous,next_points,None,winSize=(31,31),maxLevel=3)
    if back is None or back_ok is None:return None
    good=(ok.ravel()!=0)&(back_ok.ravel()!=0)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
    good &= np.isfinite(next_points.reshape(-1,2)).all(axis=1)
    a,b=points[good].reshape(-1,2),next_points[good].reshape(-1,2)
    if len(a)<6:return None
    matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5)
    if matrix is None or inliers is None or np.mean(inliers)<.6:return None
    scale=float(np.hypot(matrix[0,0],matrix[1,0]))
    if not .75<=scale<=1.33:return None
    corners=np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]],np.float32)
    warped=cv2.transform(corners,matrix)[0]
    result=[float(warped[:,0].min()),float(warped[:,1].min()),float(warped[:,0].max()),float(warped[:,1].max())]
    if result[0]<0 or result[1]<0 or result[2]>w or result[3]>h:return None
    return result
