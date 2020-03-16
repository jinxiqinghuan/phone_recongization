import cv2 as cv
img=cv.imread('caller.jpg')
#左上角的坐标是(x,y) 矩形的宽度和高度(w,h)
x,y,w,h=50,50,50,50
cv.rectangle(img,(x,y,x+w,y+h),color=(0,255,255),thickness=3) #BGR
#绘制圆center元组指圆点的坐标  radius：半径
x,y,r=100,100,50
cv.circle(img,center=(x,y),radius=r,color=(0,0,255),thickness=2)
#显示图片
cv.imshow('rectangle_img',img)
cv.waitKey(0)
cv.destroyAllWindows()
