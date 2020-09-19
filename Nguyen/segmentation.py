from PIL import Image, ImageDraw, ImageFilter
import cv2
from cv2 import resize

# im1 = Image.open("098.png")
# im2 = Image.open("blue_sky-1152x864.jpg")
#
# im2.paste(im1, (0, 0))
# im2.show()
#
# mask=Image.open("seg.jpg")
#   # im = Image.composite(im1, im2, mask)
# im=im2.copy()
# im.paste(im1,(0,0),mask)
# im.save("t.jpg")
# im.show()
def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

background= cv2.imread('blue_sky-1152x864.jpg')  #thay = tên của file muốn làm background
foreground = cv2.imread('098.png') #thay = tên của file muốn làm foreground
foreground=cv2.resize(foreground,Reverse(background.shape[:-1]))


gray=cv2.imread('098.png',0) #thay = tên của file muốn làm foreground
_,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
mask=cv2.resize(mask,Reverse(background.shape[:-1]))
mask_inv = cv2.bitwise_not(mask)


fg=cv2.bitwise_and(foreground,foreground,None,mask)
bg=cv2.bitwise_and(background,background,None,mask_inv)
dst = cv2.add(bg,fg)


cv2.imwrite("t.jpg",dst) #thay = tên của file kết quả ra