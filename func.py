import numpy as np

#n shape of kernel

#padding constant
#padding_constant=(n-1)/2


#create padding img
#padding_img=np.pad(img,[(padding_constant,padding_constant),(padding_constant,padding_constant)],mode='constant',constant_values=0)
#create and operation
def and_op(first_array,second_array):

    #first array is kernel,second_array is sliced img
    result = np.multiply(first_array, second_array)

    return result
def check_erosion(kernel,array):
    num1,count1=np.unique(kernel, return_counts=True)

    #count how many time 1 appear afeter and operation
    num2, count2 = np.unique(array, return_counts=True)
    if len(count2)>1 and len(count1)>1:
        if count1[1]==count2[1]:
            return 255
        else:
            return 0
    elif len(count2)==1 and len(count1)==1:
        if num1[0]!=0 and num2[0]!=0:
            return 255
        else:
            return 0
    else:
        return 0
def check_dilation(result_array):
    num2, count2 = np.unique(result_array, return_counts=True)
    if len(count2)>1:
        return 255
    elif len(count2)==1 and num2[0]!=0:
        return 255
    else:
        return 0
def main(mode,kernel,img):
    n = np.shape(kernel)[0]
    padding_constant = int((n - 1) / 2)
    padding_img = np.pad(img, [(padding_constant, padding_constant), (padding_constant, padding_constant)],
                         mode='constant', constant_values=0)
    h, w = np.shape(img)

    img_after_effect = np.zeros((h + padding_constant, w + padding_constant),np.uint8)
    if mode=="erosion":
        for j in range(h):
            for i in range(w): # slide along x

                img_after_effect[j+padding_constant,i+padding_constant]=check_erosion(kernel,and_op(kernel,padding_img[j:j+n,i:i+n]))

    elif mode=="dilation":
        new_kernel=flip(kernel)

        for j in range(h):
            for i in range(w): # slide along x

                img_after_effect[j+padding_constant,i+padding_constant]=check_dilation(and_op(new_kernel,padding_img[j:j+n,i:i+n]))

    final_result_img=img_after_effect[padding_constant:padding_constant+h,padding_constant:padding_constant+w]
    return final_result_img
def flip(kernel):
    n = np.shape(kernel)[0]
    new_kernel=np.zeros((n,n),np.uint8)
    for j in range(n):
        for i in range(n):
            new_kernel[n-i-1,n-j-1]=kernel[i,j]
    return new_kernel


first_array=np.zeros([3,3])
first_array[1,2]=1
first_array[0,2]=1
second_array=np.zeros([3,3])
second_array[1,2]=1
kernel=np.zeros([3,3])
kernel[1,1]=1
kernel[1,2]=1
print(check_erosion(kernel,and_op(first_array,second_array)))