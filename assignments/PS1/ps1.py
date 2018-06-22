
import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image=np.copy(image)
    r=temp_image[:,:,2]
    return r 
    
    raise NotImplementedError


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image=np.copy(image)
    g=temp_image[:,:,1].copy()
    return g 
    
    raise NotImplementedError


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image=np.copy(image)
    b=temp_image[:,:,0].copy()
    return b 
    raise NotImplementedError


def swap_green_blue(image):
    

    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    img=np.copy(image)
    b,g,r=cv2.split(img)
    img[:,:,0]=g
    img[:,:,1]=b
#     print "original:%s" % image
#     print "new:%s" % temp_image
    return img
    raise NotImplementedError
    


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    
    height=shape[0]
    width=shape[1]
    
    src_height, src_width=src.shape
    dst_height, dst_width=dst.shape
    
#     print src_height, src_width, dst_height, dst_width
    
    src_height_index_start= (src_height-height)/2
    src_width_index_start= (src_width-width)/2
    
    dst_height_index_start= (dst_height-height)/2
    dst_width_index_start= (dst_width-width)/2
#     print src_height_index_start, src_width_index_start, dst_width_index_start
    
    img=np.copy(dst)
    
    img[dst_height_index_start:dst_height_index_start+height,dst_width_index_start:dst_width_index_start+width]\
    =src[src_height_index_start:src_height_index_start+height,src_width_index_start:src_width_index_start+width]
    
    return img 
    
    raise NotImplementedError


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    img=np.copy(image)
#     print (np.min(img), np.max(img), np.mean(img), np.std(img)) 
    return (1.*np.min(img), 1.*np.max(img), np.mean(img), np.std(img)) 
    
    raise NotImplementedError


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    
    img=np.copy(image)
    
    
#     print img.dtype
#     print (np.max(img), np.min(img))
    #     img.astype(float64)
    img=1.*img      
#     print (np.max(img), np.min(img))
    
    mean=np.mean(img)
    std=np.std(img)

    img[:]=(((img[:]-mean)/std)*scale)*1.0 + mean*1.0
    
    return img 
    
    
    raise NotImplementedError


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    
    img = np.copy(image)
     
    if shift==0:
        pass 
    if shift!=0:
        img[:, :-shift] = img[:, shift:]
    #     cv2.imshow('shift', img)
    #     cv2.copyMakeBorder(image, img, 0, 0, 0, 2, cv2.BORDER_REPLICATE)
        img_border = cv2.copyMakeBorder(image,0,0,0,shift,cv2.BORDER_REPLICATE)
        img[:,-shift:]=img_border[:,-shift:]
#     cv2.imshow('shift boarder', img)
    
#     cv2.waitKey(0)
    return img 
    
    raise NotImplementedError


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    
    img_diff=1.*img1-1.*img2
    
#     print np.min(img), np.max(img)
    
    img = np.zeros(img_diff.shape)
    cv2.normalize(img_diff, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     print np.min(img), np.max(img)
    return img

    raise NotImplementedError


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    

    # # 5a
    
    img=np.copy(image)
    img=1.*img 
    
    temp_img=img[:,:,channel].copy()
   
    r,c=temp_img.shape
    
    noise=np.random.randn(r,c)*sigma
    noise_img=temp_img + noise
    
    img[:,:,channel]=noise_img
    
#     print np.max(img), np.min(img)
    return img
    
    
    
    raise NotImplementedError