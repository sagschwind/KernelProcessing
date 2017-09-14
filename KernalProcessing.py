from PIL import Image
import numpy as np
import math

"""
Input: Name of image in Images folder
Output: Return image object
"""
def openImage(imageName):
    return Image.open("Images/" + imageName)

"""
Input: Image Object
Output: Grey Scale array for each pixel on a 0.0 to 1.0 scale
"""
def getGreyScale(image):

    im = np.asarray(image)

    greyScale = [[0.0 for i in range(len(im[0]))] for j in range(len(im))]

    height, width = 0, 0

    while height < len(greyScale):
        width = 0

        while width < len(greyScale[0]):
            pix = im[height][width]
            greyScale[height][width] = (int(pix[0]) + int(pix[1]) + int(pix[2])) / 3.0
            width += 1

        height += 1

    return greyScale

def getRGB(image):
    im = np.asarray(image)

    RGB = [[(i[0],i[1],i[2]) for i in j] for j in im]

    return RGB


"""
Input: Grey scale array and a kernal (nxn array of ints where n is an odd number)
Output: Tuple that holds the (x,y) kernel operation

* This edge detection skips the outside pixels

"""
def kernelOperation(gs, kernel):
    if len(kernel) % 2 != 1 or len(kernel[0]) % 2 != 1:
        raise Exception("Kernel is not properly sized (Odd by Odd dimensioned)")

    if type(gs[0][0]) is float:
        img = [[0 for i in range(len(gs[0]))] for j in range(len(gs))]
        operFunction = summationSingleValue
    else:
        img = [[[0,0,0] for i in range(len(gs[0]))] for j in range(len(gs))]
        operFunction = summationTripleValue

    kernelGap = len(kernel) // 2

    opX, opY = 0, 0  # Current indecies that the sobel operator is focused on

    sX, sY = 0, 0  # Current indecies of sobel operator

    while opY < len(img):
        opX = 0
        while opX < len(img[0]):
            img[opY - kernelGap][opX - kernelGap] = operFunction(gs, kernel, kernelGap, opX, opY)
            opX += 1
        opY += 1

    return img

def summationSingleValue(img, kernel, gap, opX, opY):
    sX, sY = 0, 0
    summation = 0

    while sY < len(kernel):
        sX = 0
        while sX < len(kernel[0]):
            a = kernel[sY][sX]
            b = img[(opY + sY - gap) % len(img)][(opX + sX - gap) % len(img[0])]
            summation += a * b
            sX += 1
        sY += 1
    return summation

def summationTripleValue(img, kernel, gap, opX, opY):
    sX, sY = 0, 0
    summation = [0,0,0]

    while sY < len(kernel):
        sX = 0
        while sX < len(kernel[0]):
            for i in range(3):
                a = kernel[sY][sX]
                b = img[(opY + sY - gap) % len(img)][(opX + sX - gap) % len(img[0])][i]
                summation[i] += a * b
            sX += 1
        sY += 1
    return summation

"""
Input: Two arrays from edge detection (x,y)
Output: HSV arraay
"""
def gradientCreation(imgA, imgB):

    if len(imgA) != len(imgB) and len(imgA[0]) and len(imgB[0]):
        raise Exception("Dimensions of both arrrays are not equivalent")

    img = [[(0,0,0) for i in range(len(gs[0]))] for j in range(len(gs))]

    x, y = 0, 0
    maxIntensity = 0

    #Combines both images to create an array that represent the image in HSV, but V can be > 100
    while y < len(imgA):
        x = 0

        while x < len(imgB[0]):
            if imgA[y][x] == 0.0:
                img[y][x] = (0,0,0)
            else:
                deltaY = imgB[y][x] #The current pixels Y gradient component
                deltaX = imgA[y][x] #The current pixels X gradient component

                intensity = math.sqrt(deltaX**2 + deltaY**2)

                if intensity > maxIntensity :
                    maxIntensity = intensity

                img[y][x] = ((math.degrees(math.atan(deltaY/deltaX)) + 90) * 2,
                            255,
                            intensity)

            x+=1

        y += 1

    x, y = 0, 0

    #Scales V to 0-100
    while y < len(imgA):
        x = 0
        while x < len(imgA[0]):
            img[y][x] = (int(img[y][x][0]), int(img[y][x][1]), int(255.0 * img[y][x][2] / maxIntensity))
            x+=1

        y += 1

    return img


#Different matricies
sobelX = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
sobelY = [[-1,-2,-1],
          [ 0, 0, 0],
          [ 1, 2, 1]]
gaussian = [[1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16]]
sharpen = [[ 0, -1,  0],
           [-1,  5, -1],
           [ 0, -1,  0]]

im = openImage("Snow.jpg")

gs = getGreyScale(im)

rgb = getRGB(im)

imgX = kernelOperation(gs, sobelX)
imgY = kernelOperation(gs, sobelY)
# imgSharp = kernelOperation(gs, sharpen)

img = Image.new("F", (len(imgX[0]),len(imgX)))
img.putdata([abs(j) for i in imgX for j in i])
img.show(title="edgesX")

img = Image.new("F", (len(imgY[0]),len(imgY)))
img.putdata([abs(j) for i in imgY for j in i])
img.show(title="edgesY")

# img = Image.new("F", (len(imgSharp[0]),len(imgSharp)))
# img.putdata([j for i in imgSharp for j in i])
# img.show(title="sharpen")

a = gradientCreation(imgX, imgY)

img = Image.new("HSV", (len(a[0]), len(a)))
img.putdata([j for i in a for j in i])
img.show(title="Gradient")

print("Done")