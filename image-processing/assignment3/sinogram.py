# Generate a sinogram, then filter it, then backproject it to get the original image.
#
# You'll need these python libraries: numpy, scipy, pyOpenGL, and pypng
#
# You may use only the following function calls from numpy:
#
#    np.empty, np.array, np.zeroes, np.fft.fft, np.fft.ifft, sum (applied to a numpy array)
#
# You may use only the following function call from scipy:
#
#    ndimage.rotate
#
# You MAY NOT USE numpy's convolve, fftshift, ifftshift, fft, ifft2


import sys, os, math
import png
import numpy as np
from scipy import ndimage

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *


numSinoAngles = 64  # number of discrete angles in the sinogram.  Use a small number
                    # for development, then a larger number to generate good images.

windowWidth  = 1000 # window dimensions
windowHeight =  800

image        = None  # original image
sino         = None  # sinogram
sinoFiltered = None  # filtered sinogram
bp           = None  # backprojection
bpFiltered   = None  # filtered backprojection

imageFilename = ""



# Build sinogram
#
# [2 marks]
#
# The sinogram has 'sinoRows' rows, with each row corresponding to an angle.
# The sinogram has 'dim' columns, the same number of columns as the input image.
#
# The input image is square.


def buildSinogram( image, sinoRows ):

    dim = image.shape[0] # image is square: dim x dim
    
    sino = np.empty( (sinoRows,dim) )
    # YOUR CODE HERE
    for i in range(sinoRows): 
        temp = ndimage.rotate(image, 180*i/sinoRows, reshape = False) #rotate around the image 180 degrees in increments of 180/sinoRows
        sino[i] = np.sum(temp, axis = 0) #compress the image to into a small slice
        
    return sino



# Compute the filtered sinogram
#
# [2 marks]
#
# Apply the Ram-Lak filter to the sinogram IN THE FOURIER DOMAIN, then
# apply the inverse transform to bring the filtered sinogram back to
# the spatial domain.
#
# Do not convolve the inverse Fourier transform of the filter with the
# spatial-domain sinogram.
#
# You may not use numpy's fftshift or ifftshift.
#
# Return only the real part of the filtered sinogram (i.e. not the
# complex part).


def computeFilteredSinogram( sino ):
    filter = np.zeros(sino.shape[0]) #initialize filter 
    length = sino.shape[0] 

    #split filter into absolute value components
    for i in range (length//2):
        filter[i] = 0.5/32*(i) 

    for j in range (length//2 + 1):
        filter[j+31] = 0.5/32*(32-j) 

    #Convert to fft
    sinofft = np.fft.fft(sino, axis=0) 

    #Multiple sinogram fft and filter
    sinoFiltered = sinofft * filter[:, None]

    #inverse transform the sinogram fft
    sinoFiltered = np.real(np.fft.ifft(sinoFiltered, axis=0))

    return sinoFiltered





# Compute the backprojection
#
# [3 marks]
#
# Build the backprojection in a square image of dimension equal to the
# number of sinogram columns.
#
# Scale the backprojection correctly.


def computeBackprojection( sinogram ):

    sinoRows = sinogram.shape[0]

    dim = sinogram.shape[1] # final image is dim x dim
    
    image = np.zeros( (dim,dim), dtype=np.float64 )

    # YOUR CODE HERE
          
    for i in range(sinoRows):
        #get slice from sinogram 
        band = sinogram[i,:]
        tempImage = np.zeros( (dim,dim), dtype=np.float64 )

        #stretches slice to be 512x512 dimensions
        for j in range(dim):
           tempImage[j,:] = band
        
        #rotate temp image to be added to image to create the back projection
        tempImage = ndimage.rotate(tempImage, -180*i/sinoRows, reshape = False)

        image = image + tempImage

    return image



# ---------------- DO NOT CHANGE CODE BELOW THIS LINE ----------------
# ---------------- DO NOT CHANGE CODE BELOW THIS LINE ----------------
# ---------------- DO NOT CHANGE CODE BELOW THIS LINE ----------------



# Load a 16-bit image

def loadImage( path ):

    try:
        pngdata = png.Reader( path ).read()
    except:
        print( 'Failed to load image %s' % path )
        sys.exit(1)

    rows = [ row for row in pngdata[2] ]

    rows.reverse() # flip vertically

    return np.array( rows )



# Save a uint16 array as an image

def saveImage( data, path ):

    png.from_array( np.flip(data,0), 'L;16' ).save( path )



# scale an image to the 16-bit range [0,65535] and convert it to a uint16 array

def normalizeTo16bit( image ):

    min = image.min()
    max = image.max()

    if min == max or math.isnan(min) or math.isnan(max):
        return image.astype( np.uint16 )
    else:
        return (65535.0 * (image-min)/(max-min)).astype( np.uint16 )



# Handle keyboard input

def keyboard( key, x, y ):

    global sino, bp, sinoFiltered, bpFiltered
    
    if key == b'\x1b': # ESC = exit
        glutLeaveMainLoop()

    elif key == b's':             # sinogram
        if sino is None:
            sino = buildSinogram( image, numSinoAngles )
            glutPostRedisplay()

    elif key == b'b':             # backprojection
        if sinoFiltered is not None and bpFiltered is None:
            bpFiltered = normalizeTo16bit( computeBackprojection( sinoFiltered ) )
            glutPostRedisplay()
        elif sino is not None and bp is None:
            bp = normalizeTo16bit( computeBackprojection( sino ) )
            glutPostRedisplay()

    elif key == b'f':             # filter sinogram
        if sinoFiltered is None:
            sinoFiltered = computeFilteredSinogram( sino )
            glutPostRedisplay()

    else:
        print( 's   - compute sinogram' )
        print( 'b   - compute backprojection' )
        print( 'f   - filter sinogram' )

  
# Handle special key (e.g. arrows) input

def special( key, x, y ):

    pass



# Handle window reshape

def reshape( newWidth, newHeight ):

    global windowWidth, windowHeight

    windowWidth  = newWidth
    windowHeight = newHeight

    glViewport( 0, 0, windowWidth, windowHeight )

    glutPostRedisplay()



# Handle mouse click

currentButton = None

zoom = 1.0                      # amount by which to zoom images
translate = (0.0,0.0)           # amount by which to translate images

initX = 0
initY = 0
initZoom = 0
initTranslate = (0,0)



def mouse( button, state, x, y ):

    global currentButton, initX, initY, initZoom, initTranslate

    if state == GLUT_DOWN:

        currentButton = button
        initX = x
        initY = y
        initZoom = zoom
        initTranslate = translate

    elif state == GLUT_UP:

        currentButton = None



# Handle mouse dragging
#
# Zoom out/in with right button dragging up/down.
# Translate with left button dragging.


def mouseMotion( x, y ):

  global zoom, translate

  if currentButton == GLUT_RIGHT_BUTTON:

    # zoom

    factor = 1 # controls the zoom rate
    
    if y > initY: # zoom in
      zoom = initZoom * (1 + factor*(y-initY)/float(windowHeight))
    else: # zoom out
      zoom = initZoom / (1 + factor*(initY-y)/float(windowHeight))

  elif currentButton == GLUT_LEFT_BUTTON:

    # translate

    translate = ( initTranslate[0] + (x-initX)/zoom, initTranslate[1] + (initY-y)/zoom )

  glutPostRedisplay()



# Get information about how to place the images.
#
# toDraw                       2D array of images
# captions                     2D array of captions
# rows, cols                   rows and columns in array
# maxHeight, maxWidth          max height and width of images
# scale                        amount by which to scale images
# horizSpacing, vertSpacing    spacing between images


def getImagesInfo():

  toDraw = [ [ image,   sino,         bp         ],
             [ None,    sinoFiltered, bpFiltered ] ]

  captions = [ [ imageFilename, "sinogram",          "backprojection" ],
               [ None,          "filtered sinogram", "filtered backprojection" ] ]

  rows = len(toDraw)
  cols = len(toDraw[0])

  # Find max image dimensions

  maxHeight = 0
  maxWidth  = 0
  
  for row in toDraw:
    for img in row:
      if img is not None:
        if img.shape[0] > maxHeight:
          maxHeight = img.shape[0]
        if img.shape[1] > maxWidth:
          maxWidth = img.shape[1]

  # Scale everything to fit in the window

  minSpacing = 30 # minimum spacing between images

  scaleX = (windowWidth  - (cols+1)*minSpacing) / float(maxWidth  * cols)
  scaleY = (windowHeight - (rows+1)*minSpacing) / float(maxHeight * rows)

  if scaleX < scaleY:
    scale = scaleX
  else:
    scale = scaleY

  maxWidth  = scale * maxWidth
  maxHeight = scale * maxHeight

  # Draw each image

  horizSpacing = (windowWidth-cols*maxWidth)/(cols+1)
  vertSpacing  = (windowHeight-rows*maxHeight)/(rows+1)

  return toDraw, captions, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing
  

  
# Draw text in window

def drawText( x, y, text ):

    glRasterPos( x, y )
    for ch in text:
        glutBitmapCharacter( GLUT_BITMAP_8_BY_13, ord(ch) )



# Display the initial image, the two sinograms, and the two backprojected images
#
#                   sino              bp
#   initial
#               filtered sino     filtered bp

  
texID = None


def display():

  # Clear window

  glClearColor ( 1, 1, 1, 0 )
  glClear( GL_COLOR_BUFFER_BIT )

  glMatrixMode( GL_PROJECTION )
  glLoadIdentity()

  glMatrixMode( GL_MODELVIEW )
  glLoadIdentity()
  glOrtho( 0, windowWidth, 0, windowHeight, 0, 1 )

  # Set up texturing

  global texID
  
  if texID == None:
    texID = glGenTextures(1)

  glBindTexture( GL_TEXTURE_2D, texID )

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1,0,0,1] );

  # Images to draw, in rows and columns

  toDraw, captions, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

  for r in range(rows):
      for c in range(cols):

          baseX = (horizSpacing + maxWidth ) * c + horizSpacing
          baseY = (vertSpacing  + maxHeight) * (rows-1-r) + vertSpacing

          if captions[r][c] is not None:
              glColor3f( 0.8, 0, 0.2 )
              drawText( baseX, baseY - 18, captions[r][c] )

          if toDraw[r][c] is not None:

              # put image into OpenGL texture map

              img = toDraw[r][c]

              max = img.max()
              min = img.min()
              if max == min:
                  max = min+1

              imgData = np.array( (np.ravel(img) - min) / (max - min) * 255, np.uint8 )

              glBindTexture( GL_TEXTURE_2D, texID )
              glTexImage2D( GL_TEXTURE_2D, 0, GL_INTENSITY, img.shape[1], img.shape[0], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, imgData )

              # Find lower-left corner

              height = scale * img.shape[0]
              width  = scale * img.shape[1]

              # Include zoom and translate

              cx     = 0.5 - translate[0]/width
              cy     = 0.5 - translate[1]/height
              offset = 0.5 / zoom

              glEnable( GL_TEXTURE_2D )

              glBegin( GL_QUADS )
              glTexCoord2f( cx-offset, cy-offset )
              glVertex2f( baseX, baseY )
              glTexCoord2f( cx+offset, cy-offset )
              glVertex2f( baseX+width, baseY )
              glTexCoord2f( cx+offset, cy+offset )
              glVertex2f( baseX+width, baseY+height )
              glTexCoord2f( cx-offset, cy+offset )
              glVertex2f( baseX, baseY+height )
              glEnd()

              glDisable( GL_TEXTURE_2D )

              if zoom != 1 or translate != (0,0):
                  glColor3f( 0.8, 0.8, 0.8 )
                  glBegin( GL_LINE_LOOP )
                  glVertex2f( baseX, baseY )
                  glVertex2f( baseX+width, baseY )
                  glVertex2f( baseX+width, baseY+height )
                  glVertex2f( baseX, baseY+height )
                  glEnd()

  # Done

  glutSwapBuffers()



# Run an interactive session using OpenGL

def runInteractiveSession():

    global imageFilename, image

    # Load image from command-line filename

    imageFilename = sys.argv[1]
    image = loadImage( imageFilename )

    # Run OpenGL for interactive session

    glutInit()

    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB )
    glutInitWindowSize( windowWidth, windowHeight )
    glutInitWindowPosition( 50, 50 )

    glutCreateWindow( 'CT Imaging' )

    glutDisplayFunc( display )
    glutKeyboardFunc( keyboard )
    glutSpecialFunc( special )
    glutReshapeFunc( reshape )
    glutMouseFunc( mouse )
    glutMotionFunc( mouseMotion )

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1,0,0,1] );

    glEnable( GL_TEXTURE_2D )
    glDisable( GL_DEPTH_TEST )

    glutMainLoop()



# Interactive mode is entered if a single filename is on the command-line, like
#
#    gen_sino.py {image filename}
#
# The NON-INTERACTIVE command-line is
#
#    gen_sino.py {image filename} {sinogram filename} {backprojection filename} {filtered sinogram filename} {filtered backprojection filename}
#
# ABSOLUTELY DO NOT CHANGE THIS CODE.  It will be used for testing.
    
if len(sys.argv) == 1:

    print( "For interactive usage: %s {image filename}" )
    sys.exit(1)
    
elif len(sys.argv) in [3,4,5]:

    print( "For non-interactive usage: %s {image filename} {sinogram filename} {backprojection filename} {filtered sinogram filename} {filtered backprojection filename}" % sys.argv[0] )
    sys.exit(1)
    
elif len(sys.argv) == 2:

    runInteractiveSession()

else:

    # All files are specified on command-line

    imageFilename        = sys.argv[1]
    sinoFilename         = sys.argv[2]
    bpFilename           = sys.argv[3]
    sinoFilteredFilename = sys.argv[4]
    bpFilteredFilename   = sys.argv[5]

    # Load image

    image = loadImage( imageFilename )

    image = image * (65536/float(image.max())) # make as bright as possible

    # Build sinogram

    sino = buildSinogram( image, numSinoAngles )

    saveImage( normalizeTo16bit( sino ), sinoFilename )

    # Compute backprojection

    bp = normalizeTo16bit( computeBackprojection( sino ) )

    saveImage( bp, bpFilename )

    # Compute filtered sinogram

    sinoFiltered = computeFilteredSinogram( sino )

    saveImage( normalizeTo16bit( sinoFiltered ), sinoFilteredFilename )

    # Compute filtered backprojection

    bpFiltered = normalizeTo16bit( computeBackprojection( sinoFiltered ) )

    saveImage( bpFiltered, bpFilteredFilename )

