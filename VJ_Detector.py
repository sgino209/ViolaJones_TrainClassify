#!/usr/bin/env python
import cv2

# User Arguments:
NEPS_EN  = False
TRASH_EN = False
SCF_EN   = False
BARK_EN  = True
GRASS_EN = False

IMG_FILE = '../../../Data/31/31FT/usda_31_3_5g_i118.tiff'
TRAINING_PATH = '../Training/r04_20x20'

# --------------------------------------------------------------------------------
# noinspection PyPep8Naming
def vj_detect(_img, _cascadeFile, _color, _scaleFactor, _minNeighbors):
    """Performs Viola-Jones detection"""

    cascade = cv2.CascadeClassifier(_cascadeFile)
    markings = cascade.detectMultiScale(gray, _scaleFactor, _minNeighbors)
    for (x, y, w, h) in markings:
        _img = cv2.rectangle(_img, (x, y), (x + w, y + h), _color, 2)

    return len(markings)

# ----------------------------- S T A R T ----------------------------------------
# Read an input image (+RGB2GRAY):
img = cv2.imread(IMG_FILE)
#img = cv2.detailEnhance(cv2.blur(img,(3,3)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------------------------------------
# VJ Detection, using the classifiers (aka 'Cascades'):
# Parameters:
#   scaleFactor - Parameter specifying how much the image size is reduced at each image scale (def=1.1).
#   minNeighbors - Parameter specifying how many neighbors each candidate rectangle should have to retain it (def=3).
#   minSize - Minimum possible object size. Objects smaller than that are ignored (def=Size()).
#   maxSize - Maximum possible object size. Objects larger than that are ignored (def=Size()).
if NEPS_EN:
    found = vj_detect(img, TRAINING_PATH + '/training_neps/data/cascade.xml', (255, 0, 0), 1.3, 11)
    print "NEPS: Found " + str(found) + " objects"

if TRASH_EN:
    found = vj_detect(img, TRAINING_PATH + '/training_trash/data/cascade.xml', (0, 255, 0), 1.3, 11)
    print "TRASH: Found " + str(found) + " objects"

if SCF_EN:
    found = vj_detect(img, TRAINING_PATH + '/training_scf/data/cascade.xml', (255, 0, 255), 1.3, 11)
    print "SCF: Found " + str(found) + " objects"

if BARK_EN:
    found = vj_detect(img, TRAINING_PATH + '/training_bark/data/cascade.xml', (255, 255, 0), 1.3, 11)
    print "BARK: Found " + str(found) + " objects"

if GRASS_EN:
    found = vj_detect(img, TRAINING_PATH + '/training_grass/data/cascade.xml', (0, 255, 255), 1.3, 11)
    print "GRASS: Found " + str(found) + " objects"

# --------------------------------------------------------------------------------
# Plot the Image and the Detector marking result:
img_scl = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('img', img_scl)
cv2.waitKey(0)
cv2.destroyAllWindows()
