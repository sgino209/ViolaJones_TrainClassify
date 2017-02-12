#!/usr/bin/env python
import os
import re
import csv
import cv2
import sys

# User Arguments:
SAMPLES_CSV = "samples_26_31_highprob_r04.csv"
SAMPLE_SIZE = (100, 100)
WINDOW_SIZE = (20, 20)
STAGES_NUM = 18
POSITIVE_NUM = 7000
POSITIVE_EXTENSION = False

if len(sys.argv) > 1:
    STAGES_NUM = sys.argv[1]

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def generate_samples(_baseDir, _index):
    """Gets a base directory and an CSV-Index pointer, and generate sample for VJ Training phase"""

    csvIndexFile = _baseDir + '/Data/' + _index
    csvIndex_f = open(csvIndexFile)
    print "Loading Index file: " + csvIndexFile
    csv_f = csv.reader(csvIndex_f)
    pic_num = {}
    label_col, minX_col, maxX_col, minY_col, maxY_col, img_col = 0, 0, 0, 0, 0, 0

    for row in csv_f:

        if row[0] == 'object_id':
            label_col = row.index('predicted_class')
            minX_col = row.index('Coord<Minimum>_0')
            maxX_col = row.index('Coord<Maximum>_0')
            minY_col = row.index('Coord<Minimum>_1')
            maxY_col = row.index('Coord<Maximum>_1')
            img_col = row.index([s for s in row if "exported" in s][0])

        elif row[0] != "":
            lbl = row[label_col].lower()
            x = int(row[minX_col])
            y = int(row[minY_col])
            w = int(row[maxX_col]) - int(row[minX_col])
            h = int(row[maxY_col]) - int(row[minY_col])
            imgCsv = row[img_col].split('.')[0]
            imgName = re.search('.*(usda_.*)_table', imgCsv).group(1)
            serie = re.search('usda_(.+?)_.*', imgName).group(1)

            if lbl not in pic_num:
                pic_num[lbl] = 0

            dirName = _baseDir + '/Code/VJ/Training/' + lbl
            if not os.path.exists(dirName):
                os.makedirs(dirName)

            imgFile = _baseDir + '/Data/' + serie + '/' + serie + 'FT/' + imgName + '.tiff'

            img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)

            img_crop = img[y:y + h, x:x + w]
            img_scl = cv2.resize(img_crop, SAMPLE_SIZE, interpolation=cv2.INTER_CUBIC)
            sampleName = dirName + '/' + lbl + '_' + str(pic_num[lbl]) + ".jpg"
            print "Generating a new " + lbl.upper() + " sample: " + sampleName
            cv2.imwrite(sampleName, img_scl)

            pic_num[lbl] += 1

    csvIndex_f.close()
    print generate_samples.__name__ + " Done!"
    for lbl, pics_num in pic_num.iteritems():
        print lbl.upper() + ' --> ' + str(pics_num) + ' were created'

    return pic_num

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def increase_pos_num(_baseDir, _posLabel, _posNum):
    """create training samples from one image applying distortions"""

    imgsPath = os.listdir(_baseDir + '/' + _posLabel)

    dirName = _baseDir + '/training_' + _posLabel + '/vec'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    for imgFile in imgsPath:

        appName = 'opencv_createsamples'
        appArgs = ' -img ' + _baseDir + '/' + _posLabel + '/' + imgFile +\
                  ' -num ' + str(int(_posNum/len(imgsPath))) +\
                  ' -bg ' + _baseDir + '/training_' + _posLabel + '/bg1.txt' +\
                  ' -vec ' + dirName + '/' + imgFile.split('.')[0] + '.vec' +\
                  ' -maxxangle 0.6' +\
                  ' -maxyangle 0' +\
                  ' -maxzangle 0.3' +\
                  ' -maxidev 100' +\
                  ' -bgcolor 0' +\
                  ' -bgthresh 0' +\
                  ' -w ' + str(WINDOW_SIZE[0]) + ' -h ' + str(WINDOW_SIZE[1])

        print "Launching:  " + appName + " " + appArgs
        os.system(appName + appArgs)
        print increase_pos_num.__name__ + " Done!"

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def merge_pos_vec(_baseDir, _posLabel):
    """Stitch all positive samples into a vector file"""

    appName = 'mergevec.py'
    appArgs = ' -v ' + _baseDir + '/training_' + _posLabel + '/vec' +\
              ' -o ' + _baseDir + '/training_' + _posLabel + '/positives_' + _posLabel + '.vec'

    print "Launching:  " + appName + " " + appArgs
    os.system(appName + appArgs)
    print merge_pos_vec.__name__ + " Done!"

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def create_pos_n_neg(_baseDir, _posLabel):
    """Creates positive index file for posLabel and negative index files for all other labels (used by VJ training)"""

    dirName = _baseDir + '/training_' + _posLabel
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    posFile = open(dirName + '/info.dat', 'w')
    negFile1 = open(dirName + '/bg1.txt', 'w')
    negFile2 = open(dirName + '/bg2.txt', 'w')
    negCntr = 0

    for labelDir in os.listdir(_baseDir):

        if os.path.isdir(labelDir):

            if labelDir == _posLabel:
                for imgFile in os.listdir(_baseDir+'/'+labelDir):
                    line = '../' + labelDir + '/' + imgFile + ' ' +\
                           '1 0 0 ' + str(SAMPLE_SIZE[0]) + ' ' + str(SAMPLE_SIZE[1]) + '\n'
                    posFile.write(line)

            elif labelDir == 'fibers':
                for imgFile in os.listdir(_baseDir + '/' + labelDir):
                    line = labelDir + '/' + imgFile + '\n'
                    negFile1.write('../' + line)
                    negFile2.write('./' + line)
                    negCntr += 1

    negFile1.close()
    negFile2.close()
    posFile.close()
    print create_pos_n_neg.__name__ + " Done!"

    return negCntr

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def generate_pos_vector(_infoFile, _posVecFile, _posNum):
    """Stitch all positive samples into a vector file"""

    appName = 'opencv_createsamples'
    appArgs = ' -info ' + _infoFile +\
              ' -num ' + str(_posNum) +\
              ' -w ' + str(WINDOW_SIZE[0]) + ' -h ' + str(WINDOW_SIZE[1]) + \
              ' -vec ' + _posVecFile

    print "Launching:  " + appName + " " + appArgs
    os.system(appName + appArgs)
    print generate_pos_vector.__name__ + " Done!"

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def show_pos_vector(_posVecFile):
    """Show the positive vector, just for verifying that allright, toggle images by space-bar, exit with ESC"""

    appName = 'opencv_createsamples'
    appArgs = ' -w ' + str(WINDOW_SIZE[0]) + ' -h ' + str(WINDOW_SIZE[1]) + \
              ' -vec ' + _posVecFile

    print "Launching (press ESC to exit):  " + appName + " " + appArgs
    os.system(appName + appArgs)
    print generate_pos_vector.__name__ + " Done!"

# --------------------------------------------------------------------------------------------------------------------
# noinspection PyPep8Naming
def VJ_Training(_posVecFile, _outputXML, _negBgTxtFile, _posNum, _negNum, _stagesNum):
    """Viola-Jones Training, generates a corresponding cascasde file"""

    if not os.path.exists(_outputXML):
        os.makedirs(_outputXML)

    appName = 'opencv_traincascade'
    appArgs = ' -data ' + _outputXML +\
              ' -vec ' + _posVecFile +\
              ' -bg ' + _negBgTxtFile + \
              ' -numPos ' + str(_posNum) + \
              ' -numNeg ' + str(_negNum) + \
              ' -w ' + str(WINDOW_SIZE[0]) + ' -h ' + str(WINDOW_SIZE[1]) + \
              ' -numStages ' + str(_stagesNum) + \
              ' -mode ALL'

    print "Launching:  " + appName + " " + appArgs
    os.system(appName + appArgs)
    print generate_pos_vector.__name__ + " Done!"

# --------------------------------------------------- S T A R T ------------------------------------------------------

# Load CSV index - marks all objects coordinates:
labelsDict = generate_samples('../../..', SAMPLES_CSV)
del labelsDict['fibers']

for label in labelsDict.keys():

    print "-----------------------------------------------------------------------"
    print "Start Training for: " + label
    print "-----------------------------------------------------------------------"

    # Create Positive and Negative samples:
    negCntr = create_pos_n_neg('.', label)

    # Increase Positive sample volume
    if POSITIVE_EXTENSION:
        increase_pos_num('.', label, POSITIVE_NUM)
        merge_pos_vec('.', label)
    else:
        POSITIVE_NUM = len(os.listdir('./' + label))

    # Initialize parameters:
    subDir = './training_' + label
    infoName = subDir + '/info.dat'
    vecName = subDir + '/positives_' + label + '.vec'
    outputXML = subDir + '/data'
    bgName = subDir + '/bg2.txt'
    posNumVec = int(POSITIVE_NUM * 0.9)
    posNumVJ = int(POSITIVE_NUM * 0.7)
    negNumVJ = int(min(negCntr, posNumVJ/2))
    stagesNumVJ = STAGES_NUM

    # Generate Positive vector (stitches all Positive samples):
    generate_pos_vector(infoName, vecName + '_orig.vec', posNumVec)
    #show_pos_vector(vecName)

    # Generate VJ cascade, by performing Training phase:
    VJ_Training(vecName, outputXML, bgName, posNumVJ, negNumVJ, stagesNumVJ)

print "-----------------------------------------------------------------------"
print "All Done!"
print "-----------------------------------------------------------------------"
for label in labelsDict.keys():
    print label + ' ---> ' + './training_' + label + '/data/cascade.xml'
