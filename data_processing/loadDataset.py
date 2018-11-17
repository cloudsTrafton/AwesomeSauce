# from module liac-arff
# import arff
import data_processing.dataUtils as dataUtils
import os.path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# assume no more than 64-bit keycodes
def prettyNgram(keycodes):
    ret = ''
    ret += chr((keycodes & 0xFF000000) >> 24)
    ret += chr((keycodes & 0x00FF0000) >> 16)
    ret += chr((keycodes & 0x0000FF00) >> 8)
    ret += chr((keycodes & 0x000000FF))
    return ret.strip()

#debugging = True
debugging = False

# Dataset Features

# Rank of N-gram we use for features.
# This is the number of consecutive characters typed that we measure
# Needs to be 4 or less.
ngram = 3

# Discard outliers with extreme hold and seek times
discardHoldTime = 200
discardSeekTime = 10000

# ngramNameMask - only want the lower 8*ngram bits for the name
if ngram == 1:
    ngramNameMask = 0x000000FF
elif ngram == 2:
    ngramNameMask = 0x0000FFFF
elif ngram == 3:
    ngramNameMask = 0x00FFFFFF
elif ngram == 4:
    ngramNameMask = 0xFFFFFFFF


datasetRoot = '../age_anonymized/'
datasetFolders = [datasetRoot + '-15/', datasetRoot + '16-19/', datasetRoot + '20-29/', \
                 datasetRoot + '30-39/', datasetRoot + '40-49/', datasetRoot + '50+/']

#datasetFolders = [datasetRoot + '-15/']

print("Reading files in: ", datasetFolders)
print("Found the following raw dataset files:")
fileList = []
for folder in datasetFolders:
    files = os.listdir(folder)
    for file in files:
        fullName = os.path.join(folder, file)
        if os.path.isfile(fullName) and file.endswith('.csv'):
            fileList.append(fullName)
            print(file)

ngramVectors = []
holdtimeVectors = []
seektimeVectors = []
featureVectors = []
discardedUsers = []
userCount = 0
discardCount = 0

print("Processing into feature vectors...")
for file in fileList:
    print(file)

    # Looking at single user:
    totalHoldTime = 0.0
    totalSeekTime = 0.0
    totalNgramTime = 0.0
    averageHoldTime = 0.0
    averageSeekTime = 0.0
    averageNgramTime = 0.0

    with open(file, "r") as f:
        content = f.readlines()

        keystrokeCount = 1                         # note: starts at 1, this is for calculating average
        ngramCount = 0
        n = 0                                      # counter for ngrams
        # temp lists to hold intermediate values before placing them in a feature vector
        #keyCode = []
        #holdTime = []
        #seekTime = []
        featureVector = []
        ngramVector = {}                            # Dictionary containing all these ngram features
        ngramName = ngram * [0]                     # The column of this feature will be
                                                    # the three keycodes concatenated together.
                                                    # For now, store as a dictionary, later,
                                                    # we'll convert these in a sparse matrix
        ngramDuration = ngram * [0]                 # store a "sliding window" of durations as we go
        for line in content:
            n = n % ngram                           # need to keep #ngram counters running
            row = line.strip().split(',')
            if debugging: print(hex(int(row[0])), row)

            if(len(row[0]) == 0 or len(row[1]) == 0 or len(row[2]) == 0): continue

            keyCode = int(row[0])                   # keycode (ASCII)
            holdTime = int(row[1])                  # how long we hold this key down
            seekTime = int(row[2])                  # delay between release of previous key and this press

            totalHoldTime += holdTime
            totalSeekTime += seekTime

            for i in range(0, ngram):
                ngramDuration[i] += seekTime + holdTime         # add this seek time to all the counters
                ngramName[i] = (((ngramName[i] << 8) | keyCode) & ngramNameMask)  # update label by shifting keyCode left each time

                if i == n and keystrokeCount >= ngram:
                    ngramVector[ngramName[i]] = ngramDuration[i]
                    totalNgramTime += ngramDuration[i]
                    if debugging: print('Found nGram ', hex(ngramName[i]), '(', prettyNgram(ngramName[i]), ')', ' duration: ', ngramDuration[i])
                    ngramCount += 1
                    ngramName[i] = 0                        # reset this feature
                    ngramDuration[i] = 0
            n += 1
            keystrokeCount += 1

            if debugging: input()

        averageHoldTime = totalHoldTime / keystrokeCount
        averageSeekTime = totalSeekTime / keystrokeCount

        if ngramCount > 0:
            averageNgramTime = totalNgramTime / ngramCount
        else:
            averageNgramTime = 0

        #discard outliers
        if(averageHoldTime < discardHoldTime and averageSeekTime < discardSeekTime and averageNgramTime > 0):

            print('User Data:', userCount)
            print('Found ngrams for user: ', ngramVector)
            print('User average hold time: ', int(averageHoldTime), 'ms')
            print('User average seek time: ', int(averageSeekTime), 'ms')
            print('User average ngram time: ', int(averageNgramTime), 'ms (n-gram of', ngram, ')')
            ngramVectors.append(ngramVector)

            featureVector.append(userCount)
            featureVector.append(int(averageHoldTime))
            featureVector.append(int(averageSeekTime))
            featureVector.append(int(averageNgramTime))

            featureVectors.append(featureVector)
        else:
            print('--- DISCARDING DATA ---')
            print('Found ngrams for user: ', ngramVector)
            print('User average hold time: ', int(averageHoldTime), 'ms')
            print('User average seek time: ', int(averageSeekTime), 'ms')
            print('User average ngram time: ', int(averageNgramTime), 'ms (n-gram of', ngram, ')')
            discardedUsers.append(userCount)
            discardCount += 1

        userCount += 1

print("K-Means Clustering Algorithm by Team Awesomesauce")
print("Discarded ", discardCount, " data points: ", discardedUsers)
print("Using Feature Vectors: ")
print(featureVectors)
print("Exporting to CSV")
dataUtils.generateFeatureCsv(featureVectors,"jonstest",['userID','avgHoldTime','avgSeekTime','averageNgramTime'])


#data = arff.load(open('data_-15_16-19_256.arff'), 'rb')

#print(data)
