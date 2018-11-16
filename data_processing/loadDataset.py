import arff

import os.path

# assume no more than 64-bit keycodes
def prettyNgram(keycodes):
    ret = ''
    ret += chr((keycodes & 0xFF000000) >> 24)
    ret += chr((keycodes & 0x00FF0000) >> 16)
    ret += chr((keycodes & 0x0000FF00) >> 8)
    ret += chr((keycodes & 0x000000FF))
    return ret.strip()


debugging = True

# Dataset Features

# Rank of N-gram we use for features.
# This is the number of consecutive characters typed that we measure
# Needs to be 4 or less.
ngram = 2

datasetRoot = '../age_anonymized/'
datasetFolders = [datasetRoot + '-15/', datasetRoot + '16-19/', datasetRoot + '20-29/', \
                datasetRoot + '30-39/', datasetRoot + '40-49/', datasetRoot + '50+/']

#datasetFolders = [datasetRoot + 'test/']

print("Reading files in: ", datasetFolders)
print("Found the following raw dataset files:")
fileList = []
for folder in datasetFolders:
    files = os.listdir(folder)
    for file in files:
        fullName = os.path.join(folder, file)
        if os.path.isfile(fullName) and file.endswith('.csv'):
            fileList.append(fullName)

ngramVectors = []
holdtimeVectors = []
seektimeVectors = []
userCount = 0

print("Processing into feature vectors...")
for file in fileList:
    print(file)

    # Looking at single user:
    totalHoldTime = 0.0
    totalSeekTime = 0.0
    averageHoldTime = 0.0
    averageSeekTime = 0.0

    with open(file, "r") as f:
        content = f.readlines()
        print(content)

        keystrokeCount = 1                         # note: starts at 1, this is for calculating average
        n = 0                                       # counter for ngrams
        # temp lists to hold intermediate values before placing them in a feature vector
        #keyCode = []
        #holdTime = []
        #seekTime = []

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
            keyCode = int(row[0])                   # keycode (ASCII)
            holdTime = int(row[1])                  # how long we hold this key down
            seekTime = int(row[2])                  # delay between release of previous key and this press

            totalHoldTime += holdTime
            totalSeekTime += seekTime

            for i in range(0, ngram):
                ngramDuration[i] += seekTime + holdTime     # add this seek time to all the counters
                ngramName[i] = ((ngramName[i] << 8) | keyCode)  # update label by shifting keyCode left each time
                # TODO: ngramName needs to be 0 for all ngramName[i]
                if i == n and keystrokeCount >= ngram:
                    ngramVector[ngramName[i]] = ngramDuration[i]
                    print('Found nGram ', hex(ngramName[i]), '(', prettyNgram(ngramName[i]), ')', ' duration: ', ngramDuration[i])
                    ngramName[i] = 0                        # reset this feature
                    ngramDuration[i] = 0
            n += 1
            keystrokeCount += 1

            if debugging: input()

        averageHoldTime = totalHoldTime / keystrokeCount
        averageSeekTime = totalSeekTime / keystrokeCount
        print('User Data:', userCount)
        print('Found ngrams for user: ', ngramVector)
        print('User average hold time: ', averageHoldTime)
        print('User average seek time: ', averageSeekTime)
        ngramVectors.append(ngramVector)
        userCount += 1



print("K-Means Clustering Algorithm by Team Awesomesauce")
data = arff.load(open('data_-15_16-19_256.arff'), 'rb')

print(data)