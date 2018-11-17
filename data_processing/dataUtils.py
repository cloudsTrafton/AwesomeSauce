# Provides utilities for processing data, including 9


# Writes the features to a CSV file.
# takes in a list of lists, i.e. [[1,2,3],[4,5,6]] and writes to a file with each feature vector in its own line
# Parameters  -
# featureVectorsArray: array of arrays with each second level array representing a feature vectors
# fileName: the name given to the file, this does not include the extension or directory, since that is assigned
# saves the file under the featureSets directory
def generateFeatureCsv(featureVectorsArray, fileName):
    fileName = "../feature_sets/" + fileName + ".csv"
    featureVectorsCsv = open(fileName, "w+")
    for vector in featureVectorsArray:
        stringifiedVector = str(vector).replace("[", "").replace("]", "")
        featureVectorsCsv.write(stringifiedVector + '\n')