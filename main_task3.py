from __future__ import print_function
import sys
import re
import math
import numpy as np

from operator import add

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: wordcount <file> <output> ", file=sys.stderr)
    #     exit(-1)

    def featureVec(tup):
        vect = np.zeros(5000)
        for p in tup:
            i = int(p[0])
            vect[i] = vect[i] + p[1]
        return vect

    sc = SparkContext(appName="termProject_task2")
    lines = sc.textFile(sys.argv[1], 1)
    # lines = sc.textFile("trainingdata.csv")
    # filter data and transfer data into into (docId, txt) pair
    validLines = lines.map(lambda x: x.split(',')).filter(lambda p: len(p) == 6)
    keyAndText = validLines.map(lambda x: (x[0], x[1], x[5]))
    # use regular expression to transfer data text into list of words
    regex = re.compile('[^a-zA-Z]')
    # remove all non letter words and word length less or equals to 2
    keyAndListOfWords = keyAndText.map(lambda x: (x[1], regex.sub(' ', x[2]).lower().split()))
    # print(keyAndListOfWords)
    allWords = keyAndListOfWords.flatMap(lambda x: [(word, 1) for word in x[1]]).filter(lambda x: (len(x[0]) > 2))

    # get word counts using reduce by key
    allCounts = allWords.reduceByKey(add)
    # print(allCounts.take(5))
    # print(allCounts.count())
    # get 5k words based on the frequency we got last step
    topWords = allCounts.top(5000, key=lambda x: x[1])
    # print(topWords[:5])
    # print(len(topWords))
    # create an RDD with 5000 position
    fiveK = sc.parallelize(range(5000))
    # now we get our top 5k dictionary (word, pos) pair
    dictionary = fiveK.map(lambda x: (topWords[x][0], x))
    # print(dictionary.take(100))
    # now we get (word, docId) pairs for each (docId, [word1, word2, ...])
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    # note because some documents have word that is not in the dictionary
    allDictionaryWords = dictionary.join(allWords)
    # now we drop the actual word itself to get a set of [(docID,dictionaryPos), 1] pairs
    justDocAndPos = allDictionaryWords.map(lambda x: [(x[1][1], x[1][0]), 1])
    # now we get (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey().map(lambda x: (x[0], len(x[1])))

    # now we get sorted (docID, [pos, frequency]) pair
    wordsInDoc = allDictionaryWordsInEachDoc.map(lambda x: [x[0][0], (x[0][1], x[1])])
    wordsInDocSorted = wordsInDoc.groupByKey().map(lambda x: (x[0], sorted(x[1])))

    # now we get word count for each doc, first we get (docId, 1) pairs
    # then we calculate word count and get (docId, count) pair
    docCountTotal = allWords.map(lambda x: (x[1], 1)).reduceByKey(add)
    # we should get (docID, (dicPos, occur), docCount) pair and get (docId, (dicPos, Freq)) pair
    docWithCount = wordsInDocSorted.join(docCountTotal).flatMap(lambda x: ((x[0], j, x[1][1]) for j in x[1][0]))
    # print(docWithCount.take(1))
    docWithFreq = docWithCount.map(lambda x: [x[0], (x[1][0], float(x[1][1]) / float(x[2]))]).groupByKey().map(
        lambda x: (x[0], sorted(x[1])))
    # get the Feature vector for each doc
    docWithFreqVect = docWithFreq.map(lambda x: (x[0], featureVec(x[1])))
    # print(docWithFreqVect.take(1))
    # task2
    # using k-means to cluster these data points
    parsedData = docWithFreqVect.map(lambda x: x[1])

    # Build the model (cluster the data)
    model = KMeans.train(parsedData, 3, maxIterations=10, initializationMode="random")

    # get class and frequent vector
    regenum = re.compile('[^0-9]')
    # keyWithClass = keyAndText.map(lambda x: (x[1], x[0]))
    # classWithFreq = keyWithClass.join(docWithFreqVect).map(lambda x: x[1]).map(lambda x: (regenum.sub('', x[0]), x[1]))

    # testing
    # testLines = sc.textFile("testdata.csv")
    testLines = sc.textFile(sys.argv[2], 1)
    # filter data and transfer data into into (docId, txt) pair
    testValidLines = testLines.map(lambda x: x.split(',')).filter(lambda p: len(p) == 6)
    testKeyAndText = testValidLines.map(lambda x: (x[0], x[1], x[5]))
    # use regular expression to transfer data text into list of words
    # remove all non letter words
    testKeyAndListOfWords = testKeyAndText.map(lambda x: (x[1], regex.sub(' ', x[2]).lower().split()))
    testAllWords = testKeyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
    # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
    # note because some documents have word that is not in the dictionary
    testAllDictionaryWords = dictionary.join(testAllWords)
    # now we drop the actual word itself to get a set of [(docID,dictionaryPos), 1] pairs
    testJustDocAndPos = testAllDictionaryWords.map(lambda x: [(x[1][1], x[1][0]), 1])
    # now we get (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
    testAllDictionaryWordsInEachDoc = testJustDocAndPos.groupByKey().map(lambda x: (x[0], len(x[1])))

    # now we get sorted (docID, [pos, frequency]) pair
    testWordsInDoc = testAllDictionaryWordsInEachDoc.map(lambda x: [x[0][0], (x[0][1], x[1])])
    testWordsInDocSorted = testWordsInDoc.groupByKey().map(lambda x: (x[0], sorted(x[1])))

    # now we get word count for each doc, first we get (docId, 1) pairs
    # then we calculate word count and get (docId, count) pair
    testDocCountTotal = testAllWords.map(lambda x: (x[1], 1)).reduceByKey(add)
    # we should get (docID, (docPos, occur), docCount) pair and get (docId, (docPos, Freq)) pair
    testDocWithCount = testWordsInDocSorted.join(testDocCountTotal).flatMap(
        lambda x: ((x[0], j, x[1][1]) for j in x[1][0]))
    testDocWithFreq = testDocWithCount.map(lambda x: [x[0], (x[1][0], float(x[1][1]) / float(x[2]))]).groupByKey().map(
        lambda x: (x[0], sorted(x[1])))
    testDocWithFreqVect = testDocWithFreq.map(lambda x: (x[0], featureVec(x[1])))
    parsedTestData = testDocWithFreqVect.map(lambda x: x[1])

    def error(point):
        center = model.centers[model.predict(point)]
        return math.sqrt(sum([x ** 2 for x in (point - center)]))

    WSSSETEST = parsedTestData.map(lambda point: error(point)).reduce(add)
    print("Within Set Sum of Squared Error = " + str(WSSSETEST))

    # def predictCenter(point):
    #     center = model.centers[model.predict(point)]
    #     return center
    #
    # # get label and feature vector
    # testKeyWithClass = testKeyAndText.map(lambda x: (x[1], x[0]))
    # testClassWithFreq = testKeyWithClass.join(testDocWithFreqVect).map(lambda x: x[1]).map(
    #     lambda x: (regenum.sub('', x[0]), x[1]))
    # testClassWithCenter = testClassWithFreq.map(lambda x: (x[0], predictCenter(x[1])))
    # testClassCount = testClassWithCenter.map(lambda x: (x[0], 1)).reduceByKey(add)
    # print("Count for Class:", classCount.take(3))
    #
    # classCenterPairCount = classWithCenter.map(lambda x: ([x[0], x[1]], 1))\
    #     .groupBykey().map(lambda x: (x[0], len(x[1]))).collect()
    #
    # print("Count for each class and center pair", classCenterPairCount)

    sc.stop()


