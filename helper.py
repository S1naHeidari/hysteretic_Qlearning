import numpy as np


def getKey(dictOfElements, valueToFind):
    if dictOfElements: 
        listOfItems = dictOfElements.items()
        for item in listOfItems:
            if item[1] == valueToFind:
                return item[0]
    else:
        return 0


def progress(iteration, total, prefix='Progress:'):
    suffix=''
    decimals=1
    length=45
    fill='█'
    printEnd="\r"

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    
    if iteration == total:
        print()
