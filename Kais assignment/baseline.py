import sys as os
import pandas as pd
import numpy as np
import random


def trainingdata(filetrain):
    newtag  = {} 
    dictionaire = {}     
    the_tags = [] 
    vocabfreq = {} 
    pair_freq = {} 
    
    with open(filetrain) as data:
        for line in data:
            split = line.split()
            ptag = 'BOSE'
            for unit in split:
                word, slice, tag = unit.partition('/')
                if tag not in the_tags:
                    the_tags.append(tag)
                    vocabfreq[tag] = 1
                else:
                    vocabfreq[tag] += 1
                tagPair = ptag + '_' + tag
                
                if tagPair not in pair_freq.keys():
                    pair_freq[tagPair] = 1
                else:
                    pair_freq[tagPair] += 1
                    
                ptag = tag
        data.seek(0)
        for line in data:
            split = line.split()
            for unit in split:
                word, slice, tag = unit.partition('/')
                
                if word not in dictionaire.keys():
                    dictionaire[word] = dict.fromkeys(the_tags, 0)
                    dictionaire[word][tag] += 1
                    
                else:
                    dictionaire[word][tag] += 1
        
        
    return the_tags, vocabfreq, pair_freq, newtag, dictionaire


def testdata(testFile):
    
    taglist = {}
    
    with open(testFile) as data:
        
        for line in data:
            split = line.split()
            sent = []
            tags = []
            for unit in split:
                word, slice, tag = unit.partition('/')
                sent.append(word)
                tags.append(tag)
            
            sent = tuple(sent) 
            taglist[sent] = tags
        
    sent_from_test = taglist.keys()
    soltagstest = taglist.values()
    
    return sent_from_test, soltagstest, taglist


def baseline_algo(sent_from_test, the_tags, dictionaire):
    
    word_list = dictionaire.keys()
    
    tag_predicted_dictionaire = {}
    
    for sentNum, sent in enumerate(sent_from_test):
        tag_predic = ['']*len(sent)
        for wordNum, word in enumerate(sent):
            
            if word in word_list:
                word_tag_dict = dictionaire[word]
                max_tag = max(word_tag_dict, key=word_tag_dict.get)
            else:
                max_tag = random.choice(the_tags)
            
            tag_predic[wordNum] = max_tag
        
        sent = tuple(sent)
        tag_predicted_dictionaire[sent] = tag_predic
    
    return tag_predicted_dictionaire


def accuracy(tag_predicted_dictionaire, true_tag_dic_list):
    total_tags = 0
    true_tag_count = 0
    wrong_predic = []
    
    for sent in tag_predicted_dictionaire.keys():
        true_tag_list = true_tag_dic_list[sent]
        predic_tag_list = tag_predicted_dictionaire[sent]
        for wordIndex, (true_tag, predic_tag) in enumerate(zip(true_tag_list, predic_tag_list)):
            total_tags += 1
            if predic_tag == true_tag:
                true_tag_count += 1
            else:
                wrong_predic.append([sent[wordIndex], predic_tag, true_tag])
    
    acc = float(true_tag_count/total_tags)
    
    return acc, wrong_predic



def model_predicted_tags(tag_predicted_dictionaire):
    
    output = "POS.test.out"
    file = open(output, "w")

    for sent, sent_tag in tag_predicted_dictionaire.items():
        for word, tag in zip(sent, sent_tag):
            file.write(word + '/' + tag + ' ')
        file.write('\n')

    file.close()


def wrong_predictions(wrong_predic):
    print('Wrong predictions:' + '\n')
    for pred in wrong_predic[:3]:
        print('Token: ' + pred[0] + ' ')
        print('Model: ' + pred[1] + ' ')
        print('true: ' + pred[2] + '\n')



def main():

    filetrain = os.argv[1]
    testFile = os.argv[2]


    the_tags, vocabfreq, pair_freq, newtag, dictionaire = trainingdata(filetrain)


    sent_from_test, soltagstest, true_tags_dic_list = testdata(testFile)

 
    baseline_predic = baseline_algo(sent_from_test, the_tags, dictionaire)

    
    acc, wrong_predic = accuracy (baseline_predic, true_tags_dic_list)
    print('Baseline model accuraccy:', acc)

    wrong_predictions(wrong_predic)

if __name__ == "__main__":
    main()


