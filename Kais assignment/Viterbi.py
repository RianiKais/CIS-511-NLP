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


def probability_tag_given_tag(tag, ptag, pair_freq):
    
    tagPair = ptag + '_' + tag
    
    if tagPair in pair_freq:
        paircount = pair_freq[tagPair]
        totalpairs= sum(pair_freq.values())
        probability = float(paircount/totalpairs)
    else:
        probability = float(1/len(pair_freq.keys()))
    
    return probability


def probability_word_given_tag(word, tag, dictionaire, vocabfreq):
    
    if word in dictionaire:
        freqwordtag = dictionaire[word][tag]
        tagfreqtotal = vocabfreq[tag]
        probability = float(freqwordtag/tagfreqtotal)
        
    else:
        probability = float(1/len(vocabfreq.keys()))
        
    return probability


def maxfunc(Score, the_tags, current_tag, p_word, pair_freq):
    max_prob = 0
    max_tag = ''
    for tag in the_tags:
        prob = Score[(tag,p_word)] * probability_tag_given_tag(current_tag, tag, pair_freq)
        if prob > max_prob:
            max_prob = prob
            max_tag = tag
    
    return max_prob, max_tag


def max_prob_tag(Score, the_tags, word):
    max_prob = 0
    max_tag = ''
    for tag in the_tags:
        prob = Score[(tag,word)]
        if prob >= max_prob:
            max_prob = prob
            max_tag = tag
            
    return max_tag


def viterbi_algo(sent_from_test, the_tags, vocabfreq, pair_freq, dictionaire):
    
    tag_predicted_dictionaire = {}
    
    tag_word_tuple = []
    word_list = dictionaire.keys()
    for word in word_list:
        for tag in the_tags:
            tag_word_tuple.append((tag, word))
    
    for num, sent in enumerate(sent_from_test):
        tag_predic = ['']*len(sent)
        Score = dict.fromkeys(tag_word_tuple, 0)
        back_tag_Ptr = dict.fromkeys(tag_word_tuple, 0)

        for tag in the_tags:
            probability_word_tag = probability_word_given_tag(sent[0], tag, dictionaire, vocabfreq)
            probability_tag_tag = probability_tag_given_tag(tag,'BOSE', pair_freq)
            Score[(tag, sent[0])] =  probability_word_tag * probability_tag_tag
            back_tag_Ptr [(tag, sent[0])] = None
        p_word = sent[0]
        for word in sent[1:]:
            for tag in the_tags:
                max_prob, max_P_tag = maxfunc(Score, the_tags, tag, p_word, pair_freq)
                probability_word_tag = probability_word_given_tag(word, tag, dictionaire, vocabfreq)
                Score[(tag, word)] = probability_word_tag * max_prob
                back_tag_Ptr [(tag, word)] = max_P_tag
            p_word = word

        end_of_sent = sent[len(sent)-1]
        if end_of_sent in word_list:
            last_tag_predic = max_prob_tag(Score, the_tags, end_of_sent)
        else:
            last_tag_predic = random.choice(the_tags)
        tag_predic[len(sent)-1] = last_tag_predic 
            
        for index, word in enumerate(reversed(sent[:-1])):
            
            next_word = sent[len(sent)-1-index]
            
            if next_word == end_of_sent:
                next_tag_pred = last_tag_predic
            elif next_word in word_list:
                next_tag_pred = max_prob_tag(Score, the_tags, next_word)
            else:
                next_tag_pred = random.choice(the_tags)
            
            tag_predic[len(sent)-2-index] = back_tag_Ptr [(next_tag_pred, next_word)]
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


    tag_predicted_dictionaire = viterbi_algo(sent_from_test, the_tags, vocabfreq, pair_freq, dictionaire)


    acc, wrong_predic = accuracy (tag_predicted_dictionaire, true_tags_dic_list)
    print('Viterbi model accuraccy:', acc)

    wrong_predictions(wrong_predic)

    model_predicted_tags(tag_predicted_dictionaire)


if __name__ == "__main__":
    main()


