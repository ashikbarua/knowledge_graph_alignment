# -*- coding: utf-8 -*-

import pandas as pd
import json
import csv
import glob
from csv import reader
import networkx as nx

nodesJSON = json.load(open('data/generated/nodes.txt','r'))
relationsJSON = json.load(open('data/generated/relations.txt','r'))
# graph_new = nx.read_gpickle('data/generated/yago_facts.pickle')

# location = 'data/dbpedia/real'

def mappingNodeDBpedia(location):

    notFound = []
    nF = 0
    f = 0
    for csvFile in glob.glob(location+'/*.csv'):
        print('Reading file: {}'.format(csvFile))
        dbData = pd.read_csv(csvFile,',')
        fileDBpedia = open('data/dbpedia/'+csvFile.split('/')[-2]+'/'+csvFile.split('/')[-1][:-4]+'_DB2Yago_node.csv','w+', newline='')
        writerDBpedia = csv.writer(fileDBpedia)
        rows = []
        uniqueSet = set()
        writerDBpedia.writerow(['Yago ID','Yago Name', 'DBPedia ID', 'DBPedia Name'])
        for i in range(len(dbData)):
            row = dbData.iloc[i,:]
            subject = '<'+row['sub'][4:]+'>'
            object_ = '<'+row['obj'][4:]+'>'
            
            if (nodesJSON.get(subject) and nodesJSON.get(object_)):
                if subject not in uniqueSet:
                    rows.append([nodesJSON.get(subject), subject, row['sid'], row['sub']])
                    uniqueSet.add(subject)
                if object_ not in uniqueSet:
                    rows.append([nodesJSON.get(object_), object_, row['oid'], row['obj']])
                    uniqueSet.add(object_)
                f += 1
            else:
                if subject not in uniqueSet:
                    notFound.append([row['sid'], row['sub'], csvFile.split('/')[-1]])
                    uniqueSet.add(subject)
                if object_ not in uniqueSet:
                    notFound.append([row['oid'], row['obj'], csvFile.split('/')[-1]])
                    uniqueSet.add(object_)
                nF += 1

        writerDBpedia.writerows(rows)

    print('found'+str(f))
    print('nfound'+str(nF))
    fileDBpedia = open('data/yago/'+csvFile.split('/')[-2]+'/'+'NotFound_DB2Yago_node.csv','w+', newline='')
    writerDBpedia = csv.writer(fileDBpedia)
    writerDBpedia.writerows(notFound)
    print('Writing : {}'.format(csvFile))

def mappingRelDBpedia(location):
    DB2Yago = pd.read_csv('resources/linking/yd_relations/relations_dbpedia_sub_yago.tsv', '\t')

    relations = []
    notFound = []
    for csvFile in glob.glob(location+'/*.csv'):
        dbData = pd.read_csv(csvFile,',')
        for dbRel in set(dbData['pred']):
            dbRelName = dbRel.split(':')[1]
            row = getRelRow(dbRelName, DB2Yago)
            for r in row:
                yagoLabel = '<'+getYagoPredicate(r[0])+'>'
                if relationsJSON.get(yagoLabel):
                    print(relationsJSON.get(yagoLabel))
                    print(yagoLabel)
                    relations.append([relationsJSON.get(yagoLabel), yagoLabel, dbRel, set(dbData[dbData['pred']==dbRel]['pid'])])

                else:
                    notFound.append([dbRel, set(dbData[dbData['pred']==dbRel]['pid'])])

    fileDBpedia = open(
        'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'Relations' + '_DB2Yago.csv',
        'w+', newline='')
    writerDBpedia = csv.writer(fileDBpedia)
    writerDBpedia.writerow(['Yago Rel ID', 'Yago Relation', 'DBPedia Relation', 'DBPedia Rel ID'])
    writerDBpedia.writerows(relations)


    fileDBpedia = open(
        'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'NotFound_DB2Yago_relations.csv',
        'w+', newline='')
    writerDBpedia = csv.writer(fileDBpedia)
    writerDBpedia.writerow(['DBPedia Relation', 'DBPedia Relation ID'])
    writerDBpedia.writerows(notFound)

def getRelRow(relationName, DB2Yago):
    rows = []
    for i in range(len(DB2Yago)):
        row = DB2Yago.iloc[i,:]
        if relationName in row[0]:
            rows.append(row[[1]])
    return rows

def getYagoPredicate(dbLabel):
    if dbLabel.split(':')[1][-1]=='-':
        return dbLabel.split(':')[1][:-1], True
    else:
        return dbLabel.split(':')[1], False


def filterTriples(location):

    DB2Yago = pd.read_csv('resources/linking/yd_relations/relations_dbpedia_sub_yago.tsv', '\t')
    notfound = []
    foundNfound = []
    for csvFile in glob.glob(location + '/*.csv'):
        print(csvFile)
        dbData = pd.read_csv(csvFile, ',')
        rows = []
        notInFile = []
        numMatch = 0
        numNotMatch = 0
        for i in range(len(dbData)):
            row = dbData.iloc[i, :]
            subject = '<' + row['sub'][4:] + '>'
            object_ = '<' + row['obj'][4:] + '>'
            dbRelName = row['pred']
            dbRelName = dbRelName.split(':')[1]
            if (nodesJSON.get(subject) and nodesJSON.get(object_)):
                relRow = getRelRow(dbRelName, DB2Yago)
                for r in relRow:
                    yagoPredicate, inverse = getYagoPredicate(r[0])
                    yagoPredicate = '<'+yagoPredicate+'>'
                    if relationsJSON.get(yagoPredicate):
                        numMatch += 1
                        if(inverse):
                            rows.append([nodesJSON.get(object_), object_, relationsJSON.get(yagoPredicate), yagoPredicate,
                                 nodesJSON.get(subject), subject, row['class']])
                        else:
                            rows.append([nodesJSON.get(subject), subject, relationsJSON.get(yagoPredicate), yagoPredicate,
                                         nodesJSON.get(object_), object_, row['class']])
                    else:
                        numNotMatch +=1
                        notfound.append(row)
                        notInFile.append(row)
                if (len(relRow)==0):
                    numNotMatch += 1
                    notfound.append(row)
                    notInFile.append(row)
            else:
                numNotMatch+=1
                notfound.append(row)
        foundNfound.append([csvFile, numMatch, numNotMatch])

        fileDBpedia = open(
            'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'yago_' + csvFile.split('/')[-1][:-4]
            + '.csv','w+', newline='')
        writerDBpedia = csv.writer(fileDBpedia)
        writerDBpedia.writerow(list(dbData.columns))
        writerDBpedia.writerows(rows)
        fileDBpedia.close()

        fileDBpedia = open(
            'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'not_in_yago'+ csvFile.split('/')[-1][:-4]+ '.csv',
            'w+', newline='')
        writerDBpedia = csv.writer(fileDBpedia)
        writerDBpedia.writerow(list(dbData.columns))
        writerDBpedia.writerows(notInFile)
        fileDBpedia.close()

    fileDBpedia = open(
        'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'MatchNoMatch.csv',
        'w+', newline='')
    writerDBpedia = csv.writer(fileDBpedia)
    writerDBpedia.writerow(['File name', 'Matchs', 'Not matches'])
    writerDBpedia.writerows(foundNfound)
    fileDBpedia.close()

    fileDBpedia = open(
        'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'notFoundInYago.csv',
        'w+', newline='')
    writerDBpedia = csv.writer(fileDBpedia)
    writerDBpedia.writerow(list(dbData.columns))
    writerDBpedia.writerows(notfound)
    fileDBpedia.close()


# Extracting triple predicate in yago kg by inputting subject and object. Also, matching with
# equivalent predicate in dbo and finding up a correlation between the predicates from the two kgs.

def predicateMatch(location):

    columns = ['dbo_predicate', 'yago_predicate', 'file_name']
    df = pd.DataFrame(columns=columns)

    for csvFile in glob.glob(location + '/*.csv'):
        dbData = pd.read_csv(csvFile, ',')
        rows = []
        yago_pred_set = set()
        dbo_pred_set = set()
        for i in range(len(dbData)):
            row = dbData.iloc[i, :]
            subject = '<' + row['sub'][4:] + '>'
            object_ = '<' + row['obj'][4:] + '>'
            yago_pred = graph_new.get_edge_data(subject, object_)
            class_ = row['class']
            if (yago_pred==None):
                rows.append([subject, object_, "None", row['pred'], class_])
                yago_pred_set.add("None")
                dbo_pred_set.add(row['pred'])
            else:
                rows.append([subject, object_, yago_pred, row['pred'], class_])
                yago_pred_set.add(yago_pred['predicate'])
                dbo_pred_set.add(row['pred'])

        for col1 in list(yago_pred_set):
            for col2 in list(dbo_pred_set):
                r = pd.DataFrame([[col2, col1, csvFile.split('/')[-1]]], columns=columns)
                df = df.append(r)


        fileDBpedia = open(
            'resources/generated_data/' + csvFile.split('/')[-2] + '/' + 'equi_predicate_' + csvFile.split('/')[-1][
                                                                                         :-4] + '.csv',
            'w+', newline='')
        writerDBpedia = csv.writer(fileDBpedia)
        writerDBpedia.writerow(['subject', 'object', 'yago_predicate', 'dbo_predicate', 'dbo_class'])
        for row in rows:
            writerDBpedia.writerow(row)
        fileDBpedia.close()

    df.to_csv('resources/generated_data/unique_equivalent_predicate_' + csvFile.split('/')[-2] + '.csv')


# Mapping dbo to yago by manually passing the predicates

def filterTripleRevised(csvFile, yagoPredicate, writeFlag, reverse):

    dbData = pd.read_csv(csvFile, ',')
    rowsYago = []
    notFound = []
    rowsDbp = []
    for i in range(len(dbData)):
        row = dbData.iloc[i, :]
        subject = '<' + row['sub'][4:] + '>'
        object_ = '<' + row['obj'][4:] + '>'

        if (nodesJSON.get(subject) and nodesJSON.get(object_)):

            if (reverse):
                rowsYago.append([nodesJSON.get(object_), object_, relationsJSON.get(yagoPredicate), yagoPredicate,
                              nodesJSON.get(subject), subject, row['class']])
            else:
                rowsYago.append([nodesJSON.get(subject), subject, relationsJSON.get(yagoPredicate), yagoPredicate,
                             nodesJSON.get(object_), object_, row['class']])
            rowsDbp.append([row['sid'], row['sub'], row['pid'], row['pred'], row['oid'], row['obj'], row['class']])
        else:
            notFound.append([nodesJSON.get(object_), object_, relationsJSON.get(yagoPredicate), yagoPredicate,
                              nodesJSON.get(subject), subject, row['class']])


    if (writeFlag):
        fileYago = open(
            'data/yago/' + csvFile.split('/')[-2] + '/' + 'yago_' + csvFile.split('/')[-1],
            'w+', newline='')
        writerYago = csv.writer(fileYago)
        writerYago.writerow(['sid', 'sub', 'pid', 'pred', 'oid', 'obj', 'class'])
        for row in rowsYago:
            writerYago.writerow(row)
        fileYago.close()

        fileYagoNotFound = open(
            'data/yago/' + csvFile.split('/')[-2] + '/' + 'yago_not_found_' + csvFile.split('/')[-1],
            'w+', newline='')
        writerYago = csv.writer(fileYagoNotFound)
        writerYago.writerow(['sid', 'sub', 'pid', 'pred', 'oid', 'obj', 'class'])
        for row in notFound:
            writerYago.writerow(row)
        fileYagoNotFound.close()

        fileDbp = open(
            'data/dbpedia_subset/' + csvFile.split('/')[-2] + '/' + 'dbp_sub_' + csvFile.split('/')[-1],
            'w+', newline='')
        writerYago = csv.writer(fileDbp)
        writerYago.writerow(['sid', 'sub', 'pid', 'pred', 'oid', 'obj', 'class'])
        for row in rowsDbp:
            writerYago.writerow(row)
        fileDbp.close()


csvFile = "data/dbpedia/real/place_of_birth.csv"
yagoPredicate = '<wasBornIn>'
filterTripleRevised(csvFile, yagoPredicate, writeFlag=1, reverse=0)


location='data/dbpedia/synthetic'
# mappingNodeDBpedia(location)
# mappingRelDBpedia(location)
# filterTriples(location)
#
# for csvFile in glob.glob(location + '/*.csv'):
#     print(csvFile)

filelist=['place_of_birth.csv', 'place_of_death.csv', 'derived_filtered_nationality_train_25facts.csv',
          'institution.alm.csv']
predlist = ['<wasBornIn>', '<diedIn>', '<isCitizenOf>', '<isAffiliatedTo>']
for i in range(len(filelist)):
    csvFile = "data/dbpedia/real/"+filelist[i]
    yagoPredicate = predlist[i]
    filterTripleRevised(csvFile, yagoPredicate, writeFlag=1, reverse=0)

filelist = ['cross_US_Presidents_vs_First_Lady.csv', 'predpath_state_capital.csv',
            'predpath_company_president.csv', 'Player_vs_Team_NBA.csv', 'cross_Movies_vs_Directors.csv']

predlist = ['<isMarriedTo>', '<hasCapital>', '<owns>', '<isAffiliatedTo>', '<directed>']
revlist = [0, 0, 1, 0, 1]
for i in range(len(filelist)):
    csvFile = "data/dbpedia/synthetic/"+filelist[i]
    yagoPredicate = predlist[i]
    filterTripleRevised(csvFile, yagoPredicate, writeFlag=1, reverse=revlist[i])