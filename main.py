import nltk # natural language toolkit
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
import numpy as np
import json
import pandas as pd

############################################################################################
##############################PREPROCESAMENT################################################
############################################################################################

with open("intents.json") as file:
    data = json.load(file)

dictionary = []
classes = []
classes_with_num = []


#####CREACIO DE DICCIONARI##################################################################
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_array = nltk.word_tokenize(pattern)  #funcio que concatena les paraules en una llista
        dictionary.extend(word_array) #vector amb totes les paraules 

    classes.append(intent["class"])




dictionary = [stemmer.stem(word.lower()) for word in dictionary] #process de "stemming" : quedarse amb el cor de les paraules

dictionary = sorted(list(set(dictionary)))
classes = sorted(classes)

dictionary = np.array(dictionary)
classes = np.array(classes)

    
#####  TRACTAMENT DELS EXEMPLES   ##########################################################

input_x = [] #features
output_y = [] #labels


#creem un vector amb totes les frases y les corresponents classes
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentence = nltk.word_tokenize(pattern)
        words = [stemmer.stem(word) for word in sentence if word not in "?"] #Steem de input_x
        input_x.append(words) 
        output_y.append(intent["class"])



D, = dictionary.shape #numero de paraules en el diccionari (dimensions del vector)
M, = classes.shape # numero de possibles classificacions 
N = len(input_x) # numero dexemples per entrenar el model


output_y_nums = np.zeros((N,1))


for i in range(N):
    for j in range(M):
        if output_y[i]==classes[j]:
            output_y_nums[i] = j



########################################################################################
#Creem una "bag of words".
#Methode "One Hot Encoding" tenim la paraula?->1; no hi ha la paraula ->0.
########################################################################################

training = np.zeros((N, D)) 
output = output_y_nums

for j, sentence in enumerate(input_x, 0):

    bag_of_words = np.zeros(len(dictionary))

    for i in range(D):
        word = dictionary[i]
        if word in sentence:
            bag_of_words[i] = 1
        else:
            bag_of_words[i] = 0

    training[j] = bag_of_words

output_with_nums = output_y_nums



np.savetxt('test.csv', training, delimiter=',', fmt='%i')









############################################################################################
###########################  MODEL DE DEEP LEARNING  #######################################
############################################################################################


import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu, input_dim=D))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(M, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.summary()

try:
    new_model = tf.keras.models.load_model('chatbot-mauri.model')


except:
    model.fit(training, output, epochs=300)
    model.save('chatbot-mauri.model')
    new_model = model





############################################################################################
#######################################  START CHATBOT #####################################
############################################################################################


print("\n\n\n\n\n\n\n\n\n"
"############################################################################################\n"
"############################################################################################\n"
"############################################################################################\n"
"#######################################   CHATBOT    #######################################\n"
"############################################################################################\n"
"############################################################################################\n"
"############################################################################################\n"
)

import random
print("Bienvenido al chatbot-mauri, (escribe `salir` para parar el programa)\n")
print("Hablale al bot: ")
while True:

    question = input("Tu: ")
    if question.lower() == "salir":
        break

    if question == "":
        print("Bot: ", "Hay alguien ahí?")
    else:
        question = nltk.word_tokenize(question)
        question = [stemmer.stem(word.lower()) for word in question if word not in "?" if word not in "!"if word not in ","]

        bag_of_words = np.zeros(len(dictionary))

        bag_of_words = bag_of_words.reshape(D,1).T

        for i in range(D):

            word = dictionary[i]
            if word in question:
                bag_of_words[0][i] = 1
            else:
                bag_of_words[0][i] = 0

        prediction = new_model.predict(bag_of_words)
        result = classes[np.argmax(prediction)]

        for classe in data["intents"]:
            if classe["class"] == result:
                responses = classe["responses"]

        print("Bot: ", random.choice(responses))
