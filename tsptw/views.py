from django.http import HttpResponse
from django.shortcuts import render
import json
from http import HTTPStatus
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
import random
import datetime as dt
from datetime import timedelta, datetime
import time
import requests

def Json_decode(result):
    data = {
        'result' : result
    }

    data = json.dumps(data)
    Json_decode.json = data

def Data():
    excel_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data.xlsx')
    dfTime = pd.read_excel(excel_file, sheet_name='time', usecols=[1])
    dfDist = pd.read_excel(excel_file, sheet_name='dist', usecols=range(1,21))
    dfLoc  = pd.read_excel(excel_file, sheet_name='loc', usecols=[0,1,2])
    times = dfTime.to_numpy()
    Data.dist = dfDist.to_numpy()
    Data.loc = dfLoc.to_numpy()

    # data waktu dalam bentuk string
    Data_time = []
    for i in range (len(times)):
        time = ''.join(map(str, times[i]))
        time = datetime.strptime(time, '%H:%M:%S').time()
        Data_time.append(time)
    Data.time = Data_time

# generate chromosome 
def Chromosome(destination, popSize):
    chrom = []
    for i in range (popSize):
        destination = shuffle(destination)
        chrom.append(destination)
    Chromosome.result = np.array(chrom)

# memilih parent secara random dari chromosome
def Choosing_Parent(chromosome, offspring):
   parent = random.sample(range(0, chromosome.shape[0]), offspring)
   return chromosome[parent] 

# pmx crossover
def PMX_Crossover(chromosome, crossover_rate):
    popSize = len(chromosome[0])
    offspring = (round(crossover_rate * popSize) + 1)
    parents = Choosing_Parent(chromosome, offspring).tolist()

    def recursion1(tempChild, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross):
        child = np.array([0 for i in range(len(parent1))])
        for i,j in enumerate(tempChild[:firstCrossPoint]):
            c = 0
            for relation in relations:
                if j == relation[0]:
                    child[i] = relation[1]
                    c = 1
                    break

            if c == 0:
                child[i] = j

        j = 0
        for i in range(firstCrossPoint,secondCrossPoint):
            child[i] = parent2MiddleCross[j]
            j += 1

        for i,j in enumerate(tempChild[secondCrossPoint:]):
            c = 0
            for relation in relations:
                if j == relation[0]:
                    child[i + secondCrossPoint] = relation[1]
                    c = 1
                    break

            if c == 0:
                child[i + secondCrossPoint] = j

        childUnique = np.unique(child)
        if len(child) > len(childUnique):
            child = recursion1(child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        return(child)

    def recursion2(tempChild, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross):
        child = np.array([0 for i in range(len(parent1))])
        for i,j in enumerate(tempChild[:firstCrossPoint]):
            c = 0
            for relation in relations:
                if j == relation[1]:
                    child[i] = relation[0]
                    c = 1
                    break

            if c == 0:
                child[i] = j

        j = 0
        for i in range(firstCrossPoint,secondCrossPoint):
            child[i] = parent2MiddleCross[j]
            j += 1

        for i,j in enumerate(tempChild[secondCrossPoint:]):
            c = 0
            for relation in relations:
                if j == relation[1]:
                    child[i + secondCrossPoint] = relation[0]
                    c = 1
                    break

            if c == 0:
                child[i + secondCrossPoint] = j

        childUnique = np.unique(child)
        if len(child) > len(childUnique):
            child = recursion1(child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        return(child)

    child = []
    crossOff1 = []
    crossOff2 = []
    for i in range(offspring - 1):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        firstCrossPoint = np.random.randint(0, len(parent1) - 2)
        secondCrossPoint = np.random.randint(firstCrossPoint + 1, len(parent1) - 1)

        parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
        parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

        temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]
        temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]

        relations = []
        for j in range(len(parent1MiddleCross)):
            relations.append([parent2MiddleCross[j], parent1MiddleCross[j]])

        child1 = recursion1(temp_child1, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        child2 = recursion2(temp_child2, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        
        crossOff1.append(child1)
        crossOff2.append(child2)
    PMX_Crossover.result = np.vstack((crossOff1, crossOff2))

# exchange mutation
def Exchange_Mutation(chromosome, mutation_rate):
    popSize = len(chromosome[0])
    offspring = (round(mutation_rate * popSize))
    child =[]
    parents = Choosing_Parent(chromosome, offspring)
    rand = np.random.randint(0, len(chromosome[0]), 2)
    parents[:, [rand[0], rand[1]]] = parents[:, [rand[1], rand[0]]]
    child = parents
    Exchange_Mutation.result = np.array(child)

def Offspring(chromosome, crossover, mutation):
    # penggabungan offspring
    offspring = np.vstack((chromosome, crossover, mutation))
    Offspring.offspring = np.array(offspring)

# mencari jarak dari offspring
def Dist_Offspring(s_user, offspring, distance):
    # melakukan transpose dari offspring
    off = np.transpose(offspring)
    result = []
    total = []
    for s in range(len(s_user)):
        # jarak awal
        total_dist = s_user[s]
        for i in range(len(off) - 1):
            dist = distance[(off[i]) - 1, (off[i + 1]) - 1]
            result.append(dist)
            # total jarak tiap kromosom
            total_dist += dist
    total.append(total_dist)
    # jarak dari lokasi ke lokasi
    Dist_Offspring.dist_result = np.array(result)
    # total jarak
    Dist_Offspring.total_dist = np.array(total)

# mencari best time dari offspring
def Time_Offspring(offspring, time):
    best_time = []
    for j in range(len(offspring)):
        time_kromosom = []
        for i in range(len(offspring[j])):
            bestTime = time[(offspring[j][i]) - 1]
            time_kromosom.append(bestTime)
        best_time.append(time_kromosom)
    Time_Offspring.time_result = np.array(best_time)

def Distance(userLoc, offspring):
    s_user = []
    dest = offspring[:, 0]
    for destination in range(len(dest)):
        origins = userLoc
        destinations = Data.loc[(dest[destination] - 1), 1]
        url = 'https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix'
        params = {
                  'origins':origins,
                  'destinations':destinations,
                  'travelMode':'driving',
                  'key':'AvmBky03OtFQedRJ0_v2_3DhIlZvlYO7GsucpEPY6FnK18CrPM55Z_117bgfPLGy'
                 }
        r = requests.get(url = url, params = params)
        data = r.json()
        # ekstrak data dari json
        dist = data['resourceSets'][0]['resources'][0]['results'][0]['travelDistance']
        s_user.append(dist)
    Distance.s_user = np.array(s_user)

def Visit_Time(destination, visit_duration, offspring):
    # relasi antara destinasi dan waktu kunjungannya
    relations = np.dstack((destination, visit_duration))
    relations = np.squeeze(relations)

    # generate waktu kunjungan dari offspring
    duration = np.array([[0 for col in range(len(offspring[0]))] for row in range(len(offspring))])
    for i in range(len(duration)):
        for j in range(len(duration[i])):
            findRow = np.where(relations == offspring[i, j])
            row = list(zip(findRow[0]))
            duration[i, j] = relations[row, 1] 
    Visit_Time.result = duration

def Penalty_0(s_user, time_user, offspring, distances, times):
    arrived_time = []

    # penalty awal
    penalty_0 = []
    for s in range(len(s_user)):
        # waktu perjalanan dari lokasi user
        t_0 = (s_user[s]/60) * 60
        t_0 = timedelta(minutes = t_0)

        # menghitung waktu tiba ke lokasi pertama
        start_time_0 = (dt.datetime.combine(dt.date(1, 1, 1), time_user) + t_0).time()
        arrived_time.append(start_time_0)

        # mengambil best time lokasi pertama
        off_time = times[s , 0]
        if off_time > start_time_0:
            p_0 = dt.datetime.combine(dt.date(1, 1, 1), off_time) - dt.datetime.combine(dt.date(1, 1, 1), start_time_0)
        elif start_time_0 > off_time:
            p_0 = dt.datetime.combine(dt.date(1, 1, 1), start_time_0) - dt.datetime.combine(dt.date(1, 1, 1), off_time)
        p_float = p_0.total_seconds()
        penalty_0.append(p_float)
    
    Penalty_0.arrived_time = arrived_time
    Penalty_0.penalty_0 = penalty_0

def Penalty_i(start_time_0, offspring, distances, times, visit_duration):
    off = np.transpose(offspring)
    times = np.transpose(times)
    visit_dur = np.transpose(visit_duration)
    arrived_time = start_time_0

    # menyimpan waktu kunjungan tiap lokasi
    Penalty_i.arrived = []
    Penalty_i.start = []
    Penalty_i.arrived.append(arrived_time)

    # penalty i
    penalty_i = []
    for i in range(len(off) - 1):
        # distance
        dist = distances[(off[i]) - 1, (off[i + 1]) - 1]
        trip_time = (dist/60) * 60
        trip_time = trip_time * timedelta(minutes=1)

        # best time lokasi selanjutnya
        best_time = times[i + 1]

        # visit duration
        visit_duration = visit_dur[i]
        visit_duration = visit_duration * timedelta(minutes=1)

        # start time
        start_time = []
        for at in range(min(len(arrived_time), len(visit_duration))):
            start_time_i = (dt.datetime.combine(dt.date(1, 1, 1), arrived_time[at]) + visit_duration[at]).time()
            start_time.append(start_time_i)
        Penalty_i.start.append(start_time)

        # arrived time
        arrived_time_i = []
        temp_penalty_i = []
        for index in range(min(len(start_time), len(trip_time), len(best_time))):
            arrived_time = (dt.datetime.combine(dt.date(1, 1, 1), start_time[index]) + trip_time[index]).time()
            arrived_time_i.append(arrived_time)
            # best time match
            if arrived_time > best_time[index]:
                p_i = dt.datetime.combine(dt.date(1, 1, 1), arrived_time) - dt.datetime.combine(dt.date(1, 1, 1), best_time[index])
            elif best_time[index] > arrived_time:
                p_i = dt.datetime.combine(dt.date(1, 1, 1), best_time[index]) - dt.datetime.combine(dt.date(1, 1, 1), arrived_time)
            p_i_float = p_i.total_seconds()
            temp_penalty_i.append(p_i_float)
        # penalty i
        penalty_i.append(temp_penalty_i)
        # replace arrived time to the new one
        arrived_time = arrived_time_i
        Penalty_i.arrived.append(arrived_time)

    # last location
    lastArrived = Penalty_i.arrived[len(Penalty_i.arrived) - 1]
    lastVisit = visit_dur[len(visit_dur) - 1]
    lastVisit = lastVisit * timedelta(minutes=1)
    last_start = []
    for lastLoc in range(min(len(lastArrived), len(lastVisit))):
        last_start_time = (dt.datetime.combine(dt.date(1, 1, 1), lastArrived[lastLoc]) + lastVisit[lastLoc]).time()
        last_start.append(last_start_time)
    Penalty_i.start.append(last_start)

    # total penalty i
    total = [sum(x) for x in zip(*penalty_i)]
    Penalty_i.penalty_i = total         

def Penalty(penalty_0, penalty_i):
    total_penalty = [x + y for x, y in zip(penalty_0, penalty_i)]
    Penalty.penalty = total_penalty

def Fitness(distances, penalty):
    fitness = []
    for off in range(len(penalty)):
        total_fitnes = 1/(distances + penalty)
    fitness.append(total_fitnes)
    Fitness.fitness_result = np.array(fitness)

def Evaluate(offspring, fitness, popSize):
    fitness = np.transpose(fitness)
    offspring = offspring[:, :, np.newaxis]
    merg = np.hstack((offspring, fitness))
    fit_sort = sorted(merg, key=lambda col: -col[len(merg[0]) - 1])
    selected = np.array(fit_sort[0:popSize])
    selected = np.squeeze(selected)
    best = selected.astype(int)
    Evaluate.selected = best[:, 0:len(selected[0]) - 1]
    Evaluate.best = np.array(selected[0, 0:len(selected[0])])

# waktu/jadwal tiap lokasi
def Schedule(offspring, arrived, start, best):
    arrived = np.transpose(arrived)
    start = np.transpose(start)
    temp = np.dstack((offspring, arrived, start))

    for i in range(len(temp)):
        time = temp[i][:, 0]
        comparison = time == best
        if comparison.all():
            route = temp[i]
    Schedule.result = route

def Run(destination, latLong, time, visit):

   destination = list(map(int, destination.split(",")))
   visit_time = list(map(int, visit.split(",")))
   time_user = datetime.strptime(time, '%H:%M').time()
   popSize = 5
   crossover_rate = 0.5
   mutation_rate = 0.5
   epochs = 1

   # data
   Data()

   # inisialisasi chromosome
   chromosome = Chromosome(destination, popSize)
   best_fitness = []
   for epoch in range(epochs):

      # crossover
      crossover = PMX_Crossover(Chromosome.result, crossover_rate)

      # mutation
      mutation = Exchange_Mutation(Chromosome.result, mutation_rate)

      # offspring
      offspring = Offspring(Chromosome.result, PMX_Crossover.result, Exchange_Mutation.result)

      # user dist
      s_user = Distance(latLong, Offspring.offspring)

      # dist offspring
      distOffspring = Dist_Offspring(Distance.s_user, Offspring.offspring, Data.dist)

      # time offspring
      timeOffspring = Time_Offspring(Offspring.offspring, Data.time)

      # visit duration
      visit_duration = Visit_Time(destination, visit_time, Offspring.offspring)

      # penalty 0
      penalty_0 = Penalty_0(Distance.s_user, time_user, Offspring.offspring, Data.dist, Time_Offspring.time_result) 

      # penalty i
      penalty_i = Penalty_i(Penalty_0.arrived_time, Offspring.offspring, Data.dist, Time_Offspring.time_result, Visit_Time.result)
         
      # total penalty
      penalty = Penalty(Penalty_0.penalty_0, Penalty_i.penalty_i)

      # fitness
      fitness = Fitness(Dist_Offspring.total_dist, Penalty.penalty)

      # evaluate
      evaluate = Evaluate(Offspring.offspring, Fitness.fitness_result, popSize)

      # best fitnes
      best_fitness.append(Evaluate.best)

      Chromosome.result = Evaluate.selected

   # sorting best fitness
   best = np.array(sorted(best_fitness, key=lambda col: -col[len(best_fitness[0]) - 1]))

   # best fit of generations
   best = best[0, 0:len(best[0]) - 1]

   # schedule
   schedule = Schedule(Offspring.offspring, Penalty_i.arrived, Penalty_i.start, best)

   time_result = []
   best_route = Schedule.result[:, 0]
   arrvd = Schedule.result[:, 1]
   finish = Schedule.result[:, 2]
   for i in range(min(len(Schedule.result), len(arrvd), len(finish))):
       data = {
           'Destination ' + str(i + 1) : Data.loc[(best_route[i]) - 1, 0],
           'Start at' : str(arrvd[i]),  
           'Finish at' : str(finish[i])
       }
       time_result.append(data)
   print(time_result)

   Json_decode(time_result)

   return(Json_decode.json)


def index(request):
    destination = request.POST['destination']  
    visit = request.POST['visit duration']
    origin = request.POST['origin']
    start_time = request.POST['start time']
    return HttpResponse(Run(destination, origin, start_time, visit))

