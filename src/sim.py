from statistics import mean
import simpy
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
import math
from datetime import datetime

# postgres package
import psycopg2

# font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
# font_manager.findfont("Gill Sans")

"""Lognormal class courtesy of Thomas Monks, Associate Professor of Health Data Science at The University of Exeter."""
class Lognormal:
    """
    Encapsulates a lognormal distirbution
    """
    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean = mean of the lognormal distribution
        stdev = standard dev of the lognormal distribution
        """
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma
        
    def normal_moments_from_lognormal(self, m, v):
        '''
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m = mean of lognormal distribution
        v = variance of lognormal distribution
                
        Returns:
        -------
        (float, float)
        '''
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma
        
    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rand.lognormal(self.mu, self.sigma)


"""start of SimPy model"""

# class to hold global parameters - used to alter model dynamics
class p:
    # patients ids
    all_patient_ids = list(range(1000001))
    # interarrival mean for exponential distribution sampling
    inter = 7
    # mean and stdev for lognormal function which converts to Mu and Sigma used to sample from lognoral distribution
    mean_doc_consult = 35
    stdev_doc_consult = 10
    mean_nurse_triage = 15
    stdev_nurse_triage = 5

    number_docs = 4
    doc_ids = list(range(1, number_docs+1))
    number_nurses = 3
    ae_cubicles = 7

    # mean time to wait for an inpatient bed if decide to admit
    mean_ip_wait = 90
    
    # simulation run metrics
    warm_up = 120
    sim_duration = 1440
    number_of_runs = 10

    # some placeholders used to track wait times for resources
    wait_triage = []
    wait_cubicle = []
    wait_doc = []

    # MIU metrics
    mean_doc_consult_miu = 20
    stdev_doc_consult_miu = 7

    number_docs_miu = 2
    number_nurses_miu = 3
    miu_cubicles = 5

    wait_doc_miu = []


class Tracker: # currently tracking number of triage waiters
    def __init__(self) -> None:
        # some place holders to track number of waiters by points in time
        self.env_time_all = []
        self.waiters = {
            'triage': [],
            'cubicle': [],
            'ae_doc': [],
            'miu_doc': []
        }
        self.waiters_all = {
            'triage': [],
            'cubicle': [],
            'ae_doc': [],
            'miu_doc': []
        }
        # empty df to hold patient level details, including time in system, priority etc
        self.results_df = pd.DataFrame()
        self.results_df["PatientID"] = []
        #self.results_df["ProviderID"] = []
        self.results_df["ArrivialMode"] = []
        self.results_df["Priority"] = []
        self.results_df["TriageOutcome"] = []
        self.results_df["TimeInSystem"] = []
        self.results_df["AttendanceTime"] = []
        self.results_df["DepartureTime"] = []
        self.results_df["Admitted"] = []
        self.results_df["DoctorIDSeenBy"] = []
        self.results_df["ProviderID"] = []
        #self.results_df.set_index("PatientID", inplace=True)

    def plot_data(self, env_time, type):
        if env_time > p.warm_up:
            self.waiters_all[type].append(len(self.waiters[type]))
            self.env_time_all.append(env_time)

    def mean_priority_wait(self):
        self.priority_means = {}
        for i in range(1, 6):
            try:
                self.priority_means["Priority{0}".format(i)] = mean(self.results_df[self.results_df['Priority'] == i]['TimeInSystem'])
            except:
                self.priority_means["Priority{0}".format(i)] = np.NaN

    def priority_count(self):
        self.priority_counts = {}
        for i in range(1, 6):
            try:
                self.priority_counts["Priority{0}".format(i)] = len(self.results_df[self.results_df['Priority'] == i]['TimeInSystem'])
            except:
                self.priority_counts["Priority{0}".format(i)] = 0

# class representing patients coming in
class AEPatient:
    def __init__(self, p_id) -> None:
        self.p_id = p_id
        self.time_in_system = 0
        self.attendance_date_time = None
        self.departure_date_time = None
        self.admitted = "False"
        self.doctor_id_seen = None

    def set_arrival_mode(self):
        self.arrival_mode = random.choices(['Arrival by ambulance', 
                                             'Arrival by own transport', 
                                             'Arrivial by public transport',
                                             'Arrival by air ambulance'],
                                             [0.30, 0.50, 0.17, 0.03])[0]

    def set_priority(self):
        # set priority according to weighted random choices - most are moderate in priority
        self.priority = random.choices([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1])[0]

    def set_triage_outcome(self):
        # decision tree - if priority 5, go to Minor Injury Unit (MIU) or home. Higher priority go to AE
        if self.priority <5:
            self.triage_outcome = 'AE'
        elif self.priority == 5: # of those who are priority 5, 20% will go home with advice, 80% go to 'MIU'
            self.triage_outcome = random.choices(['home', 'MIU'], [0.2, 0.8])[0]

# class representing AE model
class AEModel:
    # set up simpy env
    def __init__(self, provider_id) -> None:
        self.env = simpy.Environment()
        #self.patient_counter = 0
        # set docs and cubicles as priority resources - urgent patients get seen first
        self.doc = simpy.PriorityResource(self.env, capacity=p.number_docs)
        self.nurse = simpy.Resource(self.env, capacity=p.number_nurses)
        self.cubicle = simpy.PriorityResource(self.env, capacity=p.ae_cubicles)
        # MIU resources - all FIFO
        self.doc_miu = simpy.Resource(self.env, capacity=p.number_docs_miu)
        self.nurse_miu = simpy.Resource(self.env, capacity=p.number_nurses_miu)
        self.cubicle_miu = simpy.Resource(self.env, capacity=p.miu_cubicles)

        # provider id
        self.provider_id = provider_id

        # maybe try adding self.track here 
        self.track = Tracker()
        # provider_id
        #self.provider_id = run_number

        ### next will define database connections
        # Define your PostgreSQL connection parameters
        self.postgres_host = "postgres"
        self.postgres_port = "5432"
        self.postgres_user = "postgres"
        self.postgres_password = "postgres"
        self.postgres_database = "source_system"

    # a method that generates AE arrivals
    def generate_ae_arrivals(self):
        while True:
            # add pat
            #self.patient_counter += 1
            # Pick a random ID from the list without replacement
            random_pat_id = random.sample(p.all_patient_ids, 1)[0]

            # create class of AE patient and give ID
            ae_p = AEPatient(random_pat_id)

            # simpy runs the attend ED methods
            self.env.process(self.attend_ae(ae_p))

            # Randomly sample the time to the next patient arriving to ae.  
            # The mean is stored in the p class.
            sampled_interarrival = random.expovariate(1.0 / p.inter)

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)

    def attend_ae(self, patient):
        # set arrivial mode
        patient.set_arrival_mode()
        # lets capture the attendance date/time
        if not self.track.results_df.empty and 'AttendanceTime' in self.track.results_df:
            last_time = self.track.results_df['AttendanceTime'].iloc[-1]
            patient.attendance_date_time = last_time + pd.Timedelta(minutes=random.expovariate(1.0 / p.inter))             
        else:
            patient.attendance_date_time = datetime.now()
            
        # this is where we define the pathway through AE
        triage_queue_start = self.env.now
        # track numbers waiting at each point
        self.track.plot_data(self.env.now, 'triage')
        self.track.plot_data(self.env.now, 'cubicle')
        self.track.plot_data(self.env.now, 'ae_doc')
        # request a triage nurse
        with self.nurse.request() as req:
            # append env time
            # self.track.plot_data(env.now)
            # append to current waiters
            self.track.waiters['triage'].append(patient)

            # freeze until request can be met
            yield req
            # remove from waiter list (FIFO)
            self.track.waiters['triage'].pop()
            # self.track.plot_data(env.now)
            triage_queue_end = self.env.now
            
            if self.env.now > p.warm_up:
                p.wait_triage.append(triage_queue_end - triage_queue_start)

            # sample triage time from lognormal
            lognorm = Lognormal(mean=p.mean_nurse_triage, stdev=p.stdev_nurse_triage)
            sampled_triage_duration = lognorm.sample()
            #sampled_triage_duration = random.expovariate(1.0 / p.mean_nurse_triage)
            # assign the patient a priority
            patient.set_priority()
            
            yield self.env.timeout(sampled_triage_duration)

        # sample chance of being sent home or told to wait for doc
        #proceed_to_doc = random.uniform(0,1)
        # alternative way to select choice
        patient.set_triage_outcome()

        if patient.triage_outcome == 'AE':
            cubicle_queue_start = self.env.now

            with self.cubicle.request(priority = patient.priority) as req_cub: # request cubicle before doctor
                # track cubicle
                self.track.waiters['cubicle'].append(patient)
                yield req_cub
                self.track.waiters['cubicle'].pop()
                cubicle_queue_end = self.env.now
                # record AE cubicle wait time
                if self.env.now > p.warm_up:
                        p.wait_cubicle.append(cubicle_queue_end - cubicle_queue_start)
                doc_queue_start = self.env.now

            # request doc if greater than chance sent home
                with self.doc.request(priority = patient.priority) as req_doc:
                    self.track.waiters['ae_doc'].append(patient)
                    yield req_doc
                    self.track.waiters['ae_doc'].pop()
                    doc_queue_end = self.env.now
                    if self.env.now > p.warm_up:
                        p.wait_doc.append(doc_queue_end - doc_queue_start)
                    # random select doctor
                    doc = random.choices(p.doc_ids)[0]
                    # remove doc from available list while occupied
                    p.doc_ids.remove(doc)
                    patient.doctor_id_seen = doc
                    # sample consult time from lognormal
                    lognorm = Lognormal(mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_consult_duration = lognorm.sample()

                    yield self.env.timeout(sampled_consult_duration)
                # return doctor to available list
                p.doc_ids.append(doc)
                # below prob of request for IP bed. AE doc released but not cubicle
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.3:     
                    patient.admitted = "True"               
                    sampled_ip_duration = random.expovariate(1.0 / p.mean_ip_wait) # sample the wait time for an IP bed - exponential dist
                    yield self.env.timeout(sampled_ip_duration)
                # else leave the system
                

        elif patient.triage_outcome == 'MIU':
            miu_attend_start = self.env.now

            with self.cubicle_miu.request() as req_cub:
                yield req_cub
                
                with self.doc_miu.request() as req:
                    yield req

                    miu_doc_queue_end = self.env.now
                    if self.env.now > p.warm_up:
                        p.wait_doc_miu.append(miu_doc_queue_end - miu_attend_start)
                    # sample consult time
                    lognorm = Lognormal(mean=p.mean_doc_consult_miu, stdev=p.stdev_doc_consult_miu)
                    sampled_consult_duration = lognorm.sample()

                    yield self.env.timeout(sampled_consult_duration)
        # else leave the system
        # record time in system
        patient.time_in_system = self.env.now - triage_queue_start
        patient.departure_date_time = patient.attendance_date_time + pd.Timedelta(minutes=patient.time_in_system)   
        if self.env.now > p.warm_up:
            df_to_add = pd.DataFrame({"PatientID":[patient.p_id],
                                      "ProviderID": [self.provider_id],
                                      "ArrivialMode": [patient.arrival_mode],
                                      "Priority":[patient.priority],
                                      "TriageOutcome":[patient.triage_outcome],
                                      "TimeInSystem":[patient.time_in_system],
                                      "AttendanceTime": [patient.attendance_date_time],
                                      "DepartureTime": [patient.departure_date_time],
                                      "Admitted": [patient.admitted],
                                      "DoctorIDSeenBy": [patient.doctor_id_seen]})
            # df_to_add.set_index("P_ID", inplace=True)
            frames = [self.track.results_df, df_to_add]
            self.track.results_df = pd.concat(frames, ignore_index=True)
            self.track.results_df.reset_index(drop=True, inplace=True)

    ### next we will create postgres table and insert df
    def insert_results_to_database(self):
        connection = psycopg2.connect(
            host=self.postgres_host,
            port=self.postgres_port,
            user=self.postgres_user,
            password=self.postgres_password,
            database=self.postgres_database
        )

        try:
            cursor = connection.cursor()

            # Create a table for storing the results if it doesn't exist
            create_table_query = """
                CREATE TABLE IF NOT EXISTS store.ae_attends (
                    patient_id float,
                    provider_id float,
                    arrival_mode VARCHAR(255),
                    priority INT,
                    triage_outcome VARCHAR(255),
                    time_in_system FLOAT,
                    attendance_time TIMESTAMP,
                    departure_time TIMESTAMP,
                    admitted VARCHAR(25),
                    doctor_id_seen VARCHAR(255)
                );
            """
            cursor.execute(create_table_query)
            connection.commit()

            # Insert results into the table
            for index, row in self.track.results_df.iterrows():
                insert_query = """
                    INSERT INTO store.ae_attends (patient_id, provider_id, arrival_mode, priority, triage_outcome, time_in_system, attendance_time, departure_time, admitted, doctor_id_seen)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                print(row['PatientID'], row["ProviderID"])
                data = (
                    row["PatientID"],
                    row["ProviderID"],
                    row["ArrivialMode"],
                    row["Priority"],
                    row["TriageOutcome"],
                    row["TimeInSystem"],
                    row["AttendanceTime"],
                    row["DepartureTime"],
                    row["Admitted"],
                    row["DoctorIDSeenBy"]
                )
                cursor.execute(insert_query, data)
                connection.commit()

        except (Exception, psycopg2.Error) as error:
            print("Error inserting data into PostgreSQL:", error)

        finally:
            cursor.close()
            connection.close()

            

    # method to run sim
    def run(self):
        self.env.process(self.generate_ae_arrivals())
        
        self.env.run(until=p.warm_up + p.sim_duration)

        # send data to db
        # Insert results into the PostgreSQL database
        self.insert_results_to_database()
        # calculate mean waits per priority
        self.track.mean_priority_wait()
        self.track.priority_count()
        return mean(p.wait_triage), mean(p.wait_cubicle), mean(p.wait_doc), mean(p.wait_doc_miu)

     

# For the number of runs specified in the g class, create an instance of the
# AEModel class, and call its run method


all_runs_triage_mean = []
all_runs_cubicle_mean = []
all_runs_doc_mean = []
all_runs_miu_doc_mean = []
all_time_in_system = []
all_number_of_patients = []

all_run_time_wait_key = {
    'triage': {},
    'cubicle': {},
    'ae_doc': {},
    'miu_doc': {}
}

all_run_priority_time_in_system = {
    'Priority1': [],
    'Priority2': [],
    'Priority3': [],
    'Priority4': [],
    'Priority5': []
}

all_run_priority_counts = {
    'Priority1': [],
    'Priority2': [],
    'Priority3': [],
    'Priority4': [],
    'Priority5': []
}
###
if __name__ == "__main__":
    for run in range(p.number_of_runs):
        #print (f"Run {run} of {p.number_of_runs}")

        #track = Tracker()
        my_ae_model = AEModel(provider_id=run) # run here represents provider id 
        triage_mean, cubicle_mean , doc_mean, miu_mean = my_ae_model.run()
        # reset doctors after each run
        p.doc_ids = list(range(1, p.number_docs+1))
        # some metrics if needed
        all_runs_triage_mean.append(triage_mean)
        all_runs_cubicle_mean.append(cubicle_mean)
        all_runs_doc_mean.append(doc_mean)
        all_runs_miu_doc_mean.append(miu_mean)
        # number of patients served per run
        all_number_of_patients.append(len(my_ae_model.track.results_df))
        # tracking number of waiters in key queues through sim
        for k in all_run_time_wait_key:
            for t, w in zip(my_ae_model.track.env_time_all, my_ae_model.track.waiters_all[k]):
                #print(t, w)
                all_run_time_wait_key[k].setdefault(round(t), [])
                all_run_time_wait_key[k][round(t)].append(w)
            all_run_time_wait_key[k] = dict(sorted(all_run_time_wait_key[k].items())) # sort items
        all_time_in_system.append(mean(my_ae_model.track.results_df['TimeInSystem']))
        # get priority wait times
        for i in range(1, 6):           
            all_run_priority_time_in_system["Priority{0}".format(i)].append(my_ae_model.track.priority_means["Priority{0}".format(i)])
        # number of patient per priority
        for i in range(1, 6):           
            all_run_priority_counts["Priority{0}".format(i)].append(my_ae_model.track.priority_counts["Priority{0}".format(i)])
        #print ()

    print(f"The average number of patients served by the system was {round(mean(all_number_of_patients))}")
    print(f"The overall average wait across all runs for a triage nurse was {mean(all_runs_triage_mean):.1f} minutes")
    print(f"The overall average wait across all runs for a cubicle was {mean(all_runs_cubicle_mean):.1f} minutes")
    print(f"The overall average wait across all runs for a doctor was {mean(all_runs_doc_mean):.1f} minutes")
    print(f"The overall average wait across all runs for a MIU doctor was {mean(all_runs_miu_doc_mean):.1f} minutes")
    print(f"The mean patient time in the system across all runs was {mean(all_time_in_system):.1f} minutes")
###

#print(f"The mean patient time in the system across all runs was {mean(list(itertools.chain(*all_time_in_system))):.1f} minutes")


# number of patients per priority
# patients_per_priority = []
# for k in all_run_priority_counts:
#     patients_per_priority.append(round(mean(all_run_priority_counts[k])))
# patients_per_priority


# wait_means = {
#     'triage': [],
#     'cubicle': [],
#     'ae_doc': [],
#     'miu_doc': []
# }
# # lower quartiles
# wait_lq = {
#     'triage': [],
#     'cubicle': [],
#     'ae_doc': [],
#     'miu_doc': []
# }
# # upper quartiles
# wait_uq = {
#     'triage': [],
#     'cubicle': [],
#     'ae_doc': [],
#     'miu_doc': []
# }

# for k in all_run_time_wait_key:
#     for t in all_run_time_wait_key[k]:
#         wait_means[k].append(round(mean(all_run_time_wait_key[k][t]), 2))
#         wait_lq[k].append(np.percentile(all_run_time_wait_key[k][t], 25))
#         wait_uq[k].append(np.percentile(all_run_time_wait_key[k][t], 75))

# all_run_time_wait_key
# wait_means
# wait_lq

# print("max attend date is ", self.track.results_df['AttendanceTime'].max())
# print("max depart date is ", self.track.results_df['DepartureTime'].max())
# self.track.results_df

