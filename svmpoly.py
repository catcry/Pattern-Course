# LOADING DATA AND PUTTING THEM IN 42 MATRICES


#=======================Function for number of itteration in array=======
#Results\  A function  to get No. of "1"s and "-1"s

def num_of(n,x):
    p=0
    num_of_n=0
    while p<len(x):
        if x[p]==n:
                num_of_n+=1
        p+=1
    return num_of_n
#========================================================================




import numpy as np
import csv
num_of_recz=0

with open ('KDDTrain+.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
	    num_of_recz=num_of_recz+1

a=np.zeros((num_of_recz,43),dtype='<U10')


i=0

with open ('KDDTrain+.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        a[i,:]=(row)
        i=i+1        

duration = np.copy(a[:,0].astype(np.float))
protocol_type = np.copy(a[:,1])
service =a[:,2]
flag = a[:,3]
scr_bytes = a[:,4].astype(np.float)
dst_bytes = a[:,5].astype(np.float)
land = a[:,6].astype(np.float)
wrong_fragment = a[:,7].astype(np.float)
urgent = a[:,8].astype(np.float)

hot = a[:,9].astype(np.float)
num_failed_logins = a[:,10].astype(np.float)
logged_in = a[:,11].astype(np.float) 
num_compromised = a[:,12].astype(np.float)
root_shell = a[:,13].astype(np.float)
su_attempted = a[:,14].astype(np.float)
num_root = a[:,15].astype(np.float)
num_file_creations = a[:,16].astype(np.float)
num_shell = a[:,17].astype(np.float)
num_access_files = a[:,18].astype(np.float)
num_outbound_cmds = a[:,19].astype(np.float)
is_hot_login = a[:,20].astype(np.float)
is_guest_login = a[:,21].astype(np.float)

count = a[:,22].astype(np.float)
srv_count =a[:,23].astype(np.float)
serror_rate = a[:,24].astype(np.float)
srv_serror_rate = a[:,25].astype(np.float)
rerror_rate = a[:,26].astype(np.float)
srv_rerror_rate = a[:,27].astype(np.float)
same_srv_rate= a[:,28].astype(np.float)
diff_srv_rate = a[:,29].astype(np.float)
srv_diff_host_rate = a[:,30].astype(np.float)

dst_host_count = a[:,31].astype(np.float)
dst_host_srv_count = a[:,32].astype(np.float)
dst_host_same_srv_rate = a[:,33].astype(np.float)
dst_host_diff_srv_rate = a[:,34].astype(np.float)
dst_host_same_src_port_rate = a[:,35].astype(np.float)
dst_host_srv_diff_host_rate = a[:,36].astype(np.float)
dst_host_serror_rate = a[:,37].astype(np.float)
dst_host_srv_serror_rate = a[:,38].astype(np.float)
dst_host_rerror_rate = a[:,39].astype(np.float)
dst_host_srv_error_rate = a[:,40].astype(np.float)
label_train = np.copy(a[:,41])
unknown = a[:,42].astype(np.float)


#============================================================================END OF DATA LOADING=================================================


#PreProcessing

#PreProcessing\ Numeralization

protocol_type_list = np.unique(protocol_type) 
k=len(protocol_type_list)-1
while k>-1:
    protocol_type[protocol_type==protocol_type_list[k]]=k
    k=k-1
protocol_type = protocol_type.astype(np.float)
    

service_list = np.unique(service) 
k=len(service_list)-1
while k>-1:
    service[service==service_list[k]]=k
    k=k-1
service=service.astype(np.float)

flag_list = np.unique(flag)
k=len(flag_list)-1

while k>-1:
    flag[flag==flag_list[k]]=k
    k=k-1
flag=flag.astype(np.float)

#-----------------------------------------------------------
#PreProcessing\ Normalization

def nrmlz (x):
    ma=x.max()
    mi=x.min()
    k=0
    while k<len(x):
        x[k]=(x[k]-mi)/(ma-mi)
        k+=1
    return x

nrmlz(duration)
nrmlz(protocol_type)
nrmlz(service)
nrmlz(flag)
nrmlz(scr_bytes)
nrmlz(dst_bytes)
nrmlz(wrong_fragment)
nrmlz(urgent)

nrmlz(hot)
nrmlz(num_failed_logins)
nrmlz(num_compromised)
nrmlz(num_root)
nrmlz(num_file_creations)
nrmlz(num_shell)
nrmlz(num_access_files)
#nrmlz(num_outbound_cmds) /all values in Train+ Set are equal to zero
nrmlz(count) 
nrmlz(srv_count)
nrmlz(serror_rate)
nrmlz(srv_serror_rate)
nrmlz(rerror_rate)
nrmlz(srv_rerror_rate)
nrmlz(same_srv_rate)
nrmlz(diff_srv_rate)
nrmlz(srv_diff_host_rate)

nrmlz(dst_host_count)
nrmlz(dst_host_srv_count)
nrmlz(dst_host_same_srv_rate)
nrmlz(dst_host_diff_srv_rate)
nrmlz(dst_host_same_src_port_rate)
nrmlz(dst_host_srv_diff_host_rate)
nrmlz(dst_host_serror_rate)
nrmlz(dst_host_srv_serror_rate)
nrmlz(dst_host_rerror_rate)
nrmlz(dst_host_srv_error_rate)
#-------------------------------------------------------------

#Preprocessing\ Label numericalization
label_train[label_train!='normal']=1
label_train[label_train=='normal']=0
label_train=label_train.astype(int)
#-------------------------------------------------------------


#=============================================SVM IMPLEMENTATION==========================================
#SVM\ Matices

b= np.zeros((len(duration),41))
j=0
#while j<len(duration):
#    b=np.vstack((b,[duration[j],protocol_type[j],service[j],flag[j],scr_bytes[j],dst_bytes[j],land[j],wrong_fragment[j],urgent[j],hot[j],num_failed_logins[j],logged_in[j],num_compromised[j],root_shell[j],su_attempted[j],num_root[j],num_file_creations[j],num_shell[j],num_access_files[j],num_outbound_cmds [j],is_hot_login[j],is_guest_login[j],count[j],srv_count[j],serror_rate[j],srv_serror_rate[j],rerror_rate[j],srv_rerror_rate[j],same_srv_rate[j],diff_srv_rate[j],srv_diff_host_rate[j],dst_host_count[j],dst_host_srv_count[j],dst_host_same_srv_rate[j],dst_host_diff_srv_rate[j],dst_host_same_src_port_rate[j],dst_host_srv_diff_host_rate[j],dst_host_serror_rate[j],dst_host_srv_serror_rate[j],dst_host_rerror_rate[j],dst_host_srv_error_rate[j]]))
#   j+=1
#
#b=np.delete(b, (0), axis=0)

b[:,0]=np.copy(duration)
b[:,1]=np.copy(protocol_type)
b[:,2]=np.copy(service)
b[:,3]=np.copy(flag)
b[:,4]=np.copy(scr_bytes)
b[:,5]=np.copy(dst_bytes)
b[:,6]=np.copy(land)
b[:,7]=np.copy(wrong_fragment)
b[:,8]=np.copy(urgent)

b[:,9]=np.copy(hot)
b[:,10]=np.copy(num_failed_logins)
b[:,11]=np.copy(logged_in)
b[:,12]=np.copy(num_compromised)
b[:,13]=np.copy(root_shell)
b[:,14]=np.copy(su_attempted)
b[:,15]=np.copy(num_root)
b[:,16]=np.copy(num_file_creations)
b[:,17]=np.copy(num_shell)
b[:,18]=np.copy(num_access_files)
b[:,19]=np.copy(num_outbound_cmds)
b[:,20]=np.copy(is_hot_login)
b[:,21]=np.copy(is_guest_login)

b[:,22]=np.copy(count)
b[:,23]=np.copy(srv_count)
b[:,24]=np.copy(serror_rate)
b[:,25]=np.copy(srv_serror_rate)
b[:,26]=np.copy(rerror_rate)
b[:,27]=np.copy(srv_rerror_rate)
b[:,28]=np.copy(same_srv_rate)
b[:,29]=np.copy(diff_srv_rate)
b[:,30]=np.copy(srv_diff_host_rate)

b[:,31]=np.copy(dst_host_count)
b[:,32]=np.copy(dst_host_srv_count)
b[:,33]=np.copy(dst_host_same_srv_rate)
b[:,34]=np.copy(dst_host_diff_srv_rate)
b[:,35]=np.copy(dst_host_same_src_port_rate)
b[:,36]=np.copy(dst_host_srv_diff_host_rate)
b[:,37]=np.copy(dst_host_serror_rate)
b[:,38]=np.copy(dst_host_srv_serror_rate)
b[:,39]=np.copy(dst_host_rerror_rate)
b[:,40]=np.copy(dst_host_srv_error_rate)

#==================================================================SVM Training======================

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV

tune_par = [{'kernal' : ['rbf'], 'gamma' : [1e-3, 1e-4],'C' : [1, 10, 100, 1000]}, {'kernal' : ['poly'], 'degree' : [3, 4], 'C': [1,10,100,1000]}]
clf = SVC(kernel = 'rbf', C=2000, cache_size = 5000, gamma = 'scale')
#clf = NuSVC(nu = 0.9, kernel = 'rbf', cache_size = 5000, gamma = 'scale')
clf.fit(b, label_train) 



#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=4, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)



#============================================================================================
#============================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#====================================END OF TRAINING=========================================
#============================================================================================




#=====================Test Preprocessing
num_of_recz=0
with open ('KDDTest+.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
	    num_of_recz=num_of_recz+1

a=np.zeros((num_of_recz,43),dtype='<U10')


i=0

with open ('KDDTest+.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        a[i,:]=(row)
        i=i+1        

duration = np.copy(a[:,0].astype(np.float))
protocol_type = np.copy(a[:,1])
service =a[:,2]
flag = a[:,3]
scr_bytes = a[:,4].astype(np.float)
dst_bytes = a[:,5].astype(np.float)
land = a[:,6].astype(np.float)
wrong_fragment = a[:,7].astype(np.float)
urgent = a[:,8].astype(np.float)

hot = a[:,9].astype(np.float)
num_failed_logins = a[:,10].astype(np.float)
logged_in = a[:,11].astype(np.float) 
num_compromised = a[:,12].astype(np.float)
root_shell = a[:,13].astype(np.float)
su_attempted = a[:,14].astype(np.float)
num_root = a[:,15].astype(np.float)
num_file_creations = a[:,16].astype(np.float)
num_shell = a[:,17].astype(np.float)
num_access_files = a[:,18].astype(np.float)
num_outbound_cmds = a[:,19].astype(np.float)
is_hot_login = a[:,20].astype(np.float)
is_guest_login = a[:,21].astype(np.float)

count = a[:,22].astype(np.float)
srv_count =a[:,23].astype(np.float)
serror_rate = a[:,24].astype(np.float)
srv_serror_rate = a[:,25].astype(np.float)
rerror_rate = a[:,26].astype(np.float)
srv_rerror_rate = a[:,27].astype(np.float)
same_srv_rate= a[:,28].astype(np.float)
diff_srv_rate = a[:,29].astype(np.float)
srv_diff_host_rate = a[:,30].astype(np.float)

dst_host_count = a[:,31].astype(np.float)
dst_host_srv_count = a[:,32].astype(np.float)
dst_host_same_srv_rate = a[:,33].astype(np.float)
dst_host_diff_srv_rate = a[:,34].astype(np.float)
dst_host_same_src_port_rate = a[:,35].astype(np.float)
dst_host_srv_diff_host_rate = a[:,36].astype(np.float)
dst_host_serror_rate = a[:,37].astype(np.float)
dst_host_srv_serror_rate = a[:,38].astype(np.float)
dst_host_rerror_rate = a[:,39].astype(np.float)
dst_host_srv_error_rate = a[:,40].astype(np.float)
label_test = np.copy(a[:,41])
unknown = a[:,42].astype(np.float)


#============================================================================END OF DATA LOADING=================================================


#PreProcessing

#PreProcessing\ Numeralization

protocol_type_list = np.unique(protocol_type) 
k=len(protocol_type_list)-1
while k>-1:
    protocol_type[protocol_type==protocol_type_list[k]]=k
    k=k-1
protocol_type = protocol_type.astype(np.float)
    

service_list = np.unique(service) 
k=len(service_list)-1
while k>-1:
    service[service==service_list[k]]=k
    k=k-1
service=service.astype(np.float)

flag_list = np.unique(flag)
k=len(flag_list)-1

while k>-1:
    flag[flag==flag_list[k]]=k
    k=k-1
flag=flag.astype(np.float)

#-----------------------------------------------------------
#PreProcessing\ Normalization

def nrmlz (x):
    ma=x.max()
    mi=x.min()
    k=0
    while k<len(x):
        x[k]=(x[k]-mi)/(ma-mi)
        k+=1
    return x

nrmlz(duration)
nrmlz(protocol_type)
nrmlz(service)
nrmlz(flag)
nrmlz(scr_bytes)
nrmlz(dst_bytes)
nrmlz(wrong_fragment)
nrmlz(urgent)

nrmlz(hot)
nrmlz(num_failed_logins)
nrmlz(num_compromised)
nrmlz(num_root)
nrmlz(num_file_creations)
nrmlz(num_shell)
nrmlz(num_access_files)
#nrmlz(num_outbound_cmds) /all values in Train+ Set are equal to zero
nrmlz(count) 
nrmlz(srv_count)
nrmlz(serror_rate)
nrmlz(srv_serror_rate)
nrmlz(rerror_rate)
nrmlz(srv_rerror_rate)
nrmlz(same_srv_rate)
nrmlz(diff_srv_rate)
nrmlz(srv_diff_host_rate)

nrmlz(dst_host_count)
nrmlz(dst_host_srv_count)
nrmlz(dst_host_same_srv_rate)
nrmlz(dst_host_diff_srv_rate)
nrmlz(dst_host_same_src_port_rate)
nrmlz(dst_host_srv_diff_host_rate)
nrmlz(dst_host_serror_rate)
nrmlz(dst_host_srv_serror_rate)
nrmlz(dst_host_rerror_rate)
nrmlz(dst_host_srv_error_rate)
#-------------------------------------------------------------

#Preprocessing\ Label numericalization
label_test[label_test!='normal']=1
label_test[label_test=='normal']=0
label_test=label_test.astype(int)
#-------------------------------------------------------------


#=============================================SVM IMPLEMENTATION==========================================
#SVM\ Matices

c= np.zeros((len(duration),41))
j=0
#while j<len(duration):
#    b=np.vstack((b,[duration[j],protocol_type[j],service[j],flag[j],scr_bytes[j],dst_bytes[j],land[j],wrong_fragment[j],urgent[j],hot[j],num_failed_logins[j],logged_in[j],num_compromised[j],root_shell[j],su_attempted[j],num_root[j],num_file_creations[j],num_shell[j],num_access_files[j],num_outbound_cmds [j],is_hot_login[j],is_guest_login[j],count[j],srv_count[j],serror_rate[j],srv_serror_rate[j],rerror_rate[j],srv_rerror_rate[j],same_srv_rate[j],diff_srv_rate[j],srv_diff_host_rate[j],dst_host_count[j],dst_host_srv_count[j],dst_host_same_srv_rate[j],dst_host_diff_srv_rate[j],dst_host_same_src_port_rate[j],dst_host_srv_diff_host_rate[j],dst_host_serror_rate[j],dst_host_srv_serror_rate[j],dst_host_rerror_rate[j],dst_host_srv_error_rate[j]]))
#   j+=1
#
#b=np.delete(b, (0), axis=0)

c[:,0]=np.copy(duration)
c[:,1]=np.copy(protocol_type)
c[:,2]=np.copy(service)
c[:,3]=np.copy(flag)
c[:,4]=np.copy(scr_bytes)
c[:,5]=np.copy(dst_bytes)
c[:,6]=np.copy(land)
c[:,7]=np.copy(wrong_fragment)
c[:,8]=np.copy(urgent)

c[:,9]=np.copy(hot)
c[:,10]=np.copy(num_failed_logins)
c[:,11]=np.copy(logged_in)
c[:,12]=np.copy(num_compromised)
c[:,13]=np.copy(root_shell)
c[:,14]=np.copy(su_attempted)
c[:,15]=np.copy(num_root)
c[:,16]=np.copy(num_file_creations)
c[:,17]=np.copy(num_shell)
c[:,18]=np.copy(num_access_files)
c[:,19]=np.copy(num_outbound_cmds)
c[:,20]=np.copy(is_hot_login)
c[:,21]=np.copy(is_guest_login)

c[:,22]=np.copy(count)
c[:,23]=np.copy(srv_count)
c[:,24]=np.copy(serror_rate)
c[:,25]=np.copy(srv_serror_rate)
c[:,26]=np.copy(rerror_rate)
c[:,27]=np.copy(srv_rerror_rate)
c[:,28]=np.copy(same_srv_rate)
c[:,29]=np.copy(diff_srv_rate)
c[:,30]=np.copy(srv_diff_host_rate)

c[:,31]=np.copy(dst_host_count)
c[:,32]=np.copy(dst_host_srv_count)
c[:,33]=np.copy(dst_host_same_srv_rate)
c[:,34]=np.copy(dst_host_diff_srv_rate)
c[:,35]=np.copy(dst_host_same_src_port_rate)
c[:,36]=np.copy(dst_host_srv_diff_host_rate)
c[:,37]=np.copy(dst_host_serror_rate)
c[:,38]=np.copy(dst_host_srv_serror_rate)
c[:,39]=np.copy(dst_host_rerror_rate)
c[:,40]=np.copy(dst_host_srv_error_rate)

#======================================================================================


#======================================Results:========================================

#Results\  A function  to get No. of "1"s and "-1"s

def num_of(n,x):
    p=0
    num_of_n=0
    while p<len(x):
        if x[p]==n:
                num_of_n+=1
        p+=1
    return num_of_n

#Results\ Train Prediction
prediction_train=clf.predict(b)
dif_train = label_train - prediction_train

prediction_test= clf.predict(c)
dif_test = label_test - prediction_test

train_fp = num_of(-1,dif_train)
train_fn = num_of(1,dif_train)
train_accuracy = (1-((train_fp+train_fn)/len(b)))*100


print ("======================Results of KDDTrain+ Set: ===================================")
print()
print("The Specification of Classifier is:",clf)
print ("Number of un-Detected Attacks (FN) in Training Set is: FN = ",train_fn," out of",len(b))
print ("Number of False Attack Alarms (FP) in Training Set is: FP =",train_fp," out of",len(b))
print ("Accuracy on Trainin Set: Accuracy = ",train_accuracy,"%")
print()
test_fp = num_of(-1,dif_test)
test_fn = num_of(1,dif_test)
test_accuracy = (1-((test_fp+test_fn)/len(c)))*100

print ("======================Results of KDDTest+ Set: ===================================")
print()
print ("Number of un-Detected Attacks (FN) in KDDTest+ Set is: FN =",test_fn," out of",len(c))
print ("Number of False Attack Alarms (FP) in KDDTest+ Set is: FP =",test_fp," out of",len(c))
print ("Accuracy on Trainin Set: Accuracy = ",test_accuracy,"%")
print()

from loadfunc import load
y=load ("KDDTest-21.csv")


prediction_test21= clf.predict(y[0])
dif_test21 = y[1] - prediction_test21

test21_fp = num_of(-1,dif_test21)
test21_fn = num_of(1,dif_test21)
test21_accuracy = (1-((test21_fp+test21_fn)/len(y[0])))*100
print ("======================Results of KDDTest-21 Set: ===================================")
print ()
print ("Number of un-Detected Attacks (FN) in Test21 Set is: FN = ",test21_fn," out of",len(y[0]))
print ("Number of False Attack Alarms (FP) in Test21 Set is: FP =",test21_fp," out of",len(y[0]))
print ("Accuracy on Trainin Set: Accuracy = ",test21_accuracy,"%")

