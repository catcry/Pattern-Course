# numeralizing
service_list = np.unique(service) 
k=len (service_list)

while k>0:
    service[service==serv_list[k]]=k
    k=k-1
    
flag_list = np.unique(flag)
k=len(flag_list)

while k>0
