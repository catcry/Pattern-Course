# Some Usefull Functions

def num_of(n,x):
    p=0
    num_of_n=0
    while p<len(x):
        if x[p]==n:
                num_of_n+=1
        p+=1
    return num_of_n

def rnd(x):
    i=0
    while i<len(x):
        if x[i] < 0.5:
            x[i]=0
        else:
            x[i]=1
        i+=1
    return x
