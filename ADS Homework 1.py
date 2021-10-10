#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Hello World


# In[ ]:


print("Hello, World!")


# In[ ]:


#If Else


# In[ ]:


import math
import os
import random
import re
import sys


n=int(input())
if n%2==1:
    print("Weird")
elif n>1 and n<6:
    print("Not Weird")
elif n>5 and n<21:
    print("Weird")
else:
    print("Not Weird")


# In[ ]:


#Arithmetic Operators


# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)


# In[ ]:


#Divisions


# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


# In[ ]:


#Loops


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)


# In[ ]:





# In[ ]:





# In[ ]:


#Write a Function


# In[ ]:


def is_leap(year):
    leap = False
    if year%4==0 and (year%100!=0 or year%400==0):
            return(True)
    else:return(False)
    
    # Write your logic here
    
    return leap


# In[ ]:


#Print Function


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    print(*range(1,n+1),sep='')


# In[ ]:


---------------------------Basic Data Types


# In[ ]:


#List Comprehension


# In[ ]:


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k!=n])


# In[ ]:


#Find the Runner Up score


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    arr.sort(reverse=True)
    cont=0
    a=max(arr)
    for i in arr:
        if a==i:
            cont=cont+1
    print(arr[cont])


# In[ ]:


#Nested List


# In[ ]:


if __name__ == '__main__':
    l=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        l.append([name,score])
    l=sorted(l, key=lambda x:x[1])
    a=l[0][1]
    count=0
    for i in range(len(l)):
        if l[i][1]==a:
            count=count+1
    b=l[count][1]
    s=[]
    for j in range(len(l)):
        if l[j][1]==b:
            s.append(l[j][0])
    s.sort()
    for i in range(len(s)):
       print(s[i])


# In[ ]:


#Finding the Percentage


# In[ ]:


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()


# In[ ]:


#Tuples


# In[ ]:


if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    t=tuple(integer_list)
    print(hash(t))


# In[ ]:


--------------------String Formatting


# In[ ]:


#Swap Case


# In[ ]:


def swap_case(s):
    x=s.swapcase()
    return(x)


# In[ ]:


#Split and Join


# In[ ]:


def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return(line)
    # write your code here


# In[ ]:


#What's Your Name


# In[ ]:


def print_full_name(first, last):
    a= "Hello " + first + " " + last + "! You just delved into python."
    return(print(a))


# In[ ]:


#Mutations


# In[ ]:


def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    n="".join(l)
    return(n)


# In[ ]:


#Find a String


# In[ ]:


def count_substring(string, sub_string):
    l=len(sub_string)
    count=0
    for i in range(0,len(string)-l+1):
        if string[i:i+l]==sub_string:
            count+=1
    return(count)


# In[ ]:


#String Validators


# In[ ]:


if __name__ == '__main__':
    s = input()
    count=0
    for i in range(len(s)):
        if s[i].isalnum():
            count=count+1
    if count>0:
        print("True")
    else:
        print("False")
    count=0
    for i in range(len(s)):
        if s[i].isalpha():
            count=count+1
    if count>0:
        print("True")
    else:
        print("False")
    count=0
    for i in range(len(s)):
        if s[i].isdigit():
            count=count+1
    if count>0:
        print("True")
    else:
        print("False")
    count=0
    for i in range(len(s)):
        if s[i].islower():
            count=count+1
    if count>0:
        print("True")
    else:
        print("False")
    count=0
    for i in range(len(s)):
        if s[i].isupper():
            count=count+1
    if count>0:
        print("True")
    else:
        print("False")


# In[ ]:


#Text Alignment


# In[ ]:


#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# In[ ]:


#Text Wrap


# In[ ]:


def wrap(string, max_width):
    s=textwrap.fill(string,max_width)
    return(s)


# In[ ]:


#Designer Door Mat


# In[ ]:


# Enter your code here. Read input from STDIN. Print output to STDOUT
l=input()
l=l.split(" ")
n=int(l[0])
m=int(l[1])
c=".|."

for i in range(0,(n)//2):
    print((c*(1+2*i)).center(m,"-"))
print("WELCOME".center(m,"-"))

for i in range(n//2-1,-1,-1):
    print((c*(1+2*i)).center(m,"-"))


# In[ ]:


#Capitalize


# In[ ]:


# Complete the solve function below.
def solve(s):
    s=s.replace(s[0],s[0].upper())
    l=list(s)
    for i in range(1,len(l)):
        if l[i]==" ":
            if l[i+1].isalnum():
                l[i+1]=l[i+1].upper()
    l="".join(l)
    return(l)


# In[ ]:


#The Minion Game


# In[ ]:


def minion_game(string):
    # your code goes here
    l=len(string)
    v=0
    c=0
    
    for i in range(l):
        if string[i] in "AEIOU":
            v=v+(l-i)
        else:
            c=c+(l-i)
    
    if v>c:
        print("Kevin", v)
    elif v==c:
        print("Draw")
    else:
        print("Stuart", c)


# In[ ]:


#Merge Tools


# In[ ]:


def merge_the_tools(string, k):
    l=len(string)
    for i in range(0,l,k):
        vuoto=""
        for a in string[i:i+k]:
            if a not in vuoto:
                vuoto=vuoto+a
        print(vuoto)


# In[ ]:


--------------------------------Set


# In[ ]:


#Introduction to Sets


# In[ ]:


def average(array):
    u=set(array)
    s=sum(u)
    l=len(u)
    m=round(s/l,3)
    return(m)


# In[ ]:


#No Idea!


# In[ ]:


N=input()
n=input().split(" ")
n=list(map(int,n))
A=input().split(" ")
A=set(map(int,A))
B=input().split(" ")
B=set(map(int,B))
c=0
for i in range(len(n)):
    if n[i] in A:
        c=c+1
    elif n[i] in B:
        c=c-1
    else:
        c=c
print(c)


# In[ ]:


#Symmetric Difference


# In[ ]:


m=int(input())
M=input()
M=M.split(" ")
M=list(map(int,M))
M=set(M)
n=int(input())
N=input()
N=N.split(" ")
N=list(map(int,N))
N=set(N)

a=N.difference(M)
b=M.difference(N)
q=a.union(b)
q=sorted(q)
for i in q:
    print(i)


# In[ ]:


#Set.add


# In[ ]:


l=set([str(input()) for i in range(int(input()))])
print(len(l))


# In[ ]:


#Set.discard


# In[ ]:


n = int(input())
s = set(map(int, input().split()))
for i in range(int(input())):
    eval("s.{0}({1})".format(*input().split()+[" "]))  
print(sum(s))


# In[ ]:


#Set union


# In[ ]:


n=int(input())
e=set(map(int,input().split(" ")))
b=int(input())
f=set(map(int,input().split(" ")))
u=e.union(f)
print(len(u))


# In[ ]:


#Set intersection


# In[ ]:


n=int(input())
e=set(map(int,input().split(" ")))
b=int(input())
f=set(map(int,input().split(" ")))
u=e.intersection(f)
print(len(u))


# In[ ]:


#Set difference


# In[ ]:


n=int(input())
e=set(map(int,input().split(" ")))
b=int(input())
f=set(map(int,input().split(" ")))
u=e.difference(f)
print(len(u))


# In[ ]:


#Set symmetric difference


# In[ ]:


n=int(input())
e=set(map(int,input().split(" ")))
b=int(input())
f=set(map(int,input().split(" ")))
u=e.symmetric_difference(f)
print(len(u))


# In[ ]:


#Set mutations


# In[ ]:


n=int(input())
A=set(map(int,input().split(" ")))
c=int(input())
l=[]
m=[]
for i in range(0,c):
    l.append(str(input()).split(" ")[0])
    m.append(set(map(int,input().split(" "))))
for i in range(c):
    eval("A.{0}({1})".format(l[i],m[i]))    
    
print(sum(A))


# In[ ]:


#The Captain's Room


# In[ ]:


n=int(input())
v=list(map(int,input().split(" ")))
r=set(v)
f=set()
s=set()
l=len(f)
for i in v:
    if i in f:
        s.add(i)
    else:
        f.add(i)
d=f.difference(s)
print(list(d)[0])


# In[ ]:


#Check Subset


# In[ ]:


n=int(input())
for i in range(n):
    na=int(input())
    A=set(map(int,input().split(" ")))
    nb=int(input())
    B=set(map(int,input().split(" ")))
    d=B.difference(A)
    ld=len(d)
    if nb-na==ld:
        print("True")
    else:
        print("False")


# In[ ]:


#Check Strict Superset


# In[ ]:


A=set(map(int,input().split(" ")))
n=int(input())
la=len(A)
count=0
for i in range(n):
    s=set(map(int,input().split(" ")))
    ls=len(s)
    d=A.difference(s)
    ld=len(d)
    if la-ls==ld:
        count=count+1
if count==n:
    print("True")
else:
    print("False")


# In[ ]:


-----------------------------------Collections


# In[ ]:


#collection counter


# In[ ]:


from collections import Counter
n=int(input())
l=list(map(int,input().split(" ")))
c=int(input())
mag=Counter(l)
soldi=0
for i in range(c):
    tag,pre= map(int, input().split(" "))
    if mag[tag]!=0:
        mag[tag]=mag[tag]-1
        soldi=soldi+pre
print(soldi)


# In[ ]:


#Default Dict


# In[ ]:


from collections import defaultdict
d = defaultdict(list)
list1=[]

n, m = map(int,input().split())

for i in range(0,n):
    d[input()].append(i+1) 

for i in range(0,m):
    list1=list1+[input()]  

for i in list1: 
    if i in d:
        print(" ".join(map(str,d[i])))
        #print (" ".join( map(str,d[i]) ))
    else:
        print( -1)


# In[ ]:


#Collection Named Tuple


# In[1]:


from collections import namedtuple

n = int(input())
col = input().split()
somma = 0
for i in range(n):
    stu = namedtuple('student',col)
    col1, col2, col3,col4 = input().split()
    tus = stu(col1,col2,col3,col4)
    somma += int(tus.MARKS)
print(somma/n)


# In[ ]:


#Collection.ordereddict()


# In[ ]:


from collections import OrderedDict
n=int(input())
ordered_dictionary = OrderedDict()
for i in range(n):
    obj, vuoto, prezzo=input().rpartition(' ')
    ordered_dictionary[obj]=ordered_dictionary.get(obj,0)+int(prezzo)
for obj, prezzo in ordered_dictionary.items():
    print(obj,prezzo)


# In[ ]:


#Word Order


# In[ ]:


from collections import Counter
n=int(input())
l=[]
for i in range(n):
    l.append(str(input()))
print(len(Counter(l).keys()))
print(*Counter(l).values())


# In[ ]:


#Collections.deque


# In[ ]:


from collections import deque
n=int(input())
d=deque()
for i in range(n):
    exec("d.{0}({1})".format(*input().split()+['']))
print(*d)


# In[ ]:


#Piling Up!


# In[ ]:


n=int(input())
for t in range(n):
    input()
    l = list(map(int, input().split()))
    ll = len(l)
    i = 0
    while i < ll - 1 and l[i] >= l[i+1]:
        i += 1
    while i < ll - 1 and l[i] <= l[i+1]:
        i += 1
    if i==ll-1:
        print("Yes")
    else:
        print("No")


# In[ ]:


#Company Logo


# In[ ]:


import math
import os
import random
import re
import sys
from collections import Counter


if __name__ == '__main__':
    string = sorted(Counter(input()).items(), key= lambda x: (-x[1],x[0]))[:3]
    for i in string:
        print(str(i[0])+" "+str(i[1]))


# In[ ]:


#Calendar Module


# In[ ]:


import calendar
MM, DD, YYYY=map(int,input().split())
print((calendar.day_name[calendar.weekday(YYYY,MM,DD)]).upper())


# In[ ]:


#Exceptions


# In[ ]:


n=int(input())
for i in range(n):
    try:
        a,b=map(int,input().split(" "))
        print(a//b)
        
    except Exception as e:
        print("Error Code:",e)


# In[ ]:


#Incorrect Regex Discussion


# In[ ]:


import re
n=int(input())
for i in range(n):
    r="True"
    try:
        cos=re.compile(input())
    except re.error:
        r="False"
    print(r)


# In[ ]:


#Zipped!


# In[ ]:


n,x=map(int,input().split(" "))
l=[]

for i in range(x):
    l.append(list(map(float,input().split(" "))))

for i in zip(*l):
    m=sum(i)/x
    print(m)


# In[ ]:


#Input


# In[ ]:


x,k = map ( int , raw_input().split())
print input() == k


# In[ ]:


#Python Evaluation


# In[ ]:


eval(str(input()))


# In[ ]:


#Athlete Sort


# In[ ]:


import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
    arr=sorted(arr, key=lambda x: x[k])
    
    for i in arr:
        print(*i)


# In[ ]:


#ginortS


# In[ ]:


s=list(input())
l=[]
u=[]
o=[]
e=[]
for i in s:
    if i.isalpha():
        if i.islower():
            l.append(i)
        else:
            u.append(i)
    elif i.isdigit():
        if int(i)%2==1:
            o.append(i)
        else:
            e.append(i)
l=sorted(l)
u=sorted(u)
o=sorted(o)
e=sorted(e)
print("".join(l+u+o+e))


# In[ ]:


#Map and Lamda Function


# In[ ]:


cube = lambda x: x**3# complete the lambda function 

def fibonacci(n):
    if n==0:
        return()
    elif n==1:
        return([0])
    else:
        l=[0,1]
        for i in range(2,n):
            l.append(l[i-1]+l[i-2])
        return(l)


# In[ ]:





# In[ ]:


#Detect Floating Point Number


# In[ ]:


import re
for i in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))


# In[ ]:


#Re.split()


# In[ ]:


regex_pattern = r"[,.]"	# Do not delete 'r'.
import re
print("\n".join(re.split(regex_pattern, input())))


# In[ ]:


#Group()


# In[ ]:


import re
a = re.search(r'([a-zA-Z0-9])\1+', input())
if a:
    print(a.group(1))
else:
    print(-1)


# In[ ]:


#Re.finditer()


# In[ ]:


import re
s = '[qwrtypsdfghjklzxcvbnm]'
a = re.findall('(?<=' + s +')([aeiou]{2,})' + s, input(), re.I)
print('\n'.join(a or ['-1']))


# In[ ]:


#Re.start(), re:end


# In[ ]:


S = input()
k = input()
import re
p = re.compile(k)
r = p.search(S)
if not r: print ("(-1, -1)")
while r:
    print ("({0}, {1})".format(r.start(), r.end() - 1)) 
    r = p.search(S,r.start() + 1)


# In[ ]:


#Regex Substitution


# In[ ]:


import re
n = int(input())

for i in range(n):
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', lambda x: 'and' if x.group() == '&&' else 'or', input()))


# In[ ]:


#Validating


# In[ ]:


import re
n=int(input())
for i in range(n):
    if re.match(r'[789]\d{9}$',input()):   
        print('YES')  
    else:  
        print ('NO') 


# In[ ]:


#Validating and Parsing Email Addresses


# In[ ]:


import re
n = int(input())
for _ in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)


# In[ ]:


#Hex Color


# In[ ]:


import re
pattern=r'(#[0-9a-fA-F]{3,6}){1,2}[^\n ]'
for _ in range(int(input())):
    for x in re.findall(pattern,input()):
        print(x)


# In[ ]:


#HTML Parser 1


# In[ ]:


import re

r_tag = re.compile(r'<\s*(\w+)\s*([^>]*)>|<\s*\/\s*(\w+)\s*>')
r_empty = re.compile(r'/\s*$')
r_attr = re.compile(r'\s*([^ =]+)(?:\s*=\s*(?:(?:"([^"]+)")|(?:\'([^\']+)\')))?')

N = int(input())
html = ''
while N > 0:
    N -= 1
    html += input()

html = re.sub('<!--.*?-->', '', html, re.S)
for group in r_tag.findall(html):
    start_tag = group[0]
    attrs = group[1]
    end_tag = group[2]
    status = ('Start' if not r_empty.search(attrs) else 'Empty') if start_tag else 'End  '
    
    print(status + ' : ' + (start_tag or end_tag))
    
    if start_tag:
        for attr in r_attr.findall(r_empty.sub('', attrs)):
            print('-> ' + attr[0]  + ' > ' + (attr[1] or attr[2] or 'None'))


# In[ ]:


#HTML Parser 2


# In[ ]:


from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_data(self, data):
        if data != '\n':
            print('>>> Data   ')
            print(data)
    def handle_comment(self, data):
        if len(data.split('\n')) == 1:
            print('>>> Single-line Comment  ')
            print(data)
        elif len(data.split('\n')) > 1:
            print('>>> Multi-line Comment  ',)
            for i in data.split('\n'):
                print(i) 
  
if __name__ == '__main__':
    lis = []
    for _ in range(int(input())):
        lis.append(input())
    html = '\n'.join(lis)
    parser = MyHTMLParser()
    parser.feed(html)
  


# In[ ]:


#Detect HTML Tags


# In[ ]:


from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]
        
html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# In[ ]:


#Validating UID


# In[ ]:


import re

for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')


# In[ ]:


#Validating Credit Card Number


# In[ ]:


import re
TESTER = re.compile(
    r"^"
    r"(?!.*(\d)(-?\1){3})"
    r"[456]"
    r"\d{3}"
    r"(?:-?\d{4}){3}"
    r"$")
for _ in range(int(input().strip())):
    print("Valid" if TESTER.search(input().strip()) else "Invalid")


# In[ ]:


#XML1


# In[ ]:


def get_attr_number(node):
    l=0
    for i in tree.iter():
        l+=len(i.items())
    return(l)


# In[ ]:


#Find the Maximum Depth


# In[ ]:


maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for i in elem:
        depth(i, level + 1)


# In[ ]:


#Standardize Mobile Numbers


# In[ ]:


def wrapper(f):
    def fun(l):
        f("+91 "+c[-10:-5]+" "+c[-5:] for c in l)
    return(fun)


# In[ ]:


#Decorators 2


# In[ ]:


def person_lister(f):
    def inner(people):
        for i in sorted(people, key=lambda x: int(x[2])):
            yield f(i)   
    return inner


# In[ ]:


#Arrays


# In[ ]:


def arrays(arr):
    l=numpy.array(arr,float)
    return(l[::-1])


# In[ ]:


#Shape and Reshape


# In[ ]:


import numpy

m=numpy.array(list(map(int,input().split(" "))))

print(numpy.reshape(m,(3,3)))


# In[ ]:


#Transpose and Flatten


# In[ ]:


import numpy

n,m=map(int,input().split(' '))
a=numpy.array([list(map(int,input().split(' ')))for i in range(n)])
print(numpy.transpose(a))    
print(a.flatten())


# In[ ]:


#Concatenate


# In[ ]:


import numpy

n,m,p=map(int,input().split(" "))

na=numpy.array([list(map(int,input().split(' '))) for i in range(n)])

ma=numpy.array([list(map(int,input().split(' '))) for i in range(m)])

print(numpy.concatenate((na,ma),axis=0))


# In[ ]:


#Zero and Ones


# In[ ]:


import numpy

a=tuple(map(int,input().split(" ")))

print(numpy.zeros(a, dtype = numpy.int))
print(numpy.ones(a, dtype = numpy.int))


# In[ ]:


#Eye and Identity


# In[ ]:


import numpy
numpy.set_printoptions(legacy='1.13')
n,m=map(int,input().split(' '))
print(numpy.eye(n, m))


# In[ ]:


#Array Mathematics


# In[ ]:


import numpy

n,m=map(int,input().split(' '))

a,b = (numpy.array([input().split() for i in range(n)], dtype=int) for i in range(2))

print(numpy.add(a,b))

print(numpy.subtract(a, b))

print(numpy.multiply(a, b))

print(numpy.floor_divide(a, b))

print(numpy.mod(a, b))

print(numpy.power(a, b))


# In[ ]:


#Floor, Ceil and Rint


# In[ ]:


import numpy

numpy.set_printoptions(legacy='1.13')

my_array=numpy.array(input().split(' '),float)

print(numpy.floor(my_array))

print(numpy.ceil(my_array))

print(numpy.rint(my_array))


# In[ ]:


#Sum and Prod


# In[ ]:


import numpy

n,m=map(int,input().split(' '))

my_array=numpy.array([input().split(' ') for i in range(n)],int)

s=numpy.sum(my_array, axis = 0)

print(numpy.prod(s,axis=0))


# In[ ]:


#Min and Max


# In[ ]:


import numpy
n,m=map(int,input().split(' '))
arr=numpy.array([input().split(' ') for i in range(n)],int)
print(numpy.max(numpy.min(arr,axis=1)))


# In[ ]:


#Mean, var, std


# In[ ]:


import numpy

n,m=map(int,input().split(' '))

arr=numpy.array([input().split(' ') for i in range(n)], int)

print(numpy.mean(arr,axis=1))
print(numpy.var(arr,axis=0))
print(round(numpy.std(arr),11))


# In[ ]:


#Dot and Cross


# In[ ]:


import numpy

n=int(input())

A=numpy.array([input().split(' ') for i in range(n)],int)
B=numpy.array([input().split(' ') for i in range(n)],int)

print(numpy.dot(A,B))


# In[ ]:


#Inner and Outer


# In[ ]:


import numpy

A=list(map(int,input().split(' ')))
B=list(map(int,input().split(' ')))

print(numpy.inner(A,B))
print(numpy.outer(A,B))


# In[ ]:


#Polynomials


# In[ ]:


import numpy

coef=list(map(float,input().split(' ')))

x=int(input())

print(numpy.polyval(coef,x))


# In[ ]:


#Linear Algebra


# In[ ]:


import numpy

n=int(input())

m=numpy.array([input().split(" ") for i in range(n)], float)
print(round(numpy.linalg.det(m),2))


# In[ ]:


Problema 2


# In[ ]:


#Birthday Cake Candle


# In[ ]:


import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    m=max(candles)
    c=0
    for i in candles:
        if i==m:
            c+=1
    return(c)
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# In[ ]:


#Kangaroo


# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    if x1>x2 and v1>=v2:
        return("NO")
    elif x1<x2 and v1<=v2:
        return("NO")
    elif x1-x2>0 and v1-v2<=0:
        if (x1-x2)%(v2-v1)==0:
            return("YES")
        else:
            return("NO")
    elif x1-x2<0 and v1-v2>=0:
            if (x2-x1)%(v1-v2)==0:
                return("YES")
            else: return("NO")
        

    
    
    
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# In[ ]:


#Viral Advertise


# In[ ]:


import math
import os
import random
import re
import sys

def viralAdvertising(n):
    p=[2]
    for i in range(n-1):
        p.append((3*p[i])//2)
    return(sum(p))
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# In[ ]:


#Recursive Digit Sum


# In[ ]:


import math
import os
import random
import re
import sys


def superDigit(n, k):
        x = int(n) * k % 9
        if x:
            return(x)
        else:
            return(9)
        
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# In[ ]:


#Insertion Sort 1


# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    i = n-1
    c = arr[i]
    while(i>0 and c<arr[i-1]):
        arr[i] = arr[i-1]
        print(*arr)
        i-=1
    arr[i] = c
    print(*arr)
    # Write your code here

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# In[ ]:


#Insertion Sort 2


# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for i in range(1,n):
        c=arr[i]
        j=i-1
        while j>=0 and arr[j]>c:
            arr[j+1]=arr[j]
            j=j-1
        arr[j+1]=c
        print(*arr)
    # Write your code here

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

