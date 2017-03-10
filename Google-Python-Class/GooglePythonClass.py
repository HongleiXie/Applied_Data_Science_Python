# -*- coding: utf-8 -*-
"""
"""

import sys

def Hello(name):
    if(name == 'Alice'):
        print('Alert: Alice Mode')
        name = name + '???'
    
    elif(name == 'Nick'):
        print('Nice')
        
    else:
        print('Other names')
        
    name = name + '!!!'
    print('Hello', name)
 
# Define a main() function that prints a little greeting

Hello(sys.argv[1])

# 0 and empty lists mean False
# and/or/not
 
a = 'Hello'
a = "isn't"

# would not change the original 'a' string
a.lower()

# find the first instance, returning a index number
a.find('e')

# use % to substitute
'Hi %s I have %d donuts' %('Alice', 42)

# slice, from index 1 to 3 but not including 3
a[1:3]
# omit means until the end or start from the begining
a[1:]
a[:3]#first 3 chars
a[-4:-2]
a[-3: ]#last 3 chars


for num in range(2, 10):
    if num % 2 == 0:
        print ("Found an even number", num)
        #compare: continue; break; and nothing here
    
    print ("Found a number", num)

######## list ###########################################
a = [1,2,3]
a = [1,2,3,'aa']
a = [1,2,3] + [4,5]

## NOT make a copy here, actually both a and b are pointing
## to the same variable in system
b = a
# note that list is mutable, unlike string
a[0] = 12
# check a and b

# how to make a copy
c = a[:]
c[:-1]# remove the last element

 
 for num in a: print(num)
 
 # check if something in the list: VALUE in LIST
 2 in a
 
# append is to change the list, do not return anything
 a.append(4)
# wrong way: a = a.append(4)

# pop removes the element by specifing the index number, change the list
a.pop(0)
a.pop(1)

# delete the variable
del a
a = [1,2,3]
del a[1]

# sort the list
a = [4,2,1,6]
# sorted makes a copy, not changing the original list a
sorted(a)
sorted(a, reverse = True)

a = ['ccc', 'aaaa', 'd', 'bb']
sorted(a)
# now if I want to sort a by length
sorted(a, key = len)

a[1] = 'aaaz'
# now if I want to sort by the last character in the string
# def my own function
def Last(s): return s[-1]
sorted(a, key = Last)

# cat string
b = ':'.join(a)
'\n'.join(a)

# split string
b.split(':')


result = []
for s in a: result.append(s)


############ tuple #####################
# tuple is fixed-sized and immutable
a = (1,2,3)
len(a)
a[0]

# wrong cuz tuple is immutable
#a[0] = 13

# list of tuples
a = [(1, 'b'), (2, 'a'), (1, 'a')]
sorted(a)


############ dictionary #####################
d = {}
d['a'] = 'alpha' # each key points to some value
d['b'] = 'omega'
d['g'] = 'gamma'
d

d['a']

#wrong cuz no key called as x
d['x']
# but
d.get('x') # return None if no such key
d.get('a') # otherwise returns values

# test if some key is in the dic
'a' in d
'x' in d

# return a list of keys with random order
d.keys()
# likewise; with the same random order
d.values()

for k in sorted(d.keys()): print ('key:', k, '->', d[k]) 

# return a list of tuples with size 2
d.items()

for t in d.items(): print(t)


#########################################################
def Cat(filename):
    f = open(filename, 'rU')
    
    lines = f.readlines()
    text = f.read()
    print text,
    #for line in f:
    #    print line,
    f.close()


######### Regular expression ############################    
 import re
 match = re.search('iig', 'called piiig')
 
 def Find(pat, text):
     match = re.search(pat, text)
     if match: print (match.group())
     else: print ('not found')
 
 Find('ig', 'pigg')
 
 # dot . means any char
 # \w means word character A-Z a-z, 0-9, space doesn't account as word character
 # \d means digits
 # \s means whitespace or tabs
 # \S means non-whitespace
 # + means 1 or more
 # * means 0 or more
 
 
Find('...g', 'called piiig')

Find('..g', 'called piig much better: xyzg') # only found the first one

Find('..gs', 'called piig such better: xyzgs')

Find(r'.c\.', 'hc.lled piig much better: xyzg')
# if we want to match . , use \.
# r means raw string

# : followed by three chars \w
Find(r':\w\w\w', 'blah :cat blah blah blah')

# two numbers in a row
Find(r'\d\d', 'blah 123xx blah blah blah')

# use \s to refer to space, tabs...
Find(r'\d\s\d\s\d', '12 3 4') 

# use + means 1 or more
Find(r'\d\s+\d\s+\d', '12      3        4') 

# if we don't know the length of the characters (cat) is 3
Find(r':\w+', 'blah :cat blah blah blah')
# + is greedy, it goes as far as it can and stops
Find(r'\d+', 'blah :cat12& blah 13 blah blah')
Find(r'\w+', 'blah :cat12& blah 13 blah blah')
Find(r':\w+', 'blah :cat12& blah 13 blah blah')
#numbers are considered as word char; but special symbols like & are not

Find(r':.+', 'blah :cat12& blah 13 blah blah') # . means anything

# \S
Find(r':\S', 'blah blah :cat12& blah 13 blah blah') 

# terminates at the first whitespace
Find(r':\S+', 'blah blah :cat12& blah 13 blah blah')  

###### imagine that we are going to extract someone's email address
# . is not considered as \w
Find(r'\w+@\w+', 'blah nick.p@gmail.com yatoo @12')

# [] means set of variables, . in [] means literally . not any char
Find(r'[\w.]+@\w+', 'blah nick.p@gmail.com yatoo @12')

# fix the .com issue
Find(r'[\w.]+@[\w.]+', 'blah nick.p@gmail.com yatoo @12')

Find(r'\w[\w.]+@[\w.]+', 'blah .nick.p@gmail.com yatoo @12')

###### imagine we want to extract username and host name
# use () to denote the things we care about
m = re.search(r'([\w.]+)@([\w.]+)', 'blah nick.p@gmail.com yatoo @12')
m.group()
m.group(1) # 1 means the first (), in this case, nick.p
m.group(2) # 2 means the second (), in this case, gmail.com

re.findall(r'[\w.]+@[\w.]+', 'blah nick.p@gmail.com yatoo @12')   

# return a tuple with two since there are two ()    
re.findall(r'([\w.]+)@([\w.]+)', 'blah nick.p@gmail.com yatoo @12')      


########## OS #######################################################
import os

def List(dir):
    filenames = os.listdir(dir)
    for f in filenames:
        path = os.path.join(dir, filename)
        print(path)
        print(os.path.abspath(path))

        
List(sys.argv[1])

os.path.exists('/tmp/foo')

import shutil
shutil.copy(source, dest)

import commands
# return a tuple: (status, output)
commands.getstatusoutput()


def List(dir):
    cmd = 'ls -l' + dir
    (status, output) = commands.getstatusoutput(cmd)
    if status:
        sys.stderrr.write('there was an error:' + output)
        sys.exit(1)
    print (output)
    
# useful for debugging
def List(dir):
    cmd = 'rf' + dir
    print ('about to do this:', cmd)
    return
    (status, output) = commands.getstatusoutput(cmd)
    print (output)



import urllib

uf = urllib.urlopen('http://google.com')
uf.read()

#download
urllib.urlretrive('http://google.com/intl/en_ALL/images/logo.gif')



##### List Comprehension ##########
a = ['daad', 'd', 'ccccc']
[len(s) for s in a]

a = [1,2,3,4]

[num*num for num in a if num > 2]

os.listdir('.')

import re
[f for f in os.listdir('.') if re.search(r'__\w+__', f)]

































 
 
 
 
 












































































