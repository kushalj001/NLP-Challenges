import re
import string

n = int(input())
names = [str(input()) for i in range(n)]
name_gender = {}
for name in names:
    name_gender[name] = 0

with open('corpus.txt','r',encoding='utf8') as corpus:
     text = corpus.read()

clean = ""
for ch in text:
    if ch not in string.punctuation:
        clean += ch


words = clean.split()
x = 0

# Words related to male and female gender respectively. Adding more words to this list will result in better accuracy
male_words = ['him','himself','he','man','brother','father','widower','half-brother','actor','host','Sir','king','husband','son','Earl','Duke','Emperor','emperor','Prince','monk','businessman','boy','uncle']
female_words = ['her','herself','she','woman','lady','women','actress','sister','mother','half-sister','actress','Queen','queen','wife','daughter','widow','Duchess','Princess','nun','girl','aunt']


for i,name in enumerate(names):
    indices = []
    all_appearences = []
    indices = [i for i,x in enumerate(words) if x==name]

    # Fetch all appearences of the name from the corpus
    for index in indices:
        start = index - 2
        end = index + 10
        window = words[start:end]
        all_appearences.append(window)

    # Check for salutation/title,since this gives a more a clear indication of the gender
    for appearance in all_appearences:
        if appearance[1] in ['Mr.','Sir','King']:
            name_gender[name] += 100
        elif appearance[1] in ['Mrs.','Miss','Queen']:
            name_gender[name] -= 100

    # Other determinants of the gender        
        for word in appearance:
            if word in male_words:
                name_gender[name] += 5
            elif word in female_words:
                name_gender[name] -=5


for name in names:
    if name_gender[name]<=0:
        print("Female")
    else:
        print("Male")
