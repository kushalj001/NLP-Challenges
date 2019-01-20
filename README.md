# NLP-Challenges

## Flipkart Search Query ##
* The training data is not very large and has only 111 unique entries. In order to get the most out of the limited data, the text is preprocessed thoroughly. Removal of punctuation, tokenization and removal of stop words is done. Lemmatization was not proving to be too useful hence avoided. 
* The preprocessed text is then fed to a *TF-IDF* (Term frequency-inverse document frequency) vectorizer which converts the words or text features into numeric features. Two classifiers were trained and tested, Random Forest and Support Vector Machine. SVM performed better after choosing correct set of hyperparameters by using Grid search. The problem can be solved more accurately by using string matching techniques.

## To Be Or Not To Be ##
Set of rules followed in order to find the right form of verb are as follows:-
* First, all the sentences/docs with ' ---- ' are extracted. For each of these docs, the _context_ is found. Context is a dictionary that consists of two keys viz. 'past' and 'singular' with booleans as values. The context is found by counting the number of verbs in past tense or present tense using _parts-of-speech_ tagging from nltk. Also to determine whether the subject is plural or singular, counts of plural and singular nouns are recorded. If number of verbs in past tense are greater than those in present, the doc['past] = True. Likewise for nouns.
* The context information recorded above is used only if the information provided by words (and their POS tags) just before and just after the blank is incomplete. By looking at POS tags and the words themselves that surround the blank a lot of possibilities can be ruled out.
* Once the information from words near the blank and context is combined, a set of rules are determined for assigning the correct answer:-
  * If the word before the blank is 'I' then the answer is 'am'.
  * If the word before is any modal (will,could shall) , the answer will _be_ "be".
  * If the word before is in any of ['has', 'had', 'have'] the answer will be "been".
  * If the word before is in any of ['was', 'is', 'are', 'were', 'am'] the answer will be "being".
  *  Remaining answers viz., is, are, was, were are determined by carefully analyzing the information provided by POS tags near the blank      as well as the context info. 

## Project Structure ##
Each folder contains a jupyter notebook which demonstrates the working code and a python script as well. 
