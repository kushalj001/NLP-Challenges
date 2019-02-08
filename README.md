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
  *  Remaining answers viz., is, are, was, were are determined by carefully analyzing the information provided by POS tags near the blank as well as the context info. 
## Matching Book Names with Descriptions ##
* With no training data involved, the approach employed to match a description with the correct book was to calculate the _TF-IDF_ score  for each description. This was done by fitting _and_ transforming the vectorizer on all the input descriptions. This resulted in keywords/ title of the book getting relatively larger values than other words.
* Then, the book names were transformed into feature vectors using the same vectorizer and hence the same vocabulary was used to fit the descriptions. Dot product of _book_features_ and _description_features_ was calculated to maximise the values for which similar words exist in both the book names and their descriptions.
## Gender Prediction ##
* For each input name, text/ tokens/ words around the name are fetched for each appearance of the name in the corpus. The corpus is cleaned prior to this.
* For each appearance of the name, the salutation or title is checked. If it indicates a particluar gender confidently, a high score or weight is assigned to that name. Positive weights are used for male and negative for female. 
* The tokens that surround the text are also checked for other male or female words to increase the accuracy. As it turns out the accuracy increases as we add new words to this list according to the corpus.

## Transfer Learning on Stack Exchange Data ##
* After researching a lot I came across the following paper/report. The code related to the deep learning approach is in the script transfer_stack.py. The model details and idea of using the MultiLabelBinarizer are taken from the paper and the following repo https://github.com/viig99/stackexchange-transfer-learning. However, due to limited computational resources this approach did not give the expected results.
The Paper: http://www2.agroparistech.fr/ufr-info/membres/cornuejols/Teaching/Master-AIC/PROJETS-M2-AIC/PROJETS-2016-2017/challenge-kaggle-transfer%20KHOUFI_MATMATI_THIERRY.pdf
#### Word2Vec Approach ####
* This approach is a bit crude but requires almost no training time and is not _essentially_ a deep learning approach. 
* The physics test data set is fed to a TF-IDF vectorizer which spits out the important words in a particular document among all the      documents in the text. For each word in a document/question, cosine similarity with the word-vector of 'physics' is calculated.
* The embeddings are trained Google News and are loaded in with gensim. The top three words with greatest similarity score are chosen as tags for that question. The code for this can be found in the notebook Word2Vec approach.
