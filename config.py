TRAIN_PATH = './data/train/train.csv'
TEST_PATH = './data/test/test.csv'
PROCESSED_TRAIN_PATH = './data/train/processed_train.csv'
PROCESSED_VAL_PATH = './data/train/processed_val.csv'
PROCESSED_TEST_PATH = './data/test/processed_test.csv'
SENTIMENT_INCLUDED_TRAIN_PATH = './data/train/train_with_sentiment.csv'

SENTIMENT_FOLDER_PATH = './data/train_sentiment/'

RANDOM_STATE = 526

numerical_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']
senti_features = ['PositiveSentimentScore', 'NegativeSentimentScore']
categorical_features = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 
                        'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'AdoptionSpeed']
textual_features = ['Name', 'Description']
id_features = ['RescuerID', 'PetID']
target = ['AdoptionSpeed']

puncts = ['。', ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', 
          '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '_', '{', '}', 
          '^', '`', '<', '°', '™', '♥', '½', '…', '“', '”', '–', '●', '²', '¬', '↑',
          '—', '：', '’', '☆', 'é', '¯', '♦', '‘', '）', '↓', '、', '（', '，', '♪', 
          '³', '❤', 'ï', '√']

mispell_dict = {
    "I'd": 'I would',
    "I'll": 'I will',
    "I'm": 'I am',
    "I've": 'I have',
    "ain't": 'is not',
    "aren't": 'are not',
    "can't": 'cannot',
    'cancelled': 'canceled',
    'centre': 'center',
    'colour': 'color',
    "could've": 'could have',
    "couldn't": 'could not',
    "didn't": 'did not',
    "doesn't": 'does not',
    "don't": 'do not',
    'enxiety': 'anxiety',
    'favourite': 'favorite',
    "hadn't": 'had not',
    "hasn't": 'has not',
    "haven't": 'have not',
    "he'd": 'he would',
    "he'll": 'he will',
    "he's": 'he is',
    "here's": 'here is',
    "how's": 'how is',
    "i'd": 'i would',
    "i'll": 'i will',
    "i'm": 'i am',
    "i've": 'i have',
    "isn't": 'is not',
    "it'll": 'it will',
    "it's": 'it is',
    'labour': 'labor',
    "let's": 'let us',
    "might've": 'might have',
    "must've": 'must have',
    'organisation': 'organization',
    "she'd": 'she would',
    "she'll": 'she will',
    "she's": 'she is',
    "shouldn't": 'should not',
    "that's": 'that is',
    'theatre': 'theater',
    "there's": 'there is',
    "they'd": 'they would',
    "they'll": 'they will',
    "they're": 'they are',
    "they've": 'they have',
    'travelling': 'traveling',
    "wasn't": 'was not',
    'watsapp': 'whatsapp',
    "we'd": 'we would',
    "we'll": 'we will',
    "we're": 'we are',
    "we've": 'we have',
    "weren't": 'were not',
    "what's": 'what is',
    "where's": 'where is',
    "who'll": 'who will',
    "who's": 'who is',
    "who've": 'who have',
    "won't": 'will not',
    "would've": 'would have',
    "wouldn't": 'would not',
    "you'd": 'you would',
    "you'll": 'you will',
    "you're": 'you are',
    "you've": 'you have',
    '，': ',',
    '／': '/',
    '？': '?'
}