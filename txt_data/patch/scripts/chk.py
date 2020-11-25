from stanza.server import CoreNLPClient
import os
from utils.preprocessing.extract_interrogative import InterroPhraseExt

def clean_tree_str(s):
    s = s.replace("\n", " ")
    for _ in range(20):
        new = s.replace("  ", " ")
        if new == s:
            break
        s = new
    return s

os.environ["CORENLP_HOME"] = "/home/chrislin/stanford-corenlp-4.1.0"

# text = "How does she describe Oprah Winfrey?"
text = "what's another word for cell membrane?"
text = "what is your name?"
text = "The presence of the Altan Khan in the west reduced whose influence?"
text = "The skit on JImmy Kimmel Live was a depiction of Kanye West and what reporter?"
text = "According to Genshin, what has the power to destroy karma?"
text = "What was the criticism the Buddha gave dealing with animals?"
text = "the vinaya was recited by whom?"
text = "What was the executive producer of Pop Idol in 2001?"
text = "Who told the media that the Chinese supplied with shirts and transportation?"
text = "In the 2000s, were investors seeking higher yields than those offered by this investment?"
text = "Arab countries entered the financial crisis in exceptionally strong positions giving them a cushion against what?"
text = "From where is the Irish prime minister usually selected?"
text = "Where is the Irish prime minister usually selected from?"
text = "What are extremely bright lights used to deter crime?"
text = "Has daylighting been proven to have negative effects on people?"
text = "Who elected the king?"
text = "Can the term \"black people\" have different meanings?"
text = "Whose duties are defined as \"support on Middle Eastern and North African countries?\""
text = "Until 1918 whose legal status was dependent on the russian empiresovereignty of szlachta?"
text = "Did males or females most frequently report drinking to feel the effects of the alcohol in Hong Kong?"
text = "By the 1720s, how many cafes were in Paris?"
text = "The custom of designating areas of Jewish settlement with biblical names meant that what france was called??"
text = "Unless what English firms were allow to trade with India?"
text = "How common is the consumtion of poultry in the world?"
text = "At what percentage people suffer from atypical symptoms?"
text = "Was there any advocates for the working children during the 19th century?"
text = "Under whose reign did the lute become popular?"
text = "How to call an acquisition made by IBM in 2009, name it."
text = "What is the name of an acquisition made by IBM in 2009?"
text = "Do Venezuelans still come to America for the same reasons as they did before?"
text = "Would those in favor of DST argue that it causes people to use more electricity or saves energy?"
text = "In Australia, were rural or urban areas generally more strongly opposed to DST?"
text = "Were rural or urban areas generally more strongly opposed to DST in Australia?"
text = "If diarrhea is stopped, what would could happen?"
text = "Other than Scotland's Chief Law Officer, from where are most ministers drawn from amongst?"
text = "What does a pair of strips of adhesive cells on the stomach wall of some species of beroe do?"
text = "To reduce the chances of combustion, what is required for safely handeling pure O\n2?"
text = "What does silicates of magnesium and iron make up of in the Earth?"
text = "A church was built on top of the Byzantine church of the Lazarium in what century?"
text = "How much of the normal pressure Oxygen was used at to to ensure safety of future space missions?"
text = "To whom Telenet was sold?"
text = "What releases oxygen in cellular respiration?"
text = "Do Eusocial insects provide food for their offspring full-time or part-time?"
text = "What is called finding defects once a change in code had already happened?"
text = "Until when Europe did not feel the need to posses territory in Africa?"
text = "Thoreau argues that usually majority rules but their views collectively are sometimes what?"
text = "Whose duties are defined as \"support on Middle Eastern and North African countries?\""
text = "Who has the duties defined as \"support on Middle Eastern and North African countries?\""
text = "What is the name of the last county given that Orange, San Diego, Riverside and San Bernardino make up four of the five counties?"
text = "Do Venezuelans still come to America for the same reasons as they did before?"
text = "Do they still come to America for the same reasons as they did before?"
text = "On type of system were the civil administrations of the empire based on what?"
# text = "Does the scientific community agree that earthquake prediction is possible?"
# text = "What are two hockey teams located in NYC?"
# text = "What was a version of baseball played in city streets nicknamed in the 1930s?"

ext = InterroPhraseExt()

with CoreNLPClient(
        annotators=['tokenize', 'parse'], threads=4, memory='4G',
        endpoint='http://localhost:5501', be_quiet=True, timeout=30000) as client:
    document = client.annotate(text, output_format="json")
    s = document['sentences'][0]['parse']
    s = clean_tree_str(s)
    print(s)
    w = ext(s)
    print(w)
