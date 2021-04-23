from classifiers import * 
from summarize import * 
from earthquake_data import * 
from data2template import * 
from data2template2 import * 
from website2text import * 
from data2template4 import *

def updateLog(): 
    # Read in tweepy api and news api keys
    keys_parser = configparser.ConfigParser()
    keys_parser.read('keys.ini')
    twitter_keys = [keys_parser['tweepyapi_key']['consumer_key'],
                    keys_parser['tweepyapi_key']['consumer_secret'],
                    keys_parser['tweepyapi_key']['access_token'],
                    keys_parser['tweepyapi_key']['access_token_secret']
                    ]
    news_key = keys_parser['newsapi_key']['api_key']

    # Query USGS and get social media and news data
    today = date.today()
    print("Today is: ", today)
    # Yesterday date
    yesterday = today - timedelta(days=1)
    print("Query date: ", yesterday)

    get_earthquake_data(yesterday, twitter_keys, news_key)
    
def runData2Template(inputDictionary): 
    earthquakeLogs = pd.read_csv("data/earthquakes_log.csv") 
    earthquakeLogs = earthquakeLogs.to_numpy()

    for x in range(0, 1): 
        #add 7 days check only then generate report
      
        ruptureTime = earthquakeLogs[x][3]
        city = earthquakeLogs[x][1]
        
        indexCheck = ruptureTime.find(" ") 
        ruptureTime = ruptureTime[:indexCheck] + "_" + ruptureTime[indexCheck + 1:]
        if(not (ruptureTime, city) in inputDictionary): 
            inputDictionary[(ruptureTime, city)] = [] 
            
        workingPath = os.path.join(os.getcwd(), "reports", ruptureTime + "_" + city) 
        
        if(not os.path.isdir(workingPath)): 
            os.mkdir(workingPath)
            
        initialSummary = (generateSummary("data/earthquakes_log.csv", workingPath + "/record.txt", (x)))
        tectonicSummary = (getTectonicIntensityInformation("data/earthquakes_log.csv", workingPath+ "/intensity.jpg", workingPath + "/record.txt", x))
        
        newsArticles = pd.read_csv("data/news/" + ruptureTime[: ruptureTime.find("_")] + "_" + city + "_earthquake.csv") 
        
        sentencesList = [] 
        
        urlLinks = (newsArticles["url"]).to_numpy() 
        
        for j in range(9, min(len(urlLinks), 2) + 15): 
            get_content(sentencesList, urlLinks[j], workingPath + "/article.txt") 
        
        buildingColumn = np.repeat("building", len(sentencesList)) 
        
        sentencesList = np.array(sentencesList).reshape(-1, 1)
        sentencesList = np.hstack((sentencesList, buildingColumn.reshape(-1, 1)))
        
        urlFrame = pd.DataFrame(sentencesList, columns = ["sentence", "label"]) 
        urlFrame = urlFrame.dropna() 
        
        urlFrame.to_csv(workingPath + "/sentences.csv", index = False) 
        
        pager = earthquakeLogs[x][11] + "/pager"
        
        resultText = generate_estimation(pager, workingPath + "/1.png", workingPath + "/2.png", workingPath + "/1.txt") 
        
        run_classifiers("dataset.csv", workingPath + "/sentences.csv", workingPath + "/output.csv", False) 
        dataOutput = pd.read_csv(workingPath + "/output.csv") 
        
        
        sectionsDictionary = {"Buildings": get_summary(dataOutput, "building", 0.2), "Resilience": get_summary(dataOutput, "resilience", 0.2), "Infrastructure": get_summary(dataOutput, "infrastructure", 0.2)}
        
        document = Document() 
        
        document.add_heading("Earthquake Report for " + city + " on " + ruptureTime[:ruptureTime.find("_")], 0) 
        document.add_heading("Hazard Description", 1) 
        
        document.add_paragraph(initialSummary) 
     
        document.add_paragraph(tectonicSummary) 
        document.add_picture(workingPath + "/intensity.jpg", width = Inches(4.9), height = Inches(4.9)) 
        document.add_heading("Buildings", 1) 
        document.add_paragraph(sectionsDictionary["Buildings"]) 
        
        document.add_heading("Infrastructure", 1) 
        document.add_paragraph(sectionsDictionary["Infrastructure"]) 
        
        document.add_heading("Resilience", 1) 
        document.add_paragraph(sectionsDictionary["Resilience"]) 
        document.add_paragraph(resultText) 
        
        document.add_picture(workingPath + "/1.png", width = Inches(4), height = Inches(2))
        document.add_picture(workingPath + "/2.png", width = Inches(4), height = Inches(2)) 
        
        document.add_page_break() 
        
        document.save(workingPath + "/briefing.docx")  
        
        
        
        
    #process("data/earthquakes_log.csv", "record.txt", 1);
if __name__ == '__main__': 
    updateLog()
    
    earthquakeDictionary = {} 
    
    runData2Template(earthquakeDictionary) 
  #  print(earthquakeDictionary) 

  #  run_classifiers("dataset.csv", "Albania.csv", "output.csv") 
  #  run_classifiers("dataset.csv", "sentences.csv","output.csv")

   # data = pd.read_csv("output.csv") 

   # sectionsDictionary = {"Buildings": get_summary(data, "building", 0.2), "Resilience": get_summary(data, "resilience", 0.2), "Infrastructure": get_summary(data, "infrastructure", 0.2)}

   # generateBriefing(sectionsDictionary)