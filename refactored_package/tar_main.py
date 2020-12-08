from classifiers import * 
from summarize import * 

if __name__ == '__main__': 
    run_classifiers(sys.argv[1], sys.argv[2])

    data = pd.read_csv("output.csv") 

    sectionsDictionary = {"Buildings": get_summary(data, "building", 0.2), "Resilience": get_summary(data, "resilience", 0.2), "Infrastructure": get_summary(data, "infrastructure", 0.2)}

    generateBriefing(sectionsDictionary)