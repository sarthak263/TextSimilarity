from nltk import corpus

from logAnalyzier import *


'''
def simple_textSearch():
    string_pattern = r"Hi my name is \w+"

    start_time = time.time()
    string_match_scores = analyzer.process_files_with_string_match(string_pattern, log_files)
    end_time = time.time()

    print(f"Total time taken for string match: {end_time - start_time} seconds")
    print("String Match Scores:", string_match_scores)

def transformer_textSearch():
    # List all log files (assuming they are in the folder 'log_files')

    start_time = time.time()
    scores = analyzer.process_multiple_files(log_files)
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time} seconds")
    print("Scores:", scores)
'''

if __name__ == '__main__':
    target_sentences = [
        "Is it ok to text you on your mobile number?",
        "In order to serve you better, I'm going to ask you a few questions to verify your identity and get us on the way to your solution",
    ]
    log_analyzer = LogAnalyzer(target_sentences=target_sentences)
    log_file_path = "test.txt"

    score = log_analyzer.process_log_file(log_file_path)
    print(score)

    #print(f"The number of similar questions found in {log_file_path} is {score}")
