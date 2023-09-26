from logAnalyzier import *



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


if __name__ == '__main__':
    target_sentences = [
        "What's your name?",
        "How old are you?",
        # ... add all 40 sentences here
    ]
    log_files = ["test.txt"]
    analyzer = LogAnalyzer(target_sentences)
    simple_textSearch()