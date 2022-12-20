# -*- coding: utf-8 -*-
"""searchstring_and_API
contains everything to get from the csv file after localization and OCR to a textdocument with relevant information about the books
"""

# Imports
import requests
import re
import enchant
from Levenshtein import distance as lev
from Levenshtein import hamming
from Levenshtein import jaro
from Levenshtein import jaro_winkler
from Levenshtein import ratio as str_ratio
from Levenshtein import median
from Levenshtein import median_improve
import difflib
import pandas as pd

from os import listdir
from os.path import isfile, join


def googlebooksAPI(query, number, response):
    """
    sends a search string to the google books api and extracts title, authors,
    publisher, publishing date, identifiers like isbn, and number of pages from the result
    ---
    query: valid search string
    number: the entry of the results from the google books api that should be returned
    response: all results from the google books api
    ---
    result: dictionary with useful information about the book
      all values in result are strings except identifiers:
      identifiers contains an array of dictionaries:
      [{"type": "ISBN_10",
        "identifier": "055380457X"
      },{
        "type": "ISBN_13",
        "identifier": "9780553804577"}]
    """
    title = authors = publisher = publishedDate = identifiers = pageCount = ""

    if re.sub(r'[^\+]', '', query) == "" or query == "":
        title = "No results found for empty search string"
        result = {"title": title, "authors": authors, "publisher": publisher, "publishedDate": publishedDate,
                  "identifiers": identifiers, "pageCount": pageCount, "success": False, "response": response}
        return result

    # send get request to the API, only if no old response is used
    if response is None:
        response = requests.get("https://www.googleapis.com/books/v1/volumes?q=" + str(query) + str(
            '&langRestrict=en'))  # +str('&printType=books'))

    if not 'totalItems' in response.json():
        # print(response.json())
        title = "No results found for " + str(query)
        result = {"title": title, "authors": authors, "publisher": publisher, "publishedDate": publishedDate,
                  "identifiers": identifiers, "pageCount": pageCount, "success": False, "response": response}
        return result

    # extract useful information from the result
    # print(response.status_code)#200 good, 400 bad ;)

    num = response.json()['totalItems']
    # print("Number of results: "+str(num))
    # print('Number i: '+str(number))

    if number >= num:
        # print(response.json())
        title = "No results found for " + str(query)
        result = {"title": title, "authors": authors, "publisher": publisher, "publishedDate": publishedDate,
                  "identifiers": identifiers, "pageCount": pageCount, "success": False, "response": response}
        return result

    if num > 0 and len(response.json()['items']) > number:
        # print(response.json()['items'])
        # print("First result:")
        title = response.json()['items'][number]['volumeInfo']['title']
        # print(title)#returns only 10 books in items
        if "authors" in response.json()['items'][number]['volumeInfo']:  # apparently not every title has an author
            authors = response.json()['items'][number]['volumeInfo']['authors']
            # print(authors)
        # get some more information publisher, publishedDate, identifier: ISBN_10 and _13, and pageCount
        if "publisher" in response.json()['items'][number]['volumeInfo']:
            publisher = response.json()['items'][number]['volumeInfo']['publisher']
            # print(publisher)
        if "publishedDate" in response.json()['items'][number]['volumeInfo']:
            publishedDate = response.json()['items'][number]['volumeInfo']['publishedDate']
            # print(publishedDate)
        if "industryIdentifiers" in response.json()['items'][number]['volumeInfo']:
            identifiers = response.json()['items'][number]['volumeInfo']['industryIdentifiers']
            # print(identifiers)
        if "pageCount" in response.json()['items'][number]['volumeInfo']:
            pageCount = response.json()['items'][number]['volumeInfo']['pageCount']
            # print(pageCount)
        success = True
    else:
        title = "No results found for " + str(query)
        success = False
        # print(title)
    result = {"title": title, "authors": authors, "publisher": publisher, "publishedDate": publishedDate,
              "identifiers": identifiers, "pageCount": pageCount, "success": success, "response": response}
    return result


def handleAPIoutput(results, queryString, output_path, number, matched_book):
    """
    Writes the results from the API for the given search string to a textfile "Recognized_books.txt".
    Should be modified depending on step 9 of the pipeline (Display results).

    Args:
        results: dictionary obtained from the google books API
        queryString: the string used to obtain the results
        output_path: the location were the file with the results should be placed
        number: book number based on number of books to send to the API
        matched_book: book number from the matching segmented book spine image

    Returns:
        True and the list of recovered information to be processed for adding to a dataframe
    """
    f = open(output_path + "/Recognized_books.txt", "a", encoding='utf-8')
    line = "Book #" + str(number) + ":\n"
    line += "Matched with segmented book #" + str(matched_book) + ":\n"
    line += "Result for search string:" + str(queryString.replace('+', ' ')) + "\n"
    line += str(results["title"])
    if results["authors"] != "":
        line += "\nwritten by " + str(results["authors"])
    if results["publisher"] != "":
        line += "\npublished by " + str(results['publisher'])
    if results["publishedDate"] != "":
        line += "\npublished on " + str(results['publishedDate'])
    for id in results['identifiers']:
        line += "\n" + str(id['type']) + ": " + str(id['identifier'])
    if results["pageCount"] != "":
        line += "\nnumber of pages: " + str(results['pageCount'])
    line += "\n---\n"
    # for some characters there appears a unicode encode error. If that's the case, the caracter is encoded "manually"
    try:
        f.write(line)
    except UnicodeEncodeError:
        for c in line:
            try:
                f.write(c)
            except UnicodeEncodeError:
                f.write(c.encode("utf-8"))
    f.close()
    r = [number, matched_book, queryString.replace('+', ' '), results["title"], results["authors"],
                  results["publisher"], results["publishedDate"], results['identifiers'], results["pageCount"]]
    return True, r


def niceQueryString(book, flag_sug, string=''):
    """
    Creates a query/search string for the given book
    --
    book: one book entry from the dictionary
    flag_sug: a boolean flag passed to 'selectword', decided whether to use suggestions for the search string
    --
    queryString: the produced string
    """
    # get all strings that belong to one book
    # iterate over all scenes in a book and over all reocgnitions with text in the scenes
    queryString = ""

    # if a string as basis is provided, use it otherwise get it for that book
    if string != '':
        words = string.split('+')
        for w in words:
            queryString += '+' + selectword([w], flag_sug)
    else:
        # print(book['scenes'])
        for scene in book['scenes']:  # one scene = one textbox
            # print(scene)
            words_in_scene = []
            for element in scene['recognitions']:
                # print(element)
                word = element['text'] if element['text'] is not None and pd.notna(element['text']) else ""

                # remove special characters, numbers and single letters
                word = re.sub(r'[^A-Za-z]', '', word)
                word = '' if len(word) <= 3 else word  # ignores all words smaller than 4
                words_in_scene.append(word)
            # print('word_in_scene: '+str(words_in_scene))#is good
            # one scene contains multiple recognitions fo the same word in different rotations
            # append the most likely word to the search string by using selectword
            queryString += "+" + selectword(words_in_scene, flag_sug)
            # print('string after new selection: '+str(queryString))

    # print('string before cleaning: '+str(queryString))

    # remove unnecessary '*'
    queryString = re.sub(r'\+.{0,3}\+', '+', queryString)
    queryString = re.sub(r'\+\++', '+', queryString)
    queryString = re.sub(r'\+.{0,3}\+', '+', queryString)  # ugly but works

    # print('final string: '+str(queryString))
    return queryString


def selectword(words, flag_sug):
    """
    takes words assumed in order from low to high rotation and returns most likely word or string
    ---
    words: array of strings
    flag_sug: a boolean flag that determines whether suggestions are used to replace non-english words or not
    ---
    word: string
    """
    # Filter '' words from wordlist
    words = [w for w in words if w != '']
    if len(words) == 0:
        return ""
    word = ""
    # use checkword to determine whether the words are recognized as valid english words
    is_word = [checkword(w) for w in words]
    for i in range(len(is_word)):
        if is_word[i][0]:
            word = words[i]
            # print(is_word)
            break  # so the first valid word (recognized with lowest rotation and thus most likely) is chosen

    # what if no word is valid?
    if word == "":
        # take first word and hope for the best
        if not flag_sug:
            word = words[0]
            return word

        # get suggestions for first word
        _, suggestions = checkword(words[0])
        if len(suggestions) == 0:  # if no suggestions return first word
            return words[0]
        # print(suggestions)
        # word = suggestions[0]#...and hope for the best or...

        # ...use levenshtein distance to decide on best suggestion? see example below
        # multiple options:
        # distance (Levenshtein distance) -> No (3)
        '''
        lev_dis = [lev(sug, words[0]) for sug in suggestions]
        #print(lev_dis)
        # enumerate distances, sort them and select first element (with lowest distance)
        (count, min_dis) = sorted(enumerate(lev_dis), key=lambda x:x[-1])[0]
        # get the suggestion with the lowest distance
        word = suggestions[count]
        '''
        # hamming (works only for strings of the same length) -> No
        '''
        ham_dis = [hamming(sug, words[0]) for sug in suggestions]
        (count, min_dis) = sorted(enumerate(ham_dis), key=lambda x:x[-1])[0]
        word = suggestions[count]
        '''
        # jaro - works better than levenshtein -> Test (0) zero correct
        '''
        jaro_dis = [jaro(sug, words[0]) for sug in suggestions]
        (count, min_dis) = sorted(enumerate(jaro_dis), key=lambda x:x[-1])[0]
        word = suggestions[count]
        '''
        # jaro_winkler - works better than levenshtein -> Test, better than jaro (1) one correct
        '''
        jaro_wink_dis = [jaro_winkler(sug, words[0]) for sug in suggestions]
        (count, min_dis) = sorted(enumerate(jaro_wink_dis), key=lambda x:x[-1])[0]
        word = suggestions[count]
        '''
        # ratio - works better than levenshtein -> Test (2) one correct

        ratio_dis = [str_ratio(sug, words[0]) for sug in suggestions]
        (count, min_dis) = sorted(enumerate(ratio_dis), key=lambda x: x[-1])[0]
        word = suggestions[count]

        # compute median of the strings i.e. merge strings to one string -> Test
        # very cool in theory, but probably meeds more and similar words as input
        # median
        # word = median(words)#words, suggestions, or words[0] + suggestions? (4) one correct
        # median improve
        # word = median_improve(word, suggestions) #(5) one correct
        # word = median_improve(word, suggestions)#can still improve result (6) one correct
    return word


def checkword(word):
    """
    Checks whether word is a valid english word, if not suggestions are returned
    ---
    word: string
    ---
    valid: boolean
    suggestion: array of strings (or empty array)
    """
    if word == "":
        return (False, [])
    d = enchant.Dict("en")  # "en_US"
    valid = d.check(word)
    if valid:
        suggestions = []  # no suggestions needed if word is a valid english word
    else:
        suggestions = d.suggest(word)
    return (valid, suggestions)


def import_from_csv(csv_file):
    """
    takes a csv file from the OCR output and returns the relevant data in the data structure assumed by niceQueryString()
    --
    csv_file: a csv file with recognized text and co about all books
    --
    books: the data structure that the results are included in
    """
    books = []

    # Read in necessary parts of the csv file
    data = pd.read_csv(csv_file)
    predicted_labels = data[['predicted_labels', 'book', 'text_box_id', 'includes']]

    # len_books = len(books)
    book_num_total = 0

    # iteration over all book shelves
    for shelf_num in range(data['shelf'].nunique()):
        shelf = data[data['shelf'] == data['shelf'].unique()[shelf_num]]

        # iterate over all books
        for book in range(shelf['book'].nunique()):
            # get book number
            # book_num = int(re.sub(r'book-', '', book))
            book_num = book

            # find all rows belonging to one book
            sub = shelf[shelf['book'] == shelf['book'].unique()[book_num]]
            # print('----')

            # if there is no book entry yet in books, create one
            if len(books) <= book_num_total:
                books.append(
                    {'id': book_num,
                     'scenes': []}
                )

            # if there is no scene entr yet for the book, create one
            if not 'scenes' in books[book_num_total]:
                books[book_num_total]['scenes'] = []

            # iterate over each detection/textbox per book
            already_checked = []
            for textboxid in sub['text_box_id'].unique():
                # skip detections that are related to another one and are already included in the dictionary
                if textboxid in already_checked:
                    continue
                # print('text box id: '+str(textboxid))
                # create the data element with recognized text
                recognition = {
                    'id': textboxid,
                    'recognitions': [{
                        'id': 0,  # first text gets id zero
                        'text': sub.loc[sub['text_box_id'] == textboxid, 'predicted_labels'].tolist()[0]
                        # 'text': sub.loc[sub['text_box_id'] == sub['text_box_id'].unique()[textboxid], 'predicted_labels'].tolist()[0]
                    }]}

                # check if other boxes belong to the same word
                # get related textboxes and process the string to an int array
                some_array = sub.loc[sub['text_box_id'] == textboxid, 'includes'].tolist()[0]
                # some_array = sub.loc[sub['text_box_id'] == sub['text_box_id'].unique()[textboxid], 'includes'].tolist()[0]
                some_array = re.sub(r',\Z', '', some_array.strip("[").strip("]")).split(",")
                # print('make sure there is no number '' combo: '+str(some_array)) #works
                if all([i != "" for i in some_array]):  # if not "" in some _array???
                    int_array = [int(i) for i in some_array]
                else:
                    int_array = []

                # iterate over all related text boxes
                for otherbox in int_array:
                    # append new recognition to the previously created data element
                    # print(sub.loc[sub['text_box_id'] == otherbox, 'predicted_labels'].tolist())
                    try:
                        recognition['recognitions'].append(
                            {
                                'id': len(recognition['recognitions']),
                                'text': sub.loc[sub['text_box_id'] == otherbox, 'predicted_labels'].tolist()[0]
                                # 'text': sub.loc[sub['text_box_id'] == sub['text_box_id'].unique()[otherbox], 'predicted_labels'].tolist()[0]
                            })
                    except:
                        pass
                    # remember otherbox as already checked to avoid duplicates
                    already_checked.append(otherbox)

                already_checked.append(textboxid)
                # append the data element with all recognized words to the scene
                books[book_num_total]['scenes'].append(recognition)

            book_num_total += 1

    return books


def compare_results_string(results, queryString):
    """
    return a true if the result is somewhat close to the search string, false otherwise
    ---
    results: the results dictionary obtained from the googlebooksAPI() function
    queryString: the string used to obtain the results
    ---
    is_close: boolean
    """
    is_close = False
    num_matches = 0

    # get the words and number of words in the search string
    # print(queryString)
    # words = re.split(r'\+', queryString)
    words = queryString.split('+')
    try:
        words.remove('')
    except:
        pass
    # print(words)
    num_words = len(words)

    # get the words in the title
    # title_words = re.split(r' ', results['title'])
    title_words = results['title'].split(' ')
    try:
        title_words.remove('')
    except:
        pass
    # print('title_words: '+str(title_words))

    # get words in authors
    # author_words = []
    try:
        [author_words] = [re.split(r' ', a) for a in results['authors']]
    except:
        author_words = []
    try:
        author_words.remove('')
    except:
        pass
    # print('author_words: '+str(author_words))

    # combine title and author words
    book_words = title_words + author_words
    # print('book_words: '+str(book_words))

    # find number of exact word matches and close matches (1 character wrong is allowed)
    for word in words:
        # check whether single word is close to (at least) one word in the result
        for w in book_words:
            # print("word: "+str(word))
            # print('w: '+str(w))
            has_match = True if lev(word, w) <= 1 else False
            # adjst num matches
            if has_match:
                num_matches += 1

    # calculate ratio
    if num_words > 0:
        ratio = num_matches / num_words
    else:
        return True  # to reduce further computation time

    # set threshold based on th enumber of words in the query string
    if num_words <= 3:
        threshold = 1  # demand perfect match for small number of words in the query string
    else:
        threshold = 0.6

    # generate return value based on ratio and threshold
    if ratio >= threshold:
        is_close = True
        # print('yeah')
    else:
        is_close = False

    return is_close


def retrieve_results_from_API(csv_file, output_path, segmented_path):
    """
    Takes the csv file after the OCR step as input and produces a textfile "Recognized_books.txt" with all recognized books
    ---
    csv_file: string, location of the csv file generated by the OCR step
    output_path: the path were the textfile should be placed
    """

    frame = []
    collapsed = get_collapsed_data_from_csv(csv_file)

    book_list = get_book_list_from_csv(csv_file)
    segmented_books = get_segmented_book_list(segmented_path)
    shelf_num_list = get_num_per_shelf(segmented_books)
    shelf_iterator = 0
    shelf_book = -1

    # import data from the csv created by ocr part
    myBooks = import_from_csv(csv_file)
    # print(myBooks)
    print("This image contains " + str(
        len(myBooks)) + " recognized books.")  # is the same as the number of books in the csv file

    # iterate over all books in the data
    j = 1
    for book in myBooks:
        print("Retrieving information for book number " + str(j))
        # create a query string for the books
        queryString = niceQueryString(book, False)
        # send the query string to the google API
        results = googlebooksAPI(queryString, number=0, response=None)

        i = 0
        # try to get a result from the API
        while not results['success'] and i < 5:  # or instead of and, faster maybe
            results = googlebooksAPI(queryString, number=0, response=None)
            i += 1

        # get a result with word substitution
        if not results['success']:
            # print('I am here')
            queryString = niceQueryString(book, True)
            results = googlebooksAPI(queryString, number=0, response=None)

        # check whether the result is close to the searchstring
        is_close = compare_results_string(results, queryString)
        # try the next i best results if not close
        i = 1
        other_results = results  # to access 'response'
        while (not is_close) and results['success'] and i < 5:
            # get result i+1
            # print('I am there')
            # print(other_results['response'])
            other_results = googlebooksAPI(queryString, number=i, response=other_results['response'])
            # check whether new result is close
            is_close = compare_results_string(other_results, queryString)
            # if close overwrite results
            if is_close:
                results = other_results
                break
            # increase counter
            i += 1

        # print the results to a textfile
        # print('---')
        # Determine number to be included based on csv
        new_book = int(book_list[j - 1].strip("book-"))
        if (new_book <= shelf_book):
            shelf_iterator += 1

        matched = shelf_num_list[shelf_iterator] + new_book
        s = collapsed['shelf'].iloc[j - 1].replace('-', '_')
        b = collapsed['book'].iloc[j - 1].replace('-', '_')
        matched_name = ''
        for file in segmented_books:
            if s in file and b in file:
                matched_name = file
                break

        _, book_row = handleAPIoutput(results, queryString, output_path, number=j, matched_book=matched)
        row = [matched_name, *book_row]
        frame.append(row)
        shelf_book = new_book
        j += 1

    frame = pd.DataFrame(frame,
                         columns=['file', 'number', 'matched_book', 'query_string', 'title', 'authors', 'publisher',
                                  'published_date', 'identifiers', 'pagecount'])
    return frame


def determine_word_distance(predicted_label, dictionary, method='levenshtein',
                            invalid_chars=[' ', '-', '_', '!', '?', '.'], default_distance=100):
    """
    Determines how close a given word is to any word in a given dictionary.

    Args:
        predicted_label: The input word
        dictionary: The input dictionary (used for pyenchant)
        method: The method used to determine the distance of two words
            (possible methods: levenshtein, levenshtein-ratio, difflib, difflib-quick, difflib-real-quick)
        invalid_chars: Characters that shouldn't be in a suggested word from the dictionary
        default_distance: Distance of a word that has no suggestions (meaning no distance can be determined)

    Returns:
        The distance of the word to the next best word and the next best word
    """
    distance = 0
    correct_word = ''
    valid = dictionary.check(predicted_label)

    if valid:
        if not method == 'levenshtein': distance = 1
        # print("Word correct: "+predicted_label)
    else:
        suggestions = dictionary.suggest(predicted_label)
        # print("Suggestions unclean: {}".format(suggestions))
        if len(invalid_chars) > 0:
            # This cleans the suggested words of any unwanted characters
            # Takes about half a second extra for the entire .csv
            # Note that this might not be necessary as some of these characters could very well be in a title
            suggestions = [s for s in suggestions if not any(invalid in s for invalid in invalid_chars)]
        # print("Suggestions clean: {}".format(suggestions))
        if method == 'levenshtein':
            distance = default_distance  #
            # This is the default
            # print("USING LEVENSHTEIN DISTANCE")
            for s in suggestions:
                d = lev(s, predicted_label)
                if d < distance or distance < 0:
                    distance = d
                    correct_word = s
            # print("Final distance: {}, from word: {}, suggestions: {}".format(distance, predicted_label, str(suggestions)))

        elif method == 'levenshtein-ratio':
            # Levenshtein ratio apparently treats replacements more harshly than distance
            # print("USING LEVENSHTEIN RATIO")
            for s in suggestions:
                d = str_ratio(s, predicted_label)
                if d > distance or distance < 0:
                    distance = d
                    correct_word = s

        elif method == 'difflib' or method == 'difflib-quick' or method == 'difflib-real-quick':
            # print("USING DIFFLIB: {}".format(method))
            for s in suggestions:
                d = difflib.SequenceMatcher(None, s, predicted_label)

                if method == 'difflib':
                    d = d.ratio()
                elif method == 'difflib-quick':
                    d = d.quick_ratio()
                elif method == 'difflib-real-quick':
                    d = d.real_quick_ratio()

                if d > distance or distance < 0:
                    distance = d
                    correct_word = s

    return distance, correct_word


def choose_words_from_csv(csv_path, use_ratio=True, strip_length=1):
    """choose_words but with the path of a .csv file as input instead of a pandas dataframe"""
    f = pd.read_csv(csv_path)
    choose_words(f, use_ratio, strip_length)


def choose_words(df, use_ratio=True, strip_length=1):
    """
    Determines which words per book in the input dataframe to choose as search string.
    This is based on the "correctness" of all words in a given rotation.
    The resulting search string contains vertical and horizontal text and chooses between 0° and 180° for horizontal text
    as well as 90° and 270° for vertical text

    Args:
        df: The input dataframe, based on the output .csv of the OCR module
        use_ratio: Whether the code should use ratio or distance as a metric for correctness (i.e. levenshtein vs. all other methods)
        strip_length: How long a word needs to be to even be included in the output search string

    Returns:
        A list of search strings with one string per book in the input dataframe
    """
    df = df[['predicted_labels', 'shelf', 'book', 'rotation', 'text_box_id', 'word_distance']]
    search_strings = []
    for shelf in df['shelf'].unique():
        single_shelf = df[df['shelf'] == shelf]
        # print(shelf)
        for book in single_shelf['book'].unique():
            # print(book)
            single_book = single_shelf[single_shelf['book'] == book]
            # Check average word distance for each rotation, choose the smaller one
            rot_0 = single_book[single_book['rotation'] == 0]
            if len(rot_0.index) > 0:
                average_0 = rot_0['word_distance'].sum() / len(rot_0.index)
            else:
                average_0 = -1

            rot_90 = single_book[single_book['rotation'] == 90]
            if len(rot_90.index) > 0:
                average_90 = rot_90['word_distance'].sum() / len(rot_90.index)
            else:
                average_90 = -1

            rot_180 = single_book[single_book['rotation'] == 180]
            if len(rot_180.index) > 0:
                average_180 = rot_180['word_distance'].sum() / len(rot_180.index)
            else:
                average_180 = -1

            rot_270 = single_book[single_book['rotation'] == 270]
            if len(rot_270.index) > 0:
                average_270 = rot_270['word_distance'].sum() / len(rot_270.index)
            else:
                average_270 = -1

            # print("Word distance averages for {}:".format(book))
            # print("Rotation 0: {}\nRotation 90: {}\nRotation 180: {}\nRotation 270: {}".format(average_0, average_90, average_180, average_270))

            vertical = -1
            horizontal = -1
            if use_ratio:
                if average_0 >= average_180:
                    vertical = 0
                else:
                    vertical = 180
                if average_90 >= average_270:
                    horizontal = 90
                else:
                    horizontal = 270
            else:
                if average_0 >= average_180:
                    vertical = 180
                else:
                    vertical = 0
                if average_90 >= average_270:
                    horizontal = 270
                else:
                    horizontal = 90

            searchstring = ''
            vertical_words = (single_book[single_book['rotation'] == vertical])['predicted_labels'].tolist()
            vertical_words = [v for v in vertical_words if len(v) > strip_length]
            # print(vertical_words)
            vertical_words = '+'.join(vertical_words)
            horizontal_words = (single_book[single_book['rotation'] == horizontal])['predicted_labels'].tolist()
            horizontal_words = [h for h in horizontal_words if len(h) > strip_length]
            # print(horizontal_words)
            horizontal_words = '+'.join(horizontal_words)
            searchstring = vertical_words + '+' + horizontal_words
            search_strings.append(searchstring)
            # print(searchstring)

            # print(single_book)
        # print('\n')
    return search_strings


def retrieve_books(csv_path, output_path, segmented_path):
    """
    Function based on Ludwig's retrieve_results_from_API except rotation-based search strings are considered first

    Args:
        csv_path: Path to the .csv from the OCR module
        output_path: path of where to put the resulting text file
        segmented_path: path to the folder with the output from the segmentation module

    Returns:
        A pandas dataframe that includes the information from GoogleBooksAPI as rows per book request
    """
    frame = []
    book_dict = import_from_csv(csv_path)  # Ludwigs dictionary building function

    f = pd.read_csv(csv_path)
    collapsed = get_collapsed_data(f)

    book_list = get_book_list_from_frame(f)
    segmented_books = get_segmented_book_list(segmented_path)
    shelf_num_list = get_num_per_shelf(segmented_books)
    shelf_iterator = 0
    shelf_book = -1

    d = enchant.Dict('en')
    word_distances = [determine_word_distance(pred, d, method='levenshtein-ratio', invalid_chars=[' ', ]) for pred in
                      f['predicted_labels']]
    f = pd.concat([f, pd.DataFrame(word_distances, columns=('word_distance', 'corrected_word'))], axis=1)
    # f.to_csv('/content/words.csv') # If you want to save it
    search_strings = choose_words(f, strip_length=2)
    r = re.compile(r'\d')
    search_strings = [r.sub('', s) for s in search_strings]

    j = 1
    for book in book_dict:
        queryString = search_strings[j - 1]
        res = googlebooksAPI(queryString, number=0, response=None)
        # The following is taken directly from Ludwig's implementation
        i = 0
        while not res['success'] and i < 5:
            res = googlebooksAPI(queryString, number=0, response=None)
            i += 1

        if not res['success']:
            # print('I am here')
            queryString = niceQueryString(book, True, string=queryString)
            res = googlebooksAPI(queryString, number=0, response=None)

        # check whether the result is close to the searchstring
        # TODO tweak this a bit since the search string from the rotation only part ("search_strings") don't look like Ludwig's.. probably
        is_close = compare_results_string(res, queryString)
        # try the next i best results if not close
        i = 1
        other_results = res  # to access 'response'

        while (not is_close) and res['success'] and i < 5:
            # get result i+1
            # print('I am there')
            # print(other_results['response'])
            other_results = googlebooksAPI(queryString, number=i, response=other_results['response'])
            # check whether new result is close
            is_close = compare_results_string(other_results, queryString)
            # if close overwrite results
            if is_close:
                res = other_results
                break
            # increase counter
            i += 1

        # print the results to a textfile
        # print('---')
        new_book = int(book_list[j - 1].strip("book-"))

        if (new_book <= shelf_book):
            shelf_iterator += 1

        matched = shelf_num_list[shelf_iterator] + new_book
        s = collapsed['shelf'].iloc[j - 1].replace('-', '_')
        b = collapsed['book'].iloc[j - 1].replace('-', '_')
        matched_name = ''
        for file in segmented_books:
            if s in file and b in file:
                matched_name = file
                break

        _, book_row = handleAPIoutput(res, queryString, output_path, number=j, matched_book=matched)
        row = [matched_name, *book_row]
        frame.append(row)
        shelf_book = new_book
        j += 1

    frame = pd.DataFrame(frame,
                         columns=['file', 'number', 'matched_book', 'query_string', 'title', 'authors', 'publisher',
                                  'published_date', 'identifiers', 'pagecount'])
    return frame


def get_book_list_from_csv(csv_path):
    """ Same as get_book_list_from_frame but with the path to a .csv as input """
    df = pd.read_csv(csv_path)
    return get_book_list_from_frame(df)


def get_book_list_from_frame(df):
    """
    Retrieves the list of books from a dataframe, with book names being unique across shelves *only*.
    This only works in Python 3.5+.
    """
    shelves = df['shelf'].unique()
    books = []
    for s in shelves:
        books = [*books, *df[df['shelf'] == s]['book'].unique()]

    return books


def get_segmented_book_list(segmented_path):
    """ Returns a list of the file names of the output from the segmentation module (segmented book spine images) """
    segmented_books = [f for f in listdir(segmented_path) if isfile(join(segmented_path, f)) and '_book_' in f]
    return segmented_books


def get_num_per_shelf(files):
    """ Honestly I forgot what this does, but it's important """
    curr_shelf = "shelf_" + files[0].split("_shelf_")[1].split("_")[0]
    shelf_numbers = [0, ]
    iterator = 0
    for i in range(len(files)):
        if not curr_shelf in files[i]:
            curr_shelf = "shelf_" + files[i].split("_shelf_")[1].split("_")[0]
            shelf_numbers.append(iterator)
        iterator += 1
    shelf_numbers.append(iterator)

    return shelf_numbers


def get_collapsed_data_from_csv(csv_path):
    """ Same as get_collapsed_data but with the path to a .csv file as input """
    f = pd.read_csv(csv_path)
    return get_collapsed_data(f)


def get_collapsed_data(frame):
    """
    Takes output from the OCR module and groups the book names for each shelf so that the resulting dataframe includes only one
    book entry per book and shelf (unlike one book entry for each textbox within the dataframe).
    The resulting dataframe keeps the first row for each book and should have as many rows as there are recognized books.
    """
    small_data = []
    for shelf in frame['shelf'].unique():
        small_data.append(frame[frame['shelf'] == shelf].groupby('book', as_index=False).nth(0))
    small_data = pd.concat(small_data)
    small_data = small_data.reset_index()
    return small_data
