import sys
'''
Program that checks the file with the list of books and returns whether a certain book is in the file or not.
It returns the full title found and which number in the list the book has so it will be easier to find.
The book to search for is specified by the user via command line and can be author or title. Type "quit" to end.

Assumptions: The search string is the perfect title/auther name
             No other book info contains the same words in that order
             Capital letters in the beginning of title/author is important!
'''

class Search:
    def searchWord(word):
        # Open .txt file with list of books
        with open('Recognized_books.txt','r') as f:
            # Get list of strings (one string for each line in the .txt file)
            lines = f.readlines()
        # Loop over list of strings to search for the title
        bookNumber = 1
        line = 0
        for item in lines:
            line += 1
            if line == 4:
                title = item

            # For each new book
            newBook = item.find("---")
            if newBook != -1:
                bookNumber +=1
                line = 0

            pos = item.find(word)
            # If pos is NOT -1 that means the word is found
            if pos != -1:
                f.close()
                return True, bookNumber, title

        # If after looping over the file the word is not found
        f.close()
        return False, 0, "nothing"

def main():
    while True:
        word = input("What is the author/title you want to search for. (write "'quit'" to end program): ")
        if word == 'quit':
            print('Ending program' '\n')
            exit()
        bool, bookNumber, title = Search.searchWord(word)
        if bool==True:
            print('\n''The following book was found in the shelf as book number', bookNumber, ':', title)
        else:
            print('\n''The book is NOT in the shelf! Remember to use capital letters correctly in the search.' '\n')

main()
