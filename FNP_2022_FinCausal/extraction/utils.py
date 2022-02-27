from nltk import sent_tokenize
import os
from bs4 import BeautifulSoup
from secedgar.parser import MetaParser
import re
import pandas as pd
from datetime import date
from pathlib import Path
from secedgar import filings, FilingType
objectParser = MetaParser()

cwd = Path(__file__)
root = cwd.parent
MODE = True

def save_fillings(reference: str) -> GeneratorExit:
    """
    Declare a Name and a valide email as the user_agent
    reference: cik
    return: filling if MODE=False, dumps to disk otherwise
    """

    print("REQUEST 10-K FOR, ", reference)
    try:
        company_filings = filings(start_date=date(2017, 1, 1),
                                  cik_lookup=reference,
                                  filing_type=FilingType.FILING_10K,
                                  user_agent="Name (email)")

        if MODE and company_filings:
            os.makedirs(root / "fillings", exist_ok=True)

            company_filings.save(root / "fillings")
            print('fillings saved to disk', root / "fillings")
        else:
            return company_filings
    except Exception as E:
        print(E)
        pass





def getMgmtDisc(fpath: Path) -> tuple:

    """
    param: data -> bs4 extraction from SEC
    return: list of sentences in MDAS section, document_type
    """
    with open(fpath, "r") as f:
        data = f.read()
        try:
            soup = BeautifulSoup(data, features="html.parser")

            # The document tags contain the various components of the total 10K filing pack
            for filing_document in soup.find_all('document'):
                # The 'type' tag contains the document type
                document_type = filing_document.type.find(text=True, recursive=False).strip()
                if document_type and document_type == "10-K":  # Once the 10K text body is found
                    print('DOC TYPE ', document_type)
                elif document_type and document_type != "10-K":
                    print('DOC TYPE ', document_type)
                else:
                    print('DOC TYPE ', 'NO META TYPE')
                    document_type = 'NoDocType'

                # Grab and store the 10K text body
                TenKtext = filing_document.find('text').extract().text

                # Set up the regex pattern
                matches = re.compile(r'(item\s(7[\.\s]|8[\.\s])|'
                                     'discussion\sand\sanalysis\sof\s(consolidated\sfinancial|financial)\scondition|'
                                     '(consolidated\sfinancial|financial)\sstatements\sand\ssupplementary\sdata)',
                                     re.IGNORECASE)

                matches_array = pd.DataFrame([(match.group(), match.start()) for match in matches.finditer(TenKtext)])

                # Set columns in the dataframe
                matches_array.columns = ['SearchTerm', 'Start']

                # Get the number of rows in the dataframe
                Rows = matches_array['SearchTerm'].count()

                # Create a new column in 'matches_array' called 'Selection'
                # and add adjacent 'SearchTerm' (i and i+1 rows) text concatenated
                count = 0  # Counter to help with row location and iteration
                while count < (Rows - 1):  # Can only iterate to the second last row
                    matches_array.at[count, 'Selection'] = (matches_array.iloc[count, 0] + matches_array.iloc[
                        count + 1, 0]).lower()  # Convert to lower case
                    count += 1

                # Set up 'Item 7/8 Search Pattern' regex patterns
                matches_item7 = re.compile(r'(item\s7\.discussion\s[a-z]*)')
                matches_item8 = re.compile(r'(item\s8\.(consolidated\sfinancial|financial)\s[a-z]*)')

                # Lists to store the locations of Item 7/8 Search Pattern matches
                Start_Loc = []
                End_Loc = []

                # Find and store the locations of Item 7/8 Search Pattern matches
                count = 0  # Set up counter

                while count < (Rows - 1):  # Can only iterate to the second last row

                    # Match Item 7 Search Pattern
                    if re.match(matches_item7, matches_array.at[count, 'Selection']):
                        # Column 1 = 'Start' columnn in 'matches_array'
                        Start_Loc.append(matches_array.iloc[
                                             count, 1])  # Store in list => Item 7 will be the starting location (column '1' = 'Start' column)

                    # Match Item 8 Search Pattern
                    if re.match(matches_item8, matches_array.at[count, 'Selection']):
                        End_Loc.append(matches_array.iloc[count, 1])

                    count += 1

                TenKItem7 = TenKtext[Start_Loc[-1]:End_Loc[-1]]

                TenKItem7 = TenKItem7.strip()  # Remove starting/ending white spaces
                TenKItem7 = TenKItem7.replace('\n', ' ')  # Replace \n (new line) with space
                TenKItem7 = TenKItem7.replace('\r', '')  # Replace \r (carriage returns-if you're on windows) with space
                TenKItem7 = TenKItem7.replace(' ', ' ')  # Replace " " (a special character for space in HTML) with space
                TenKItem7 = TenKItem7.replace(' ', ' ')  # Replace " " (a special character for space in HTML) with space
                TenKItem7 = re.sub(r'( {2})\1\1', '\n', TenKItem7)
                TenKItem7 = re.sub(r'(    )', '\n', TenKItem7)
                TenKItem7 = TenKItem7.replace(' ', ' ')

                while '  ' in TenKItem7:
                    TenKItem7 = TenKItem7.replace('  ', ' ')  # Remove extra spaces

                TenKItem7list = sent_tokenize(TenKItem7)
                print("TEXT LENGTH: ", len(TenKItem7))

                return TenKItem7list, document_type


        except Exception as E:
            print(E)
            return None, None
            pass




