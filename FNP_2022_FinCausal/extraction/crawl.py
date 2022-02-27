"""
source: https://github.com/sec-edgar/sec-edgar

"""

from utils import *
MODE = False


if __name__ == '__main__':

    cwd = Path(__file__)
    root = cwd.parent

    ticker = pd.read_csv(root / "data/tickers2extract.txt", sep="\t", header=None)
    ticker.columns = ["cik", "meanmve"]
    tickers = pd.read_csv(root / "data/tickers.txt", sep="\t", header=None)
    tickers.columns = ["tick", "cik"]

    ref = ticker.merge(tickers, on='cik')
    print("REFERENCES")
    print(ref)
    references = ref.tick.tolist()
    MDAcount = 0

    for reference in references:
        save_fillings(reference)

    pathTree = root / "fillings"
    print(pathTree)
    listFile = pathTree.glob("**/*.txt")
    MDAcount = 0
    XMLcount = 0
    for i in list(listFile):
        print("INPUT    ", i)
        fpath = root / i
        mda, doctype = getMgmtDisc(i)
        tick = fpath.parent.parent.as_posix().split("/")[-1]
        cik = "".join(str(i) for i in ref.cik[ref.tick == tick].values)
        print("CIK      ", tick)
        print("DATA     ", fpath.stem)
        dpath = fpath.parent.parent.parent.parent / "MDAS"

        if mda and len(mda) > 1:
            LOCAL_FILE = dpath / tick
            if not os.path.exists(LOCAL_FILE):
                os.makedirs(LOCAL_FILE)
            with open(os.path.join(LOCAL_FILE / fpath.stem), "w", encoding='utf-8') as fout:
                for i in mda:
                    fout.write(i + '\n')
                print("OUTPUT   ", LOCAL_FILE / fpath.stem)
                MDAcount += 1


        elif mda and len(mda) == 0:
            print("EMPTY MDA, NOT DUMPED")
            with open('data/10K_summary.csv', "a") as file_object:
                listed = [tick, cik, fpath.stem, doctype, 'EmptyMDA']
                string_ = ",".join(element for element in listed)
                file_object.write("\n")
                file_object.write(string_)
        else:
            print("NO MDA, NOT DUMPED")
            with open('data/10K_summary.csv', "a") as file_object:
                listed = [tick, cik, fpath.stem, doctype, 'NoMDA']
                string_ = ",".join(element if element else 'NaN' for element in listed)
                file_object.write("\n")
                file_object.write(string_)
        print()
        print(MDAcount)
