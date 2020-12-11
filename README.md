# Search Engine IRWA2020
## Information Retrieval and Web Analytics
**Authors**

Benet i Cugat, Berta - 204721

Budan, Maria Elena - 206330

__________________________

### Repository structure

[data/](https://github.com/bertabenet/search_engine_IRWA2020/tree/main/data) Contains the data file `final_data.json` with all the tweets in json format. For specific information go to [Data information](#data-info).

[notebook/](https://github.com/bertabenet/search_engine_IRWA2020/tree/main/notebook) Contains all three `RQ*.ipynb` files with the answers for each section.

[other-outputs/](https://github.com/bertabenet/search_engine_IRWA2020/tree/main/other-outputs) Contains output files from each of the python notebooks. `RQ1_*.tsv` are the ones for RQ1 and `RQ2_*.tsv` are the ones for RQ2.

[search-engine/](https://github.com/bertabenet/search_engine_IRWA2020/tree/main/search-engine) Contains the following files: `config.ini` that incorporates the credentials for the tweepy API and the keywords to scrape tweets, `search_engine.py` the search engine itself and `utils.py` the python file containing all the functions used in the search engine.

__________________________

### <a name="data-info"></a> Data information

| Type | Date | Total tweets |
| :---: | :---: | :---: |
| Scraped tweets | 06/12/2020 | 200200 |
| Hydrated tweets | 07/12/2020 | 194434 |

__________________________

### Requirements to run the search engine and the notebooks
`argparse`, `array`, `chronometer`, `collections`, `config`, `configparser`, `copy`, `csv`, `datetime`, `gensim`, `glob`, `igraph`, `implicit`, `json`, `jsonlines`, `math`, `matplotlib`, `nltk`, `numpy`, `os`, `pandas`, `re`, `scipy`, `sklearn`, `sys`, `time`, `twarc` and `tweepy`.
