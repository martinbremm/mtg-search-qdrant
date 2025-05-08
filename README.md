# MTG Search using Qdrant

This is a prototype for a semantic search engine, powered by Qdrant. The idea is to enhance the search for specific MTG cards, by incorporating semantic search based on embeddings of the cards rules text.

![image](https://github.com/martinbremm/mtg-search-qdrant/assets/79272801/3cb0a9b5-6206-4332-8465-6f313305287f)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

1. **Install Pyenv:** If you haven't already installed Pyenv, you can do so by following the instructions in the [Pyenv GitHub repository](https://github.com/pyenv/pyenv#installation).

2. **Install Python 3.11:** Once Pyenv is installed, you can install Python 3.11 using the following command:
   ```sh
   pyenv install 3.11.0

3. **Create a virtualenv using pyenv**:
   ```sh
   pyenv virtualenv 3.11.0 mtg-search-qdrant

4. **Activate the virtualenv**:
   ```sh
   pyenv activate mtg-search-qdrant

5. **Install the required Python dependencies**:
   ```sh
   pip install -r requirements.txt

## Usage

 1. **Check Docker installation**: Tested on Docker version 20.10.24:
    ```sh
    docker --version

2. **Setup MTG data**: Run setup script to download the MTG card data and preprocess it:
   ```sh
   sh scripts/prepjson.sh

3. **Setup Qdrant container**: Run setup script to initialize Docker container and load the embedded MTG card data into a collection:
   ```sh
   sh scripts/setup.sh

4. **Run Streamlit App**: After indexing, run the Streamlit App to search through the collection of MTG cards:
   ```sh
   streamlit run src/app.py


## License

This project is released under the MIT license, allowing for freedom to use, modify, and distribute the software with minimal restrictions.
