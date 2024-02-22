curl -o mtgjson/data/cards.csv.zip https://mtgjson.com/api/v5/AllPrintingsCSVFiles.zip 

unzip -o mtgjson/data/cards.csv.zip -d mtgjson/data/

rm mtgjson/data/cards.csv.zip

python mtgjson/prepjson.py