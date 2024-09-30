#!/bin/bash
sh ./clean.sh
pip install -r requirements.txt
mkdir "tmp"
cp Yaguar.g4 tmp
cd tmp
antlr4 -Dlanguage=Python3 Yaguar.g4
touch __init__.py
echo "from geodio.core.parser.tmp.YaguarLexer import *
from geodio.core.parser.tmp.YaguarParser import *
from geodio.core.parser.tmp.YaguarListener import *" >> __init__.py