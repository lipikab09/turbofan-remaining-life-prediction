"""python run.py [--fd FD001|FD002|FD003|FD004] [--synthetic] [--open]"""
import argparse, sys, webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from pipeline import run

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--fd', default='FD001'); a.add_argument('--synthetic', action='store_true'); a.add_argument('--open', action='store_true')
    a = a.parse_args(); root = Path(__file__).parent
    run(a.fd, root / 'data', root / 'results', a.synthetic)
    if a.open: webbrowser.open((root / 'dashboard' / 'index.html').as_uri())
