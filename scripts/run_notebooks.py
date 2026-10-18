from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import os
def main():
    os.chdir(ROOT)
    for path in sorted((ROOT/'notebooks').glob('[0-9]*.ipynb')):
        namespace={'__name__':'__main__','__file__':str(path)}
        for index,cell in enumerate(json.loads(path.read_text(encoding='utf-8'))['cells']):
            if cell['cell_type']=='code':exec(compile(''.join(cell['source']),f'{path.name}:cell{index}','exec'),namespace)
        print(f'{path.name}: completed',flush=True)
if __name__=='__main__':main()
