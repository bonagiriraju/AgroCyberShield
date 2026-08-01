import argparse
def main():
 p=argparse.ArgumentParser(prog='agrocyber'); p.add_argument('--version',action='store_true'); a=p.parse_args();
 if a.version: print('1.0.0')
 else: p.print_help()
