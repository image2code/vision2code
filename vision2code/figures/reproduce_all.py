from __future__ import annotations
from vision2code.figures.make_benchmark_stats import reproduce as stats
def main():
    outputs=[]
    for fn in [stats]: outputs.extend(fn())
    for p in outputs: print(p)
if __name__=='__main__': main()
