from __future__ import annotations
from vision2code.figures.make_benchmark_stats import reproduce as stats
from vision2code.figures.make_error_analysis import reproduce as errors
from vision2code.figures.make_leaderboard_tables import reproduce as leaderboard
from vision2code.figures.make_self_training_figures import reproduce as ablations
def main():
    outputs=[]
    for fn in [stats,leaderboard,errors,ablations]: outputs.extend(fn())
    for p in outputs: print(p)
if __name__=='__main__': main()
