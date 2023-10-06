from src.data.scorer import make_thresholded_tsv
from src.data.build_subdataset import build_subdataset


def score_pipe(micron, pipe_args):
        for scor_type, thr, filename, on_test in pipe_args:
            print(
                "_____________________________________________________________________"
            )
            make_thresholded_tsv(
                micron=micron,
                stype=scor_type,
                thr=float(thr),
                on_test_set=on_test,
                filename=filename,
                sfilename="",
            )

if __name__ == "__main__":
    # scores = [float(f"0.{n}") for n in range(10)]
    # pipe_args = [("nuclei", s, "", True) for s in scores]

    # type = "black"
    # thr = "0.2"
    # input = ""
    # on_test = 0
    # pipe_args = [(type, thr, input, on_test)]
    # subset_filename = f"scored_{type}_{thr}_{input}" if input != "" else f"scored_{type}_{thr}"

    pipe_args = [("black","0.2","",1),("nuclei", "0.1", "scored_black_0.2", 1)]
    score_pipe(micron="400", pipe_args=pipe_args)
