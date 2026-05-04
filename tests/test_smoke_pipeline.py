from pathlib import Path
from image2code.figures.make_leaderboard_tables import reproduce
from image2code.rendering.render_python import render_matplotlib_code

def test_smoke_render_and_tables(tmp_path: Path):
    result=render_matplotlib_code('import matplotlib.pyplot as plt\nplt.plot([0,1],[0,1])\nplt.savefig(OUTPUT_PATH)\n', tmp_path/'render.png')
    assert result['render_success'] is True
    outputs=reproduce(tmp_path/'tables')
    assert any(p.name=='main_leaderboard.csv' for p in outputs)
