import sys
import pandas as pd
import altair as alt
from src.transformer import subsequent_mask


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    chart = alt.Chart(LS_data)\
        .mark_rect()\
        .properties(height=250, width=250)\
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )

    chart.save('figures/example_mask.png', scale_factor=2.0)


if __name__ == "__main__":
    func_name = sys.argv[1]
    getattr(sys.modules[__name__], func_name)()
