import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.decomposition import PCA

__all__: list[str] = ["biplot", "screeplot"]


def biplot(
    axis: Axes,
    names: tuple[str],
    colors: tuple[str],
    arrow_props: dict[str, str],
    label_props: dict[str, str],
    axis_ticks: bool,
    model: PCA,
    projections: NDArray[np.integer | np.floating],
) -> Axes:
    """
    Lookup the following sources for implementation details::
    - https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    - https://plotly.com/python/pca-visualization/
    - https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot

    """

    if (len(names) == len(colors)) or (len(colors) == 1):
        raise ValueError("Mismatch in the lengths of names and colours!")
    axis.scatter(x=projections[:, 0], y=projections[:, 1], edgecolor="black", c=,#
                  alpha=0.75)
    for label, (dx, dy) in zip(names, np.abs(projections).max(axis=0) * model.components_.T):
        axis.annotate(  # type: ignore
            text="",
            xytext=(0, 0),  # origin of the arrow
            xy=(dx, dy),  # horizontal and vertical distance to the distal end of the vector from the origin
            arrowprops=arrow_props,
        )
        axis.text(x=dx, y=dy, s=label, fontdict=label_props)  # label the arrow tip # type: ignore
        axis.set_xlabel(f"PC1 ({model.explained_variance_ratio_[0] * 100:.3f}%)")  # type: ignore
        axis.set_ylabel(f"PC2 ({model.explained_variance_ratio_[1] * 100:.3f}%)")  # type: ignore

        if not axis_ticks:
            axis.set_xticks([])  # type: ignore
            axis.set_yticks([])  # type: ignore

    return axis


# INCOMPLETE
def screeplot(axis: Axes, color: str, line_props: dict[str, str], model: PCA) -> Axes:
    """ """
    axis.plot(model.explained_variance_, *line_props, color=color)  # type: ignore
    return axis
