from  matplotlib.axes import Axes
from sklearn.decomposition import PCA


def draw_pca_loadings(names: tuple[str], colors: tuple[str], vprops: dict[str, str], axis: Axes, model: PCA) -> None:
    """
    Lookup the following sources for implementation details::

    - https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    - https://plotly.com/python/pca-visualization/
    - https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot

    components define the direction of the vector
    explained variance defines the squared-length of the vector
    """

    assert(len(names)==len(colors)), ""

    for (label, color, direction, squared_length) in zip(names, colors, model.components_, model.explained_variance_):

    axis.annotate(arrowprops=vprops, )
    return
