# code from - https://r-graph-gallery.com/130-ring-or-donut-chart.html
doughnut <- function (x,
                      edges = 200,
                      outer.radius = 0.8,
                      inner.radius = 0.6,
                      clockwise = FALSE,
                      init.angle = if (clockwise) 90 else 0,
                      density = NULL,
                      angle = 45,
                      col = NULL,
                      border = FALSE,
                      lty = NULL,
                      main = NULL,
                      new = FALSE)
{

    if (!is.numeric(x) || any(is.na(x) | x < 0)) stop("'x' values must be positive.")

    x <- c(0, cumsum(x) / sum(x)) # slices as cumulative fractions
    dx <- diff(x)
    nx <- length(dx)

    if (new) plot.new()

    pin <- par("pin")
    xlim <- ylim <- c(-1, 1)

    if (pin[1L] > pin[2L]) xlim <- (pin[1L]/pin[2L]) * xlim
    else ylim <- (pin[2L]/pin[1L]) * ylim

    plot.window(xlim, ylim, "", asp = 1) # plot a square window (with y / x aspect ratio = 1.00)
    if (is.null(col))
        col <- if (is.null(density)) palette() else par("fg")

    col <- rep(col, length.out = nx)
    border <- rep(border, length.out = nx)
    lty <- rep(lty, length.out = nx)
    angle <- rep(angle, length.out = nx)
    density <- rep(density, length.out = nx)

    twopi <- if (clockwise) -2 * pi else 2 * pi

    t2xy <- function(t, radius) {
        t2p <- twopi * t + init.angle * pi/180
        list(x = radius * cos(t2p), y = radius * sin(t2p))
    }


    for (i in 1L:nx) {

        # plot each slice of the doughnut chart
        n <- max(2, floor(edges * dx[i]))
        P <- t2xy(seq.int(x[i], x[i + 1], length.out = n), outer.radius)
        polygon(c(P$x, 0), c(P$y, 0), density = density[i], angle = angle[i], border = border[i], col = col[i], lty = lty[i])

        # slices of the inner white disc
        Pin <- t2xy(seq.int(0, 1, length.out = n*nx), inner.radius)
        polygon(Pin$x, Pin$y, density = NA, angle = angle[i], border = NA, lty = lty[i], col = "white")
    }

    invisible(NULL)
}
