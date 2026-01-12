square <- function(x) {
    if (x%%2) { warning("Odd number!") }
    else { message("Even number!") }
    x^2
}

# the function apecified to handle signals (error, warning, message) should take a single argument of type condition!!!

sqwrapper <- function(x) {
    res <- 0
    withCallingHandlers(
        res <<- square(x),
        warning = function (condition) {
            invokeRestart(r = "muffleWarning")
        },
        message = function (condition) {
            invokeRestart(r = "muffleWarning")
        }
    )
    res
}

sqwrapper(12)

sqtrycatch <- function(x) {
    res <- 0
    tryCatch(
        warning = function (condition) { invokeRestart(r = "muffleWarning") },
        message = function (condition) { invokeRestart(r = "muffleWarning") },
        res <<- square(x)
    )
    res
}

sqtrycatch(12)

tryCatch(
    message = function(condition) cat("Caught a message!\n"),
    {
        message("Someone there?")
        message("Why, yes!")
    }
)


withCallingHandlers(
    message = function(cnd) cat("Caught a message!\n"),
    {
        message("Someone there?")
        message("Why, yes!")
    }
)
#> Caught a message!
#> Someone there?
#> Caught a message!
#> Why, yes!


sqtrycatch_v2 <- function(x) {
    res <- 0
    tryCatch(
        warning = function (condition) { print(conditionMessage(condition)) },
        message = function (condition) { print(conditionMessage(condition)) },
        res <<- square(x)
    )
    res
}

sqtrycatch_v2(131)

sqtrycatch_v3 <- function(x) {
    res <- 34
    tryCatch(
        warning = function (condition) { print(conditionMessage(condition)) },
        message = function (condition) { print(conditionMessage(condition)) },
        {   res_ <- 11
            withCallingHandlers(
                warning = function (condition) { },
                message = function (condition) { },
                res_ <<- square(x)
            )
            res <<- res_
        }
    )
    res
}

sqtrycatch_v3(123)

sqtrycatch_v4 <- function(x) {
    res <- 20
    withCallingHandlers(
        warning = function (condition) { },
        message = function (condition) { },
        tryCatch(
            expr = { res <<- square(x) },
            warning = function (condition) NULL,
            message = function (condition) NULL
        )
    )
    res
}

sqtrycatch_v4(176)

# however, this always works
withCallingHandlers(suppressWarnings(warning("hi")), warning = function(w) {
    print(w)
})


suppressMessages(suppressWarnings(square(12)))
suppressMessages(suppressWarnings(square(11)))


sums <- 0

withCallingHandlers({ sums <<- square(12)},
                    warning = function(cnd) {},
                    message = function(cond) {})

sums

tryCatch(expr = { sums <<- square(11)}, warning = function(cnd) {}, message = function(cond) {})

sums

shit_happened <- FALSE
tryCatch(
    withCallingHandlers(expr = { sums <<- square(13) }, warning = function(cond) { shit_happened <<- TRUE }, message = function(cond) { }),
    warning = function(cond) { }
)

shit_happened
sums

# Source - https://stackoverflow.com/a
# Posted by r2evans, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-13, License - CC BY-SA 4.0

withCallingHandlers(y <- sqrt(-1), warning = function(w) invokeRestart("muffleWarning"))
y


sqrt(-1)

sqwrapper_v2 <- function(x) {
    withCallingHandlers(
        res <- square(x),
        warning = function (condition) invokeRestart("muffleWarning"), # use muffleWarning for warnings
        message = function (condition) invokeRestart("muffleMessage") # use muffleMessage for messages
    )
    res
}

sqwrapper_v2(8)
sqwrapper_v2(7)

even_sq <- vector(mode = "numeric", length = 1000)

for (i in 1:1000) {
    res <- 0
    is_odd <- FALSE

    repeat {
        withCallingHandlers(
            warning = function(cond) { # square() will warn for odd numbers
                is_odd <<- TRUE
                invokeRestart("muffleWarning")
            },
            message = function(cond) {
                is_odd <<- FALSE
                invokeRestart("muffleMessage") # for even numbers
            },
            res <<- square(sample(1:100, size = 1))
        )
        if (!is_odd) break
    }
    even_sq[i] <- res
}

for (i in 1:1000) {
    res <- 0
    is_odd <- FALSE

    withCallingHandlers(
        warning = function(cond) { # square() will warn for odd numbers
            is_odd <<- TRUE
            invokeRestart("muffleWarning")
        },
        message = function(cond) {
            is_odd <<- FALSE
            invokeRestart("muffleMessage") # for even numbers
        },
        repeat {
            res <<- square(sample(1:100, size = 1))
            if (!is_odd) break
        }
    )
    even_sq[i] <- res
}
