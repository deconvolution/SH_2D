using Documenter
using SH_2D

makedocs(
    sitename = "SH_2D",
    format = Documenter.HTML(),
    modules = [SH_2D]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
