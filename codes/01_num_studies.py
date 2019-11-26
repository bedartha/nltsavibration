#! /usr/bin/env python
"""
Plots the number of NLTSA and RP studies per year
=================================================

"""

# Created: Fri Dec 14, 2018  03:46pm
# Last modified: Thu Feb 21, 2019  03:12pm
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
import matplotlib.pyplot as pl

# matplotlib text params
pl.rcParams["text.usetex"] = True
pl.rcParams["font.family"] = ["serif"]

if __name__ == "__main__":
    # load data
    DATPATH = "../data/publication_lists/wos/"
    rpall = np.genfromtxt(DATPATH + "recurrenceplots_allfields.txt",
                          delimiter=",", dtype="int", skip_header=1)
    rpall = {"years": rpall[1:, 0],
             "count": rpall[1:, 1]
            }
    rpttl = np.genfromtxt(DATPATH + "recurrenceplots_title.txt",
                          delimiter=",", dtype="int", skip_header=1)
    rpttl = {"years": rpttl[1:, 0],
             "count": rpttl[1:, 1]
            }
    nltsall = np.genfromtxt(DATPATH + "nonlineartimeseries_allfields.txt",
                            delimiter=",", dtype="int", skip_header=1)
    nltsall = {"years": nltsall[1:, 0],
               "count": nltsall[1:, 1]
              }
    nltsttl = np.genfromtxt(DATPATH + "nonlineartimeseries_title.txt",
                            delimiter=",", dtype="int", skip_header=1)
    nltsttl = {"years": nltsttl[1:, 0],
               "count": nltsttl[1:, 1]
              }

    # set up figure
    fig = pl.figure(figsize=[7.480315, 3.543307])   # 190 mm wide, 90 mm tall
    # fig = pl.figure(figsize=[3.543307, 7.086614])   # 90 mm wide, 180 mm tall
    lm, bm, wd, ht = 0.100, 0.175, 0.40, 0.775
    ax1 = fig.add_axes([lm, bm, wd, ht])
    ax2 = fig.add_axes([lm + wd, bm, wd, ht])
    axlabfs, tiklabfs = 12, 11
    all_clr, ttl_clr = "MediumTurquoise", "GoldenRod"
    splabs = ["A", "B"]
    legbba = [[0.75, 0.025], [0.65, 0.025]]

    # plot
    ax1.barh(y=nltsall["years"], width=nltsall["count"],
             align="center", label="All fields",
             color=all_clr, alpha=0.75,
             )
    ax1.barh(y=nltsttl["years"], width=nltsttl["count"],
             align="center", label="In title",
             color=ttl_clr, alpha=0.75,
             )
    ax2.barh(y=rpall["years"], width=rpall["count"],
             align="center", label="All fields",
             color=all_clr, alpha=0.75,
             )
    ax2.barh(y=rpttl["years"], width=rpttl["count"],
             align="center", label="In title",
             color=ttl_clr, alpha=0.75,
             )

    # prettify figure
    for i, ax in enumerate(fig.axes):
        ax.tick_params(size=8, labelsize=tiklabfs)
        ax.set_ylabel("Year", fontsize=axlabfs, labelpad=10)
        leg = ax.legend(loc="lower right",
                        bbox_to_anchor=legbba[i])
        for txt in leg.get_texts():
            txt.set_size(tiklabfs)
        ax.set_xlim(0., 100.)
        ax.set_xticks(np.arange(0., 101., 20.))
        if i == 0:
            ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(1980., 2018.)
        ax.text(0.08, 0.95,
                splabs[i],
                ha="right", va="center",
                fontsize=axlabfs, fontweight="bold", family="sans-serif",
                usetex=False,
                transform=ax.transAxes
                )
    ax2.tick_params(left="off", right="on",
                    labelleft="off", labelright="on")
    ax2.yaxis.set_label_position("right")
    fig.text(0.5, 0.05,
             "Number of publications",
             ha="center", va="center",
             fontsize=axlabfs,
             )

    # add Mendeley search terms as text boxes in top right
    ax1.text(0.75, 0.275,
            "``Nonlinear time series''",
            fontsize=tiklabfs, fontweight="bold",
            ha="right", va="center", color="IndianRed",
            transform=ax1.transAxes)
    ax2.text(0.26, 0.275,
            "``Recurrence plots''",
            fontsize=tiklabfs, fontweight="bold",
            ha="left", va="center", color="IndianRed",
            transform=ax2.transAxes)

    # save figure
    FN = "../plots/" + __file__[2:-3] + ".pdf"
    fig.savefig(FN, rasterized=True, dpi=1200)
    print("figure saved to: %s" % FN)

