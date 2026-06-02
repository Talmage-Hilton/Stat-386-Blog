---
layout: post
title:  "Bingo, but Make It Statistical: Building a Card Generator in R and Python"
date: 2026-06-01
description: Why settle for someone else's bingo card? I built a randomized General Conference bingo card generator in R and Python. Here's exactly how it works.
image: "/assets/img/conference-bingo.jpg"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">G</span>eneral Conference bingo is a tradition in a lot of LDS households, but I kept running into the same recycled cards online. So naturally, I did what any statistician would do: I wrote code. This post breaks down how I built a fully randomized, customizable bingo card generator in R and Python.</p>
<p class="intro">Cover image source: <a href="https://newsroom.churchofjesuschrist.org/article/general-conference-related-games-and-activities-to-download">Church Newsroom</a></p>


### Introduction

Every April and October, members of The Church of Jesus Christ of Latter-day Saints tune in for General Conference, which includes two days of talks from Church leaders. It's a beloved tradition, and so is the unofficial companion activity: Conference Bingo. The problem is that most bingo cards floating around the internet are the same ones that have been recycled for years. Even worse, most of them are intended for young children. I wanted something fresh, randomized, and big enough that no two cards would ever look the same.

The solution? Build a generator. In this post I'll walk through how I created a randomized General Conference 11×11 bingo card generator using R, and then show how you can do the same thing in Python.


### The Square Pool

The foundation of any bingo generator is a good pool of squares to draw from. The bigger the pool, the more unique each card can be. I put together a list of over 110 squares ranging from classic Conference moments to more niche observations:

{%- highlight r -%}
all_squares <- c(
  "Book of Mormon scripture",
  "Starting talk with joke",
  "Uchtdorf mentions flying",
  "Fly / bug around the speaker",
  "Voice crack",
  "Holy Ghost",
  "Talk makes you cry",
  "Something about AI",
  "Eyring hits the pulpit",
  # ... and 100+ more
)
{%- endhighlight -%}

The more squares you include, the more variety you get across cards. I ended up with 150 squares, which means a 11×11 card (with a FREE space in the center) draws from a pool that is plenty big enough to get a unique bingo sheet for each user.


### Randomizing the Board

With the pool defined, filling a card is straightforward. We sample 120 squares without replacement — 60 for before the FREE space and 60 for after — and slot them in:

{%- highlight r -%}
chosen <- sample(all_squares, 120)
board  <- c(chosen[1:60], "FREE", chosen[61:120])
{%- endhighlight -%}

Because `sample()` draws without replacement by default, no square appears twice on the same card. And since we're drawing randomly from a large pool, the chance of two cards looking identical is extremely small.


### Laying Out the Grid

The actual card is drawn using R's `grid` package, which gives fine-grained control over placement and styling. The layout is built on normalized 0–1 coordinates so everything scales cleanly regardless of page size.

{%- highlight r -%}
library(grid)

N        <- 11        # 11×11 grid
MARGIN   <- 0.02
HEADER_H <- 0.10
GRID_TOP <- 1 - MARGIN - HEADER_H
GRID_BOT <- MARGIN
GRID_L   <- MARGIN
GRID_R   <- 1 - MARGIN

cell_w <- (GRID_R - GRID_L) / N
cell_h <- (GRID_TOP - GRID_BOT) / N
{%- endhighlight -%}

Each cell is drawn as a rectangle with alternating background colors to make the grid easier to read, and the FREE space in the center gets its own distinct styling:

{%- highlight r -%}
bg <- if (free) COL_FREE_BG else if ((i + j) %% 2 == 0) COL_ALT_BG else COL_CELL_BG

grid.rect(
  x = cx, y = cy,
  width = cell_w, height = cell_h,
  gp = gpar(fill = bg, col = COL_GRID, lwd = 1.2),
  just = "centre"
)
{%- endhighlight -%}


### Fitting Text into Cells

This was the trickiest part. Bingo squares vary a lot in length — "Faith" fits easily in one line, but "Story about a random stranger from long ago" needs some help. I wrote a `fit_text()` function that tries different wrap widths and font sizes, picking the combination that fills the cell best without overflowing:

{%- highlight r -%}
fit_text <- function(label, max_width, max_height, min_size = 7, max_size = 12) {
  best <- list(size = min_size, text = label)
  
  for (wrap_w in seq(10, 25, by = 1)) {
    wrapped <- wrap_label(label, width = wrap_w)
    fontsize <- max_size
    
    repeat {
      tg <- textGrob(wrapped, gp = gpar(fontsize = fontsize, fontfamily = "sans"), just = "centre")
      w  <- convertWidth(grobWidth(tg), "npc", valueOnly = TRUE)
      h  <- convertHeight(grobHeight(tg), "npc", valueOnly = TRUE)
      
      if (w <= max_width && h <= max_height) {
        if (fontsize > best$size) best <- list(size = fontsize, text = wrapped)
        break
      }
      
      fontsize <- fontsize - 0.5
      if (fontsize < min_size) break
    }
  }
  return(best)
}
{%- endhighlight -%}

The logic is simple: try wrapping the text at a given width, check if it fits in the cell at the current font size, and if not, shrink the font and try again. Whatever combination produces the largest readable font that still fits wins.


### Generating Multiple Cards

The whole drawing routine is wrapped in a `draw_bingo_page()` function, and a `generate_bingo()` wrapper handles saving each card as its own PDF:

{%- highlight r -%}
generate_bingo <- function(n = 1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  for (i in seq_len(n)) {
    outfile <- paste0("conference_bingo", i, ".pdf")
    cairo_pdf(outfile, width = 11, height = 8.5, bg = COL_PAGE_BG)
    draw_bingo_page()
    dev.off()
    message("Saved: ", outfile)
  }
}

generate_bingo(n = 5)
{%- endhighlight -%}

Running this produces five uniquely randomized, print-ready bingo cards as landscape PDFs. The `seed` argument is there if you ever want to reproduce a specific set of cards.


### Doing It in Python

The same logic translates cleanly to Python using `matplotlib`. The setup is a bit more explicit, but the core idea — sample from a pool, lay out a grid, fit text into cells — is identical.

{%- highlight python -%}
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from textwrap import fill

all_squares = [
    "Book of Mormon scripture",
    "Starting talk with joke",
    "Uchtdorf mentions flying",
    "Voice crack",
    "Holy Ghost",
    "Talk makes you cry",
    "Something about AI",
    # ... add your full list here
]

def generate_card(all_squares):
    chosen = random.sample(all_squares, 24)
    return chosen[:12] + ["FREE"] + chosen[12:]

def draw_card(board, filename="bingo_card.pdf"):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    
    for idx, text in enumerate(board):
        row = idx // 5
        col = idx % 5
        
        color = "#1a3a5c" if text == "FREE" else ("#eef3fa" if (row + col) % 2 == 0 else "white")
        rect = patches.Rectangle((col, 4 - row), 1, 1, linewidth=1.2,
                                   edgecolor="#1a3a5c", facecolor=color)
        ax.add_patch(rect)
        
        wrapped = fill(text, width=15)
        fontcolor = "#f5c842" if text == "FREE" else "#1a1a1a"
        fontsize = 7 if len(text) > 30 else 8
        
        ax.text(col + 0.5, 4.5 - row, wrapped, ha="center", va="center",
                fontsize=fontsize, color=fontcolor, wrap=True,
                multialignment="center")
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

for i in range(1, 6):
    board = generate_card(all_squares)
    draw_card(board, filename=f"bingo_card_{i}.pdf")
    print(f"Saved bingo_card_{i}.pdf")
{%- endhighlight -%}

This produces the same result as the R version — five randomized, printable bingo cards — using only standard Python libraries.


### Final Thoughts

This was one of those projects that started as a small convenience and turned into something genuinely fun to build. The text-fitting problem in particular was more interesting than I expected — it's a surprisingly non-trivial constraint satisfaction problem at small scale. If you want to make your own version, the easiest place to start is just swapping out `all_squares` for whatever theme you want. The rest of the code is plug-and-play.

You can find the full R and Python source code on my [GitHub]([LINK]). Happy Conference weekend!