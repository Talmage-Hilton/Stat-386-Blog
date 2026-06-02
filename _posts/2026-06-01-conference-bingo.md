---
layout: post
title:  "Bingo, but Make It Statistical: Building a Card Generator in R and Python"
date: 2026-06-01
description: Why settle for someone else's bingo card? I built a randomized General Conference bingo card generator in R and Python. Here's exactly how it works.
image: "/assets/img/conference_bingo.jpg"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">G</span>eneral Conference bingo is a tradition in a lot of LDS households, but I kept running into the same recycled cards online. So naturally, I did what any data scientist would do: I wrote code. This post breaks down how I built a fully randomized, customizable bingo card generator in R and Python.</p>
<p class="intro">Cover image source: <a href="https://stock.adobe.com/search?k=%22bingo+balls%22">Adobe Stock</a></p>


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
import argparse
import random
import textwrap

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ALL_SQUARES = [
    "Book of Mormon scripture",
    "Starting talk with joke",
    "Uchtdorf mentions flying",
    "Voice crack",
    "Holy Ghost",
    "Talk makes you cry",
    "Something about AI",
    # ... add your full list here
]

# ── Colour palette ────────────────────────────────────────────────────────────
COL_HEADER_BG = (0.102, 0.227, 0.361)   # #1a3a5c  dark navy
COL_HEADER_FG = (1.0,   1.0,   1.0)     # white
COL_TITLE_FG  = (0.961, 0.784, 0.259)   # #f5c842  gold
COL_FREE_BG   = (0.102, 0.227, 0.361)   # navy
COL_FREE_FG   = (0.961, 0.784, 0.259)   # gold
COL_CELL_BG   = (1.0,   1.0,   1.0)     # white
COL_ALT_BG    = (0.933, 0.953, 0.980)   # #eef3fa  light blue tint
COL_CELL_FG   = (0.102, 0.102, 0.102)   # near-black
COL_GRID      = (0.102, 0.227, 0.361)   # navy
COL_PAGE_BG   = (0.941, 0.957, 0.984)   # #f0f4fb  light page bg

# ── Text fitting helper ───────────────────────────────────────────────────────
def fit_text(label: str, cell_w_pt: float, cell_h_pt: float,
             min_size: float = 7.0, max_size: float = 11.0):
    """
    Return (font_size, wrapped_text) that fills the cell as large as possible
    without overflowing. Tries multiple wrap widths (in chars) and font sizes.

    ReportLab canvas uses points; approximate character/line dimensions are
    estimated from the font metrics of Helvetica (sans-serif, like R's 'sans').
    """
    CHAR_W_RATIO = 0.55   # avg char width ≈ 0.55 × font_size  (points)
    LINE_H_RATIO = 1.25   # line height   ≈ 1.25 × font_size

    PAD_X = 0.10
    PAD_Y = 0.10
    avail_w = cell_w_pt * (1 - PAD_X)
    avail_h = cell_h_pt * (1 - PAD_Y)

    best_size = min_size
    best_text = textwrap.fill(label, width=15, break_long_words=False, break_on_hyphens=False)

    for wrap_chars in range(8, 28):
        wrapped = textwrap.fill(label, width=wrap_chars, break_long_words=False, break_on_hyphens=False)
        lines   = wrapped.split("\n")
        n_lines = len(lines)
        max_line_chars = max(len(l) for l in lines)

        for fsize in [s * 0.5 for s in range(int(max_size * 2), int(min_size * 2) - 1, -1)]:
            text_w = max_line_chars * CHAR_W_RATIO * fsize
            text_h = n_lines       * LINE_H_RATIO  * fsize

            if text_w <= avail_w and text_h <= avail_h:
                if fsize > best_size:
                    best_size = fsize
                    best_text = wrapped
                break

    return best_size, best_text


# ── Draw one bingo card onto the current canvas page ─────────────────────────
def draw_bingo_page(c: canvas.Canvas, page_w: float, page_h: float):
    N = 11   # 11 × 11 grid  (matches the R version)

    MARGIN   = 0.02 * page_w
    HEADER_H = 0.10 * page_h

    grid_l = MARGIN
    grid_r = page_w - MARGIN
    grid_t = page_h - MARGIN - HEADER_H
    grid_b = MARGIN

    cell_w = (grid_r - grid_l) / N
    cell_h = (grid_t - grid_b) / N

    # ── Page background ───────────────────────────────────────────────────────
    c.setFillColorRGB(*COL_PAGE_BG)
    c.rect(0, 0, page_w, page_h, stroke=0, fill=1)

    # ── Header band ──────────────────────────────────────────────────────────
    hdr_y = grid_t   # bottom of header band = top of grid
    c.setFillColorRGB(*COL_HEADER_BG)
    c.rect(MARGIN, hdr_y, page_w - 2 * MARGIN, HEADER_H, stroke=0, fill=1)

    # "CONFERENCE!" letters in white boxes
    letters = list("CONFERENCE!")
    for j, letter in enumerate(letters):
        cx = grid_l + cell_w * j + cell_w / 2
        cy = hdr_y + HEADER_H / 2   # vertical centre of header

        box_w = cell_w * 0.72
        box_h = HEADER_H * 0.52

        # white box
        c.setFillColorRGB(*COL_HEADER_FG)
        c.setStrokeColorRGB(*COL_HEADER_BG)
        c.setLineWidth(1.5)
        c.rect(cx - box_w / 2, cy - box_h / 2, box_w, box_h, stroke=1, fill=1)

        # navy letter
        c.setFillColorRGB(*COL_HEADER_BG)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(cx, cy - 6, letter)

    # ── Sample squares ────────────────────────────────────────────────────────
    chosen = random.sample(ALL_SQUARES, 120)
    board  = chosen[:60] + ["FREE"] + chosen[60:]

    # ── Grid cells ────────────────────────────────────────────────────────────
    for i in range(N):        # row, top → bottom
        for j in range(N):    # col, left → right
            idx  = i * N + j
            txt  = board[idx]
            free = txt == "FREE"

            cx = grid_l + j * cell_w          # left edge of cell
            cy = grid_t - (i + 1) * cell_h    # bottom edge of cell

            if free:
                bg = COL_FREE_BG
            elif (i + j) % 2 == 0:
                bg = COL_ALT_BG
            else:
                bg = COL_CELL_BG

            # Cell fill
            c.setFillColorRGB(*bg)
            c.setStrokeColorRGB(*COL_GRID)
            c.setLineWidth(1.2)
            c.rect(cx, cy, cell_w, cell_h, stroke=1, fill=1)

            # Cell text
            if free:
                c.setFillColorRGB(*COL_FREE_FG)
                c.setFont("Helvetica-Bold", 10)
                c.drawCentredString(cx + cell_w / 2, cy + cell_h / 2 - 4, "FREE")
            else:
                fsize, wrapped = fit_text(txt, cell_w, cell_h)
                lines   = wrapped.split("\n")
                n_lines = len(lines)
                line_h  = fsize * 1.25

                c.setFillColorRGB(*COL_CELL_FG)
                c.setFont("Helvetica", fsize)

                # Vertically centre the text block
                total_h = n_lines * line_h
                start_y = cy + cell_h / 2 + total_h / 2 - fsize * 0.85

                for k, line in enumerate(lines):
                    y = start_y - k * line_h
                    c.drawCentredString(cx + cell_w / 2, y, line)

    # ── Outer border ──────────────────────────────────────────────────────────
    c.setStrokeColorRGB(*COL_GRID)
    c.setLineWidth(2.5)
    c.rect(grid_l, grid_b,
           grid_r - grid_l,
           grid_t - grid_b,
           stroke=1, fill=0)


# ── Main generator ────────────────────────────────────────────────────────────
def generate_bingo(n: int = 5, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    page_w, page_h = landscape(letter)   # 792 × 612 pt  (11 × 8.5 in)

    for i in range(1, n + 1):
        outfile = f"conference_bingo{i}.pdf"
        c = canvas.Canvas(outfile, pagesize=(page_w, page_h))
        draw_bingo_page(c, page_w, page_h)
        c.save()
        print(f"Saved: {outfile}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Change these two values directly and run the script
    N_CARDS = 5
    SEED    = None   # set to an integer for reproducible cards, e.g. 42
    generate_bingo(n=N_CARDS, seed=SEED)
{%- endhighlight -%}

This produces the same result as the R version — five randomized, printable bingo cards — using only standard Python libraries.


### Final Thoughts

This was one of those projects that started as a small convenience and turned into something genuinely fun to build. The text-fitting problem in particular was more interesting than I expected — it's a surprisingly non-trivial constraint satisfaction problem at small scale. If you want to make your own version, the easiest place to start is just swapping out `all_squares` for whatever theme you want. The rest of the code is plug-and-play.

You can find the full R and Python source code on my [GitHub]("https://github.com/Talmage-Hilton"). Happy Conference weekend!