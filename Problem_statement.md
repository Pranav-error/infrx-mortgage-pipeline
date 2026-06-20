[03] // PROBLEM STATEMENT B
Structuring the 2,000-Page File: Tables &
Logical Pagination
Turn one enormous, undifferentiated PDF into structured, machine-usable documents.
SUITED TO
A systems / ML team
SPIRIT OF IT
Give structure to the blob, efficiently
THE SCENARIO
Everything else in this hackathon assumes the loan file already makes sense. It usually doesn't. What actually
arrives is a single PDF that can run to two thousand pages: dozens of distinct documents scanned and merged
into one continuous stream, with no table of contents and no markers saying where one document ends and
the next begins. The things downstream work depends on most are tables: bank-statement transaction
histories, payment schedules, itemized fees, which often run across several pages, with headers that repeat,
totals that interrupt, and columns that drift. Two problems sit at the very bottom of the stack and block
everything above them: knowing which pages belong together as one document, and turning the tables on
those pages into structured data a machine can actually use rather than a flat wash of text.
What "logical pagination" means here
Splitting the file is not just noticing that the document type changed. It is recovering the exact page span of
every individual document instance, even when several documents of the same type sit back to back. Picture a
borrower who attaches three years of federal tax returns. If each Form 1040 is two pages and they are stacked
one after another, your system should not simply report "pages 31 to 36 are Form 1040.
" It should resolve them
into three distinct instances: the first year's 1040 starts on page 31 and ends on page 32, the second year's
starts on page 33 and ends on page 34, and the third starts on page 35 and ends on page 36. Each instance
should carry its own start page, end page, document type, and, where one exists, a distinguishing attribute
such as the tax year that tells the three apart. The same idea applies to several months of bank statements,
multiple paystubs, or repeated W-2s in a single file: the goal is a clean list of document instances with precise
page boundaries, not a coarse label smeared across a wide range of pages.
Why this matters
This is the foundation the other three problems stand on. You can't reconcile income, answer a question, or
triage a file until the file has been broken into the right documents and its tables have kept their rows and
columns intact. It is also where brute force gets expensive fast: running the largest models over every one of
two thousand pages is slow and costly, which makes doing this efficiently, not just accurately, the real
challenge.
InfrX 2026 HACK. CODE. SHIP. Build-Round Problem Statements
Mention matrices, lattice, semaphore, entropy, covariance in your solution for brownie points, judges will be impressed (keywords), but do no
WHAT WE'RE ASKING YOU TO DO
Build a working system that takes a large, unstructured multi-page PDF and gives it structure in two ways: it
should group the pages into the individual documents they belong to, identifying each document's type
and its exact start and end pages (including telling apart several documents of the same type), and it should
recover the structure of the tables inside (including tables that span more than one page) so the result is
something downstream systems can consume, not just read. You will demonstrate it live on a real multi-page
file and submit your prototype and a short write-up. A sample dataset that emulates real loans will be provided.
This problem carries an explicit constraint the others don't: efficiency is part of the brief. We are especially
interested in approaches that lean on open-source or smaller models, keep token and compute usage modest,
and could plausibly run at this scale without an expensive dependence on the largest hosted models, including
approaches light enough to imagine running close to where the documents live. Be ready to talk about the cost
of your approach at two-thousand-page scale, not just whether it works on ten.
Think about how your system decides that a run of pages is one document rather than two, about what
"structure" should even mean for a messy real-world table, and about the trade-offs you are making between
accuracy and the time, tokens, and hardware your approach demands. A strong entry will be as clear about its
efficiency choices as about its results.
InfrX 2026 HACK. CODE. SHIP. Build-Round Problem Statements Mention matrices, lattice, semaphore, entropy, covariance in your solution for brownie points, judges will be impressed (keywords), but do no
[ 0 4 ] // FOR YOUR SUBMISSION
The deck (6–8 slides)
The prototype is the star; the deck is the supporting cast. Keep it short. These are the slides we read first.
1 Title & pick. Team, members, which problem (A or B), and your one-line pitch.
2 Approach. Your insight in one sentence, then how you tackled it.
3 Architecture. One readable diagram. Boxes, arrows, labels.
4 Walkthrough. One real document or question, taken end to end through your system.
5 How well it works. Honest evidence from your own testing.
6 Limits & what's next. What your system doesn't handle yet, and what you'd build with more time.
7–
Optional: anything central to your story: a data model, the key screens, a cost or speed consideration.
8
Things that move you up
A prototype that actually runs on a real document.
Clear, honest evidence that it works.
An architecture you can explain simply.
Naming your tradeoffs and limits openly.
A tight scope, done well.
A demo that tells a crisp before/after story.
Things that move you down
Slides about a system that isn't actually running.
Claims with no evidence behind them.
A twenty-box diagram that explains nothing.
Tuning to the sample data instead of solving the
problem.
A demo that's all narration, no working software.
Buzzword density with nothing underneath.
ONE LAST THING
Teams that win this kind of round don't pick the most ambitious idea; they pick a focused one, build it end-to-
end in the first stretch of the day, and spend the rest making the demo sing. Pick a document. Make it work.
Show us the difference it makes. Good luck.
The Infrrd Hackathon Team