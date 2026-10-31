"""Convert the IEEEtran-layout manuscript into Elsevier's elsarticle layout.

Usage (from the package root):
    python3 figure_sources/to_elsarticle.py react_manuscript_ESR_IEEElayout.tex react_manuscript_ESR.tex
Body text, figures, tables, and references are left untouched; only the class, front matter,
keywords, appendix switch, and bibliography style change.
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()

t = t.replace(r"\documentclass[journal]{IEEEtran}",
              "\\documentclass[5p,times,twocolumn]{elsarticle}\n"
              "\\journal{Energy Strategy Reviews}")
t = t.replace("\\usepackage{balance}\n", "")
t = t.replace("\\usepackage{cite}\n", "")          # elsarticle loads natbib
t = t.replace("\\usepackage[T1]{fontenc}\n",
              "\\usepackage[T1]{fontenc}\n\\renewcommand{\\ttdefault}{pcr}  % Type 1 (Courier) typewriter font (e-mails, URLs)\n")
t = t.replace("\\usepackage{hyperref}", "\\usepackage[switch]{lineno}  % line numbers for review\n\\usepackage{hyperref}")

title = re.search(r"\\title\{(.*?)\}\n\\author", t, re.S).group(1)
a0 = t.index(r"\title{")
a1 = t.index(r"\maketitle") + len(r"\maketitle")
abs0 = t.index(r"\begin{abstract}")
abs1 = t.index(r"\end{abstract}") + len(r"\end{abstract}")
kw = re.search(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}", t, re.S)
keywords = [k.strip() for k in kw.group(1).split(",")]

front = r"""\begin{frontmatter}

\title{%s}

\author[mep]{Prashant Pant\corref{cor1}}
\ead{prashant.pant@tum.de}
\author[ox]{Satyam Mohla}
\ead{satyam.mohla@dtc.ox.ac.uk}
\author[mep,eet]{Reinaldo Tonkoski Jr.}
\ead{reinaldo.tonkoski@tum.de}
\author[mep,ens]{Thomas Hamacher}
\ead{thomas.hamacher@tum.de}
\cortext[cor1]{Corresponding author.}

\affiliation[mep]{organization={Munich Institute of Integrated Materials, Energy and Process
  Engineering (MEP), Technical University of Munich (TUM)}, city={Munich}, country={Germany}}
\affiliation[ens]{organization={Chair of Renewable and Sustainable Energy Systems, TUM School of
  Engineering and Design, Technical University of Munich (TUM)}, city={Munich}, country={Germany}}
\affiliation[eet]{organization={Chair of Electric Power Transmission and Distribution, TUM School
  of Engineering and Design, Technical University of Munich (TUM)}, city={Munich},
  country={Germany}}
\affiliation[ox]{organization={University of Oxford}, city={Oxford}, country={United Kingdom}}

%s

\begin{keyword}
%s
\end{keyword}

\end{frontmatter}

\linenumbers
""" % (title, t[abs0:abs1], " \\sep\n".join(keywords))

t = t[:a0] + front + t[abs1:]
t = t[:t.index(r"\begin{IEEEkeywords}")] + t[t.index(r"\end{IEEEkeywords}") + len(r"\end{IEEEkeywords}"):]
t = t.replace("\\appendices\n", "\\appendix\n")
# elsarticle's \ref to an appendix section already prints "Appendix X"
t = re.sub(r"Appendices~\\ref\{([^}]*)\}\s+and\s+\\ref\{", r"\\ref{\1} and \\ref{", t)
t = t.replace("Appendix~\\ref{", "\\ref{")
t = t.replace("\\balance\n", "")
t = t.replace(r"\bibliographystyle{IEEEtran}", r"\bibliographystyle{elsarticle-num}")
t = t.replace("%% REACT manuscript -- Energy Strategy Reviews submission version (revised figures)",
              "%% REACT manuscript -- Energy Strategy Reviews submission version (elsarticle layout)")
open(dst, "w", encoding="utf-8").write(t)
print("written", dst)
